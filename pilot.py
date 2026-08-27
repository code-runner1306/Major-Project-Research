"""V100-only measurement pilot for choosing recurrence and training optimizations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import math
from statistics import mean, median, pstdev
import time
from typing import Any, Iterable
from uuid import uuid4

import torch
import torch.optim as optim

from artifacts import atomic_write_json, fingerprint
from config import ExperimentConfig
from controls import construct_model
from dataset import tinystories_dataloader
from energy import EnergyProvider, normalized_energy
from orchestration import recurrence_options_for
from preflight import require_production_cuda
from recurrence import backend_metadata, ensure_backend_available
from train import seed_everything


@dataclass(frozen=True)
class PilotSettings:
    scales: tuple[str, ...] = ("55M", "100M")
    variants: tuple[str, ...] = ("scalar", "channel-wise")
    warmup_steps: int = 25
    measured_steps: int = 200
    data_fed: bool = False
    collect_trace: bool = False


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "p50": None, "p95": None}
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(values), "mean": mean(values), "std": pstdev(values),
        "p50": median(values), "p95": ordered[p95_index],
    }


def _memory(device: torch.device) -> dict[str, float]:
    return {
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
    }


def _resident_batches(profile: ExperimentConfig, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device=device).manual_seed(91_337)
    shape = (profile.budget.batch_size, profile.budget.context_length)
    return [
        (
            torch.randint(0, profile.vocab_size, shape, generator=generator, device=device),
            torch.randint(0, profile.vocab_size, shape, generator=generator, device=device),
        )
        for _ in range(profile.budget.grad_accum_steps)
    ]


def _step(
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    batches = list(batches)
    for tokens, targets in batches:
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            _, loss = model(tokens, targets)
            assert loss is not None
            scaled_loss = loss / len(batches)
        scaler.scale(scaled_loss).backward()
        loss_sum += float(loss.item())
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return loss_sum


def _timed_step(work: Any, device: torch.device) -> tuple[float, float, float]:
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start_event.record()
    loss = float(work())
    end_event.record()
    end_event.synchronize()
    return start_event.elapsed_time(end_event) / 1000.0, time.perf_counter() - started, loss


def _next_data_batches(
    iterator: Any, loader: Iterable[tuple[torch.Tensor, torch.Tensor]], *, count: int, device: torch.device,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], Any]:
    """Fetch and transfer one accumulated optimizer batch, restarting safely."""

    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(count):
        try:
            tokens, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            try:
                tokens, targets = next(iterator)
            except StopIteration as exc:
                raise RuntimeError("TinyStories pilot loader produced no batches") from exc
        batches.append((tokens.to(device, non_blocking=True), targets.to(device, non_blocking=True)))
    return batches, iterator


def _pilot_output_dir(profile: ExperimentConfig, settings: PilotSettings, timestamp: str) -> Path:
    identity = {"profile": profile.to_manifest_dict(), "settings": asdict(settings), "timestamp": timestamp}
    return profile.output_dir / "pilots" / f"pilot_{timestamp}_{fingerprint(identity)[:12]}_{uuid4().hex[:8]}"


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# V100 training pilot", "", "This is a profiling artifact, not a research training result.", ""]
    lines.append(f"- Backend: `{payload['recurrence']['resolved_backend']}`")
    lines.append(f"- Device: `{payload['preflight']['gpu_name']}` / `{payload['preflight']['compute_capability']}`")
    lines.append(f"- Warmup / measured steps: {payload['settings']['warmup_steps']} / {payload['settings']['measured_steps']}")
    lines.extend(["", "| Scale | Variant | GPU p50 s/step | Wall p50 s/step | Tokens/sec | Peak GiB |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for row in payload["runs"]:
        gpu = row["gpu_seconds"]["p50"]
        wall = row["wall_seconds"]["p50"]
        lines.append(
            f"| {row['scale']} | {row['variant']} | {gpu:.4f} | {wall:.4f} | "
            f"{row['tokens_per_second']:.1f} | {row['memory']['peak_allocated_gib']:.2f} |"
        )
    lines.extend(["", "Interpret wall-minus-GPU time as host/data/launch overhead. Compare scalar and channel-wise only on this same artifact/device."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_v100_pilot(profile: ExperimentConfig, settings: PilotSettings) -> dict[str, Any]:
    """Measure fixed-shape real optimizer work without unlocking production training."""

    if profile.run_mode != "production":
        raise ValueError("--pilot requires the production profile and a CUDA V100; use --smoke for CPU verification")
    if settings.warmup_steps < 1 or settings.measured_steps < 1:
        raise ValueError("Pilot warmup and measured step counts must both be positive")
    unknown = set(settings.scales) - set(profile.scales)
    if unknown:
        raise ValueError(f"Unknown pilot scales: {sorted(unknown)}")
    unknown_variants = set(settings.variants) - set(profile.variants)
    if unknown_variants:
        raise ValueError(f"Unknown pilot variants: {sorted(unknown_variants)}")

    preflight = require_production_cuda(recurrence_backend=profile.recurrence_backend)
    device = torch.device("cuda")
    options = recurrence_options_for(profile)
    recurrence = ensure_backend_available(options, device=device)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = _pilot_output_dir(profile, settings, timestamp)
    output.mkdir(parents=True, exist_ok=False)
    provider = EnergyProvider.create(device)
    runs: list[dict[str, Any]] = []
    trace: dict[str, Any] = {"requested": settings.collect_trace, "status": "not_requested"}

    for scale in settings.scales:
        for variant in settings.variants:
            seed_everything(42)
            model, _, parameter = construct_model(
                profile.model_for(scale), variant, profile.vocab_size, recurrence_options=options,
            )
            model.to(device)
            optimizer = optim.AdamW(model.parameters(), lr=profile.learning_rate, weight_decay=profile.weight_decay,
                                    betas=(profile.beta1, profile.beta2))
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            resident = _resident_batches(profile, device)
            for _ in range(settings.warmup_steps):
                _step(model, optimizer, scaler, resident, device)
            if settings.collect_trace and trace["status"] == "not_requested":
                try:
                    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                    with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=True) as profiler:
                        _step(model, optimizer, scaler, resident, device)
                    trace_path = output / "first_optimizer_step_trace.json"
                    profiler.export_chrome_trace(str(trace_path))
                    trace = {"requested": True, "status": "collected", "path": trace_path.name}
                except Exception as exc:
                    trace = {"requested": True, "status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            gpu_seconds: list[float] = []
            wall_seconds: list[float] = []
            losses: list[float] = []
            with provider.region(f"pilot_{scale}_{variant}") as energy_region:
                for _ in range(settings.measured_steps):
                    gpu, wall, loss = _timed_step(lambda: _step(model, optimizer, scaler, resident, device), device)
                    gpu_seconds.append(gpu)
                    wall_seconds.append(wall)
                    losses.append(loss)
            assert energy_region.result is not None
            tokens_per_step = profile.budget.batch_size * profile.budget.grad_accum_steps * profile.budget.context_length
            wall_p50 = float(_summary(wall_seconds)["p50"] or 0.0)
            runs.append({
                "scale": scale, "variant": variant, "parameter": parameter,
                "gpu_seconds": _summary(gpu_seconds), "wall_seconds": _summary(wall_seconds),
                "host_or_launch_overhead_seconds": _summary([max(0.0, wall - gpu) for gpu, wall in zip(gpu_seconds, wall_seconds)]),
                "loss": _summary(losses), "memory": _memory(device),
                "tokens_per_step": tokens_per_step,
                "tokens_per_second": tokens_per_step / wall_p50 if wall_p50 else 0.0,
                "energy": normalized_energy(energy_region.result, tokens=tokens_per_step * settings.measured_steps / 1_000_000, unit="million_training_tokens"),
            })
            if settings.data_fed:
                # Construct/download and establish the iterator outside all measured
                # windows.  This paired window deliberately includes fetch,
                # tokenization, H2D copy, and the same optimizer work.
                loader = tinystories_dataloader(profile, split="train", data_seed=142, run_dir=output)
                iterator = iter(loader)

                def data_fed_step() -> float:
                    nonlocal iterator
                    batches, iterator = _next_data_batches(
                        iterator, loader, count=profile.budget.grad_accum_steps, device=device,
                    )
                    return _step(model, optimizer, scaler, batches, device)

                for _ in range(settings.warmup_steps):
                    data_fed_step()
                fed_gpu_seconds: list[float] = []
                fed_wall_seconds: list[float] = []
                for _ in range(settings.measured_steps):
                    gpu, wall, _ = _timed_step(data_fed_step, device)
                    fed_gpu_seconds.append(gpu)
                    fed_wall_seconds.append(wall)
                runs[-1]["tiny_stories_fed"] = {
                    "gpu_seconds": _summary(fed_gpu_seconds),
                    "wall_seconds": _summary(fed_wall_seconds),
                    "host_input_or_launch_overhead_seconds": _summary([
                        max(0.0, wall - gpu) for gpu, wall in zip(fed_gpu_seconds, fed_wall_seconds)
                    ]),
                    "tokens_per_second": tokens_per_step / float(_summary(fed_wall_seconds)["p50"] or 0.0),
                    "comparison": "Compare this wall time to resident-tensor wall time; the difference includes CPU input work and host-to-device transfer.",
                }
            del model, optimizer, scaler
            torch.cuda.empty_cache()

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "v100_training_pilot",
        "settings": asdict(settings),
        "preflight": preflight.to_dict(),
        "recurrence": {**recurrence, "resolved_backend": options.backend},
        "runs": runs,
        "data_fed": {
            "requested": settings.data_fed,
            "status": "paired_per_run" if settings.data_fed else "not_requested",
            "note": "TinyStories loader construction/download is excluded; each paired sample includes fetch, tokenization, H2D transfer, and optimizer work." if settings.data_fed else None,
        },
        "trace": trace,
        "interpretation": "Use GPU versus wall time to locate launch/host overhead; this artifact does not select a training budget.",
    }
    atomic_write_json(output / "pilot_result.json", payload)
    _write_markdown(output / "pilot_summary.md", payload)
    return {"status": "complete", "output_dir": str(output), "runs": len(runs)}
