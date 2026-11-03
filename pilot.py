"""V100-only measurement pilot for choosing recurrence and training optimizations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import gc
import multiprocessing as mp
import os
from pathlib import Path
import math
from statistics import mean, median, pstdev
import time
import traceback
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
from retrieval import RetrievalDataset, retrieval_dataloader
from resource_telemetry import DeviceResourceCapture
from train import seed_everything


@dataclass(frozen=True)
class PilotSettings:
    scales: tuple[str, ...] = ("55M", "100M")
    variants: tuple[str, ...] = ("scalar", "channel-wise")
    warmup_steps: int = 25
    measured_steps: int = 200
    data_fed: bool = False
    collect_trace: bool = False
    planned_optimizer_steps: int = 1_000
    seed_counts: dict[str, int] = field(default_factory=dict)
    parallel_scales: tuple[str, ...] = ("10M", "30M")
    parallel_worker_counts: tuple[int, ...] = (2, 3)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "p50": None, "p95": None}
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(values), "mean": mean(values), "std": pstdev(values),
        "p50": median(values), "p95": ordered[p95_index],
    }


def _pilot_progress(message: str, *, started_at: float | None = None) -> None:
    """Print a timestamped pilot status line immediately for interactive runs/logs."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    elapsed = f" +{time.perf_counter() - started_at:,.1f}s" if started_at is not None else ""
    print(f"[{timestamp}{elapsed}] [pilot] {message}", flush=True)


def _progress_interval(total: int) -> int:
    """Emit about ten progress updates per loop, including tiny pilot loops."""
    return max(1, math.ceil(total / 10))


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


def _pilot_data_loader(
    profile: ExperimentConfig, family: str, *, data_dir: Path, seed: int,
):
    if family == "tinystories":
        return tinystories_dataloader(profile, split="train", data_seed=seed, run_dir=data_dir)
    dataset = RetrievalDataset(
        task=family, examples=profile.retrieval.train_examples, config=profile.retrieval,
        seed=seed, split="train",
    )
    return retrieval_dataloader(dataset, profile.budget.batch_size)


def _parallel_worker_process(
    profile: ExperimentConfig, family: str, scale: str, variant: str, worker_index: int,
    warmup_steps: int, measured_steps: int, options: Any, data_dir: str,
    start_event: Any, result_queue: Any, trial_label: str, started_at: float,
) -> None:
    """Run one parallel pilot worker in its own process, like production."""
    try:
        device = torch.device("cuda")
        seed_everything(42 + worker_index)
        backbone = profile.model_for(scale)
        if family != "tinystories":
            backbone = replace(backbone, attention_layout="pure_gdn")
        vocab = profile.vocab_size if family == "tinystories" else profile.retrieval.vocabulary_size
        model, _, _ = construct_model(backbone, variant, vocab, recurrence_options=options)
        model.to(device)
        optimizer = optim.AdamW(
            model.parameters(), lr=profile.learning_rate, weight_decay=profile.weight_decay,
            betas=(profile.beta1, profile.beta2),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        loader = _pilot_data_loader(
            profile, family, data_dir=Path(data_dir) / family / f"worker_{worker_index}",
            seed=142 + worker_index,
        )
        stream = torch.cuda.Stream(device=device)
        iterator = iter(loader)

        def step() -> float:
            nonlocal iterator
            batches, iterator = _next_data_batches(
                iterator, loader, count=profile.budget.grad_accum_steps, device=device,
            )
            return _step(model, optimizer, scaler, batches, device)

        process = None
        try:
            import psutil  # type: ignore[import-not-found]
            process = psutil.Process()
        except Exception:
            process = None
        with torch.cuda.stream(stream):
            for _ in range(warmup_steps):
                step()
            stream.synchronize()
            result_queue.put({"worker_index": worker_index, "status": "ready"})
            if not start_event.wait(timeout=180):
                raise TimeoutError("parallel pilot start gate timed out")
            cpu_before = sum(process.cpu_times()[:2]) if process is not None else None
            started = time.perf_counter()
            events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
            losses: list[float] = []
            interval = _progress_interval(measured_steps)
            for measured_index in range(1, measured_steps + 1):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record(stream)
                losses.append(step())
                end.record(stream)
                events.append((begin, end))
                if measured_index % interval == 0 or measured_index == measured_steps:
                    _pilot_progress(
                        f"parallel {trial_label}: process worker {worker_index + 1} measured "
                        f"{measured_index}/{measured_steps} steps", started_at=started_at,
                    )
            stream.synchronize()
            elapsed = time.perf_counter() - started
            cpu_after = sum(process.cpu_times()[:2]) if process is not None else None
        gpu_seconds = [begin.elapsed_time(end) / 1000.0 for begin, end in events]
        cpu_percent = None
        if cpu_before is not None and cpu_after is not None and elapsed > 0:
            cpu_percent = max(0.0, (cpu_after - cpu_before) * 100.0 / elapsed)
        result_queue.put({
            "worker_index": worker_index, "status": "complete", "wall_seconds": elapsed,
            "gpu_seconds": sum(gpu_seconds), "step_wall_seconds": elapsed / measured_steps,
            "step_gpu_seconds": sum(gpu_seconds) / measured_steps,
            "mean_loss": sum(losses) / len(losses), "process_cpu_percent": cpu_percent,
        })
    except BaseException as exc:
        result_queue.put({
            "worker_index": worker_index, "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
        })


def _run_parallel_qualification(
    profile: ExperimentConfig, settings: PilotSettings, baseline_runs: list[dict[str, Any]],
    output: Path, device: torch.device, options: Any, *, started_at: float,
) -> dict[str, Any]:
    families = ("tinystories", "mqar", "overwrite")
    results: dict[str, Any] = {}
    logical_cpus = os.cpu_count() or 1
    variants = ("scalar", "channel-wise")
    total_trials = len(settings.parallel_scales) * len(families) * len(settings.parallel_worker_counts)
    trial_number = 0
    _pilot_progress(
        f"parallel qualification started: {len(settings.parallel_scales)} scales × {len(families)} datasets × "
        f"{len(settings.parallel_worker_counts)} worker counts ({total_trials} requested trials)",
        started_at=started_at,
    )
    for scale in settings.parallel_scales:
        scale_result: dict[str, Any] = {}
        base_rows = [row for row in baseline_runs if row["scale"] == scale]
        for family in families:
            family_rows = [row for row in base_rows if f"{family}_fed" in row]
            baseline_step_times = [
                float(row[f"{family}_fed"]["wall_seconds"]["p50"])
                for row in family_rows
                if row["variant"] in variants
                if row[f"{family}_fed"]["wall_seconds"].get("p50")
            ]
            if not baseline_step_times:
                _pilot_progress(f"parallel {scale}/{family}: skipped; no single-worker data-fed baseline", started_at=started_at)
                scale_result[family] = {"status": "unavailable", "reason": "No data-fed single-worker baseline"}
                continue
            peak_reserved = max((
                float(row.get(f"{family}_fed", {}).get("memory", row["memory"])["peak_reserved_gib"])
                for row in family_rows
            ), default=0.0)
            baseline_throughput = sum(1.0 / value for value in baseline_step_times) / len(baseline_step_times)
            trials: list[dict[str, Any]] = []
            for worker_count in settings.parallel_worker_counts:
                trial_number += 1
                trial_label = f"{scale}/{family}, {worker_count} workers (trial {trial_number}/{total_trials})"
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                free_gib, total_gib = free_bytes / (1024 ** 3), total_bytes / (1024 ** 3)
                required_gib = peak_reserved * worker_count
                usable_gib = free_gib * 0.85
                if required_gib > usable_gib:
                    _pilot_progress(
                        f"parallel {trial_label}: skipped by VRAM gate; estimated {required_gib:.2f} GiB required, "
                        f"{free_gib:.2f} GiB free ({usable_gib:.2f} GiB after safety margin)",
                        started_at=started_at,
                    )
                    trials.append({
                        "workers": worker_count, "status": "skipped_vram_capacity",
                        "estimated_required_vram_gib": required_gib,
                        "free_vram_gib": free_gib, "usable_free_vram_gib": usable_gib,
                        "safety_fraction": 0.85,
                    })
                    continue

                trial: dict[str, Any] = {
                    "workers": worker_count, "status": "running",
                    "estimated_required_vram_gib": required_gib,
                    "free_vram_before_trial_gib": free_gib,
                }
                try:
                    _pilot_progress(
                        f"parallel {trial_label}: launching isolated worker processes",
                        started_at=started_at,
                    )
                    context = mp.get_context("spawn")
                    start_event = context.Event()
                    result_queue = context.Queue()
                    processes = []
                    worker_results = []
                    try:
                        for worker_index in range(worker_count):
                            process = context.Process(
                                target=_parallel_worker_process,
                                args=(
                                    profile, family, scale, variants[worker_index % len(variants)],
                                    worker_index, settings.warmup_steps, settings.measured_steps,
                                    options, str(output / "parallel_data"), start_event, result_queue,
                                    trial_label, started_at,
                                ),
                                name=f"pilot-{family}-{scale}-worker-{worker_index + 1}",
                            )
                            process.start()
                            processes.append(process)

                        ready_workers: set[int] = set()
                        ready_deadline = time.monotonic() + 180
                        while len(ready_workers) < worker_count:
                            remaining = ready_deadline - time.monotonic()
                            if remaining <= 0:
                                raise TimeoutError("parallel pilot workers did not finish warmup within 180 seconds")
                            message = result_queue.get(timeout=remaining)
                            if message.get("status") == "failed":
                                raise RuntimeError(
                                    f"worker {message.get('worker_index')} failed during setup: "
                                    f"{message.get('reason')}\n{message.get('traceback', '')}"
                                )
                            if message.get("status") == "ready":
                                ready_workers.add(int(message["worker_index"]))

                        _pilot_progress(f"parallel {trial_label}: all processes warmed up; starting measured steps", started_at=started_at)
                        with DeviceResourceCapture(device) as capture:
                            start_event.set()
                            while len(worker_results) < worker_count:
                                message = result_queue.get(timeout=300)
                                if message.get("status") == "failed":
                                    raise RuntimeError(
                                        f"worker {message.get('worker_index')} failed: {message.get('reason')}\n"
                                        f"{message.get('traceback', '')}"
                                    )
                                if message.get("status") == "complete":
                                    worker_results.append(message)
                            for process in processes:
                                process.join(timeout=10)
                                if process.is_alive():
                                    raise TimeoutError(f"pilot worker {process.name} did not exit after reporting results")
                                if process.exitcode != 0:
                                    raise RuntimeError(f"pilot worker {process.name} exited with code {process.exitcode}")
                    finally:
                        start_event.set()
                        for process in processes:
                            if process.is_alive():
                                process.terminate()
                            process.join(timeout=10)
                        result_queue.close()
                        result_queue.join_thread()
                    parallel_wall = max(row["wall_seconds"] for row in worker_results)
                    aggregate_steps_per_second = worker_count * settings.measured_steps / parallel_wall
                    serial_expected_wall = worker_count * settings.measured_steps / baseline_throughput
                    speedup = serial_expected_wall / parallel_wall
                    activity = capture.result["activity_summary"] if capture.result else {}
                    child_cpu_values = [
                        float(row["process_cpu_percent"]) for row in worker_results
                        if isinstance(row.get("process_cpu_percent"), (int, float))
                    ]
                    process_cpu = sum(child_cpu_values) if child_cpu_values else None
                    system_cpu = activity.get("system_cpu_percent", {}).get("mean")
                    cpu_frequency = activity.get("cpu_frequency_current_mhz", {}).get("mean")
                    gpu_util = activity.get("gpu_utilization_percent", {}).get("mean")
                    cpu_saturated = bool(
                        (isinstance(process_cpu, (int, float)) and process_cpu >= 85 * logical_cpus)
                        or (isinstance(system_cpu, (int, float)) and system_cpu >= 85)
                    )
                    trial.update({
                        "status": "complete", "worker_results": worker_results,
                        "parallel_wall_seconds": parallel_wall,
                        "aggregate_optimizer_steps_per_second": aggregate_steps_per_second,
                        "single_worker_baseline_steps_per_second": baseline_throughput,
                        "speedup_over_serial_equivalent": speedup,
                        "resource_capture": capture.result,
                        "process_cpu_scope": "sum of worker-process CPU utilization; may exceed 100%",
                        "mean_process_cpu_percent": process_cpu,
                        "mean_system_cpu_percent": system_cpu,
                        "mean_cpu_frequency_current_mhz": cpu_frequency,
                        "mean_gpu_utilization_percent": gpu_util,
                        "logical_cpu_count": logical_cpus,
                        "cpu_saturation_indicator": cpu_saturated,
                        "cpu_throttling_interpretation": (
                            "CPU saturation/contention indicator observed; thermal/frequency throttling is not directly measured."
                            if cpu_saturated else
                            "No CPU saturation indicator observed; thermal/frequency throttling is not directly measured."
                        ),
                    })
                    _pilot_progress(
                        f"parallel {trial_label}: complete in {parallel_wall:.1f}s; "
                        f"speedup {speedup:.2f}×, aggregate {aggregate_steps_per_second:.3f} steps/s, "
                        f"CPU process/system {process_cpu}%/{system_cpu}%, GPU {gpu_util}%",
                        started_at=started_at,
                    )
                except Exception as exc:
                    trial.update({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
                    _pilot_progress(f"parallel {trial_label}: FAILED: {type(exc).__name__}: {exc}", started_at=started_at)
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()
                trials.append(trial)

            completed = [row for row in trials if row["status"] == "complete"]
            best_throughput = max(
                (row["aggregate_optimizer_steps_per_second"] for row in completed), default=0.0,
            )
            best = next((row for row in completed if row["aggregate_optimizer_steps_per_second"] == best_throughput), None)
            if best is not None and best_throughput >= baseline_throughput * 1.10:
                recommendation = int(best["workers"])
                recommended_speedup = float(best["speedup_over_serial_equivalent"])
                reason = "Highest measured aggregate throughput with at least 10% improvement over sequential execution."
            else:
                recommendation = 1
                recommended_speedup = 1.0
                reason = "No tested worker count improved aggregate throughput by at least 10%."
            scale_result[family] = {
                "status": "complete", "single_worker_baseline_steps_per_second": baseline_throughput,
                "single_worker_peak_reserved_vram_gib": peak_reserved,
                "trials": trials, "recommended_workers_for_pilot_workload": recommendation,
                "recommended_speedup_over_sequential": recommended_speedup,
                "recommendation_reason": reason,
            }
        results[scale] = scale_result
    return {
        "requested": True,
        "worker_counts": list(settings.parallel_worker_counts),
        "families": list(families),
        "scales": results,
        "worker_execution_mode": "spawned processes matching the production launcher",
        "interpretation": "Pilot-only guidance for the measured device/workload; it does not alter production execution settings.",
    }


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
    lines.extend(["", "See `training_time_estimate.md` for per-dataset extrapolations and 1/3/5-seed totals."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _training_time_estimate(payload: dict[str, Any], settings: PilotSettings) -> dict[str, Any]:
    families = ("tinystories", "mqar", "overwrite")
    variants = tuple(settings.variants)
    by_scale: dict[str, dict[str, Any]] = {}
    for scale in settings.scales:
        rows = [row for row in payload["runs"] if row["scale"] == scale]
        family_rows: dict[str, Any] = {}
        for family in families:
            variant_hours: dict[str, float] = {}
            per_step: dict[str, float | None] = {}
            sources: set[str] = set()
            for variant in variants:
                run = next((item for item in rows if item["variant"] == variant), None)
                if run is None:
                    per_step[variant] = None
                    continue
                fed = run.get(f"{family}_fed")
                summary = fed.get("wall_seconds", {}) if isinstance(fed, dict) else {}
                p50 = summary.get("p50") if isinstance(summary, dict) else None
                if isinstance(p50, (int, float)) and p50 > 0:
                    seconds = float(p50)
                    sources.add("data_fed_p50")
                else:
                    seconds = float(run["wall_seconds"]["p50"])
                    sources.add("resident_compute_proxy")
                per_step[variant] = seconds
                variant_hours[variant] = seconds * settings.planned_optimizer_steps / 3600.0
            paired_hours = sum(variant_hours.values()) if len(variant_hours) == len(variants) else None
            family_rows[family] = {
                "seconds_per_optimizer_step_by_variant": per_step,
                "training_hours_per_variant_seed": variant_hours,
                "training_hours_per_seed_all_variants": paired_hours,
                "estimate_source": sorted(sources),
            }
        configured_seeds = settings.seed_counts.get(scale, 3)
        per_seed_hours = sum(
            item["training_hours_per_seed_all_variants"] or 0.0 for item in family_rows.values()
        )
        by_scale[scale] = {
            "seed_count": configured_seeds,
            "datasets": family_rows,
            "training_hours_per_seed_all_datasets": per_seed_hours,
            "training_hours_at_configured_seed_count": per_seed_hours * configured_seeds,
        }

    uniform_seed_totals = {
        str(seed_count): sum(
            scale_data["training_hours_per_seed_all_datasets"] * seed_count
            for scale_data in by_scale.values()
        )
        for seed_count in (1, 3, 5)
    }
    configured_total = sum(
        item["training_hours_at_configured_seed_count"] for item in by_scale.values()
    )
    parallel_training = payload.get("parallel_training", {"requested": False})
    configured_parallel_total = None
    uniform_parallel_totals = None
    if parallel_training.get("requested"):
        def plan_hours(seed_count_for_scale: Any) -> float:
            total = 0.0
            for scale, scale_data in by_scale.items():
                family_parallel = parallel_training.get("scales", {}).get(scale, {})
                seeds_for_scale = seed_count_for_scale(scale_data)
                for family, family_data in scale_data["datasets"].items():
                    family_total = family_data["training_hours_per_seed_all_variants"] or 0.0
                    speedup = family_parallel.get(family, {}).get("recommended_speedup_over_sequential", 1.0)
                    total += family_total * seeds_for_scale / max(float(speedup or 1.0), 1.0)
            return total

        configured_parallel_total = plan_hours(lambda scale_data: scale_data["seed_count"])
        uniform_parallel_totals = {
            str(seed_count): plan_hours(lambda _scale_data: seed_count)
            for seed_count in (1, 3, 5)
        }
    return {
        "planned_optimizer_steps_per_variant_seed": settings.planned_optimizer_steps,
        "families": list(families),
        "variants": list(variants),
        "seed_counts_by_scale": {scale: item["seed_count"] for scale, item in by_scale.items()},
        "per_scale": by_scale,
        "uniform_seed_count_totals_hours": uniform_seed_totals,
        "uniform_seed_count_parallel_totals_hours": uniform_parallel_totals,
        "configured_seed_plan_total_hours": configured_total,
        "configured_seed_plan_estimated_hours_with_parallelism": configured_parallel_total,
        "parallel_training": parallel_training,
        "scope": "training-only; excludes post-training evaluation and benchmarks",
        "caveats": [
            "Times extrapolate short pilot p50 optimizer-step measurements to the planned fixed step budget.",
            "Training totals include both configured gate variants for every dataset and seed.",
            "Measured data-fed timings include input fetch and host-to-device work for each family when available.",
            "Resident-compute fallbacks omit dataset input overhead and are labeled as proxies.",
            "Pilot measurements use very few steps; treat estimates as planning ranges, not guarantees.",
            "Startup, checkpoint writes, final evaluation, efficiency benchmarks, and report generation are excluded.",
            "Seed-count totals are sequential training totals; parallel wall-time savings are reported separately from the bounded concurrency pilot.",
        ],
    }


def _write_training_time_estimate(path: Path, estimate: dict[str, Any]) -> None:
    lines = [
        "# Pilot-based training time estimate", "",
        f"Planned budget: **{estimate['planned_optimizer_steps_per_variant_seed']:,} optimizer steps per variant/seed**.",
        "Times below are training-only and include scalar plus channel-wise variants.", "",
        "## Per model, dataset, and seed", "",
        "| Model | Dataset | Scalar step p50 (s) | Channel-wise step p50 (s) | Scalar hours/seed | Channel-wise hours/seed | Both variants hours/seed | Timing source |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for scale, scale_data in estimate["per_scale"].items():
        for family, result in scale_data["datasets"].items():
            steps = result["seconds_per_optimizer_step_by_variant"]
            hours = result["training_hours_per_variant_seed"]
            lines.append(
                f"| {scale} | {family} | {steps.get('scalar') or 0:.2f} | {steps.get('channel-wise') or 0:.2f} | "
                f"{hours.get('scalar', 0):.2f} | {hours.get('channel-wise', 0):.2f} | "
                f"{result['training_hours_per_seed_all_variants'] or 0:.2f} | {', '.join(result['estimate_source'])} |"
            )
        lines.append(
            f"| **{scale} total** | all datasets | — | — | — | — | "
            f"{scale_data['training_hours_per_seed_all_datasets']:.2f} h/seed | "
            f"{scale_data['seed_count']} configured seed(s): {scale_data['training_hours_at_configured_seed_count']:.2f} h |"
        )
    lines.extend(["", "## Overall totals across selected model sizes", "", "| Seeds per model size | Sequential training hours | Estimated hours using measured parallel speedups | Sequential days, one GPU |", "| ---: | ---: | ---: | ---: |"])
    for seed_count, hours in estimate["uniform_seed_count_totals_hours"].items():
        parallel_hours = estimate.get("uniform_seed_count_parallel_totals_hours")
        parallel_value = f"{parallel_hours[seed_count]:.1f}" if parallel_hours is not None else "—"
        lines.append(f"| {seed_count} | {hours:.1f} | {parallel_value} | {hours / 24:.1f} |")
    lines.extend([
        "", "## Configured seed plan", "",
        "| Model | Seeds | Estimated training hours |", "| --- | ---: | ---: |",
    ])
    for scale, scale_data in estimate["per_scale"].items():
        lines.append(
            f"| {scale} | {scale_data['seed_count']} | {scale_data['training_hours_at_configured_seed_count']:.1f} |"
        )
    lines.append(f"| **Total** | — | **{estimate['configured_seed_plan_total_hours']:.1f}** |")
    parallel_total = estimate.get("configured_seed_plan_estimated_hours_with_parallelism")
    if parallel_total is not None:
        savings = estimate["configured_seed_plan_total_hours"] - parallel_total
        lines.append(
            f"\nUsing measured pilot speedups as an idealized schedule gives **{parallel_total:.1f} hours** "
            f"for the configured seed plan (about **{savings:.1f} hours saved**). This excludes evaluation and assumes the pilot speedups hold across full runs."
        )
    parallel = estimate.get("parallel_training", {})
    lines.extend(["", "## Low-tier parallel-training pilot", ""])
    if not parallel.get("requested"):
        lines.append(f"Parallel qualification was not run: {parallel.get('reason', 'not requested')}.")
    else:
        lines.append("These trials use data-fed short optimizer workloads. VRAM-unsafe worker counts are skipped before model allocation.")
        lines.extend([
            "", "| Model | Dataset | Recommended workers | Workers tested (status / speedup / process CPU / system CPU / GPU utilization) |",
            "| --- | --- | ---: | --- |",
        ])
        for scale, family_results in parallel.get("scales", {}).items():
            for family, result in family_results.items():
                if result.get("status") != "complete":
                    lines.append(f"| {scale} | {family} | — | {result.get('reason', result.get('status'))} |")
                    continue
                trial_notes = []
                for trial in result.get("trials", []):
                    workers = trial.get("workers")
                    if trial.get("status") == "complete":
                        trial_notes.append(
                            f"{workers}: speedup {trial.get('speedup_over_serial_equivalent', 0):.2f}x; "
                            f"CPU proc/system {trial.get('mean_process_cpu_percent')}%/"
                            f"{trial.get('mean_system_cpu_percent')}%; CPU clock {trial.get('mean_cpu_frequency_current_mhz')} MHz; "
                            f"GPU {trial.get('mean_gpu_utilization_percent')}%"
                        )
                    else:
                        trial_notes.append(f"{workers}: {trial.get('status')} ({trial.get('estimated_required_vram_gib', 0):.1f} GiB estimated)")
                lines.append(
                    f"| {scale} | {family} | {result.get('recommended_workers_for_pilot_workload')} | "
                    f"{'<br>'.join(trial_notes)} |"
                )
        lines.extend([
            "", "CPU process utilization may exceed 100% when multiple cores are busy. High CPU/system utilization with weak throughput gain indicates likely CPU/input contention.",
            "Current CPU frequency is sampled when psutil exposes it, but thermal throttle flags/frequency caps are not directly read; this is a utilization-based contention check, not a definitive thermal-throttling diagnosis. Recommendations apply only to the tested GPU and data pipeline.",
        ])
    lines.extend(["", "## Estimate limits", ""])
    lines.extend(f"- {caveat}" for caveat in estimate["caveats"])
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
    unknown_parallel = set(settings.parallel_scales) - set(profile.scales)
    if unknown_parallel:
        raise ValueError(f"Unknown parallel pilot scales: {sorted(unknown_parallel)}")
    if any(worker_count < 2 for worker_count in settings.parallel_worker_counts):
        raise ValueError("Parallel pilot worker counts must be at least 2")
    unknown_variants = set(settings.variants) - set(profile.variants)
    if unknown_variants:
        raise ValueError(f"Unknown pilot variants: {sorted(unknown_variants)}")

    started_at = time.perf_counter()
    _pilot_progress(
        f"starting; scales={','.join(settings.scales)}, variants={','.join(settings.variants)}, "
        f"warmup={settings.warmup_steps}, measured_steps={settings.measured_steps}, data_fed={settings.data_fed}",
        started_at=started_at,
    )
    preflight = require_production_cuda(recurrence_backend=profile.recurrence_backend)
    device = torch.device("cuda")
    options = recurrence_options_for(profile)
    recurrence = ensure_backend_available(options, device=device)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = _pilot_output_dir(profile, settings, timestamp)
    output.mkdir(parents=True, exist_ok=False)
    _pilot_progress(f"CUDA preflight passed; output directory: {output}", started_at=started_at)
    provider = EnergyProvider.create(device)
    runs: list[dict[str, Any]] = []
    trace: dict[str, Any] = {"requested": settings.collect_trace, "status": "not_requested"}

    total_model_trials = len(settings.scales) * len(settings.variants)
    model_trial = 0
    for scale in settings.scales:
        for variant in settings.variants:
            model_trial += 1
            label = f"{scale}/{variant} (model trial {model_trial}/{total_model_trials})"
            _pilot_progress(f"{label}: constructing resident-batch model and optimizer", started_at=started_at)
            seed_everything(42)
            model, _, parameter = construct_model(
                profile.model_for(scale), variant, profile.vocab_size, recurrence_options=options,
            )
            model.to(device)
            optimizer = optim.AdamW(model.parameters(), lr=profile.learning_rate, weight_decay=profile.weight_decay,
                                    betas=(profile.beta1, profile.beta2))
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            resident = _resident_batches(profile, device)
            _pilot_progress(f"{label}: resident-batch warmup started ({settings.warmup_steps} steps)", started_at=started_at)
            for warmup_index in range(1, settings.warmup_steps + 1):
                _step(model, optimizer, scaler, resident, device)
                if warmup_index % _progress_interval(settings.warmup_steps) == 0 or warmup_index == settings.warmup_steps:
                    _pilot_progress(f"{label}: resident-batch warmup {warmup_index}/{settings.warmup_steps}", started_at=started_at)
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
            _pilot_progress(f"{label}: resident-batch measured run started ({settings.measured_steps} steps)", started_at=started_at)
            with provider.region(f"pilot_{scale}_{variant}") as energy_region:
                interval = _progress_interval(settings.measured_steps)
                for measured_index in range(1, settings.measured_steps + 1):
                    gpu, wall, loss = _timed_step(lambda: _step(model, optimizer, scaler, resident, device), device)
                    gpu_seconds.append(gpu)
                    wall_seconds.append(wall)
                    losses.append(loss)
                    if measured_index % interval == 0 or measured_index == settings.measured_steps:
                        _pilot_progress(
                            f"{label}: resident-batch measured {measured_index}/{settings.measured_steps}; "
                            f"latest loss={loss:.4f}, wall step={wall:.2f}s",
                            started_at=started_at,
                        )
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
            base_model_alive = True
            if settings.data_fed:
                for family in ("tinystories", "mqar", "overwrite"):
                    data_label = f"{label}, dataset={family}"
                    _pilot_progress(f"{data_label}: preparing data-fed model and loader", started_at=started_at)
                    # Loader setup is excluded; measured updates include CPU fetch or
                    # generation, host-to-device transfer, and optimizer work.
                    if family == "tinystories":
                        loader = tinystories_dataloader(
                            profile, split="train", data_seed=142,
                            run_dir=output / "pilot_data" / "tinystories",
                        )
                    else:
                        if base_model_alive:
                            del model, optimizer, scaler
                            base_model_alive = False
                            gc.collect()
                            torch.cuda.empty_cache()
                        backbone = replace(profile.model_for(scale), attention_layout="pure_gdn")
                        data_model, _, _ = construct_model(
                            backbone, variant, profile.retrieval.vocabulary_size,
                            recurrence_options=options,
                        )
                        data_model.to(device)
                        data_optimizer = optim.AdamW(
                            data_model.parameters(), lr=profile.learning_rate,
                            weight_decay=profile.weight_decay, betas=(profile.beta1, profile.beta2),
                        )
                        data_scaler = torch.amp.GradScaler("cuda", enabled=True)
                        dataset = RetrievalDataset(
                            task=family, examples=profile.retrieval.train_examples,
                            config=profile.retrieval, seed=142, split="train",
                        )
                        loader = retrieval_dataloader(dataset, profile.budget.batch_size)
                    iterator = iter(loader)

                    def data_fed_step() -> float:
                        nonlocal iterator
                        batches, iterator = _next_data_batches(
                            iterator, loader, count=profile.budget.grad_accum_steps, device=device,
                        )
                        active_model = model if family == "tinystories" else data_model
                        active_optimizer = optimizer if family == "tinystories" else data_optimizer
                        active_scaler = scaler if family == "tinystories" else data_scaler
                        return _step(active_model, active_optimizer, active_scaler, batches, device)

                    _pilot_progress(f"{data_label}: warmup started ({settings.warmup_steps} steps)", started_at=started_at)
                    torch.cuda.reset_peak_memory_stats(device)
                    for warmup_index in range(1, settings.warmup_steps + 1):
                        data_fed_step()
                        if warmup_index % _progress_interval(settings.warmup_steps) == 0 or warmup_index == settings.warmup_steps:
                            _pilot_progress(f"{data_label}: warmup {warmup_index}/{settings.warmup_steps}", started_at=started_at)
                    fed_gpu_seconds: list[float] = []
                    fed_wall_seconds: list[float] = []
                    _pilot_progress(f"{data_label}: measured run started ({settings.measured_steps} steps)", started_at=started_at)
                    interval = _progress_interval(settings.measured_steps)
                    for measured_index in range(1, settings.measured_steps + 1):
                        gpu, wall, _ = _timed_step(data_fed_step, device)
                        fed_gpu_seconds.append(gpu)
                        fed_wall_seconds.append(wall)
                        if measured_index % interval == 0 or measured_index == settings.measured_steps:
                            _pilot_progress(
                                f"{data_label}: measured {measured_index}/{settings.measured_steps}; wall step={wall:.2f}s",
                                started_at=started_at,
                            )
                    _pilot_progress(
                        f"{data_label}: complete; median wall step={_summary(fed_wall_seconds)['p50']:.2f}s",
                        started_at=started_at,
                    )
                    runs[-1][f"{family}_fed"] = {
                        "gpu_seconds": _summary(fed_gpu_seconds),
                        "wall_seconds": _summary(fed_wall_seconds),
                        "memory": _memory(device),
                        "host_input_or_launch_overhead_seconds": _summary([
                            max(0.0, wall - gpu) for gpu, wall in zip(fed_gpu_seconds, fed_wall_seconds)
                        ]),
                        "tokens_per_second": tokens_per_step / float(_summary(fed_wall_seconds)["p50"] or 0.0),
                        "comparison": "Includes dataset CPU fetch/generation, host-to-device transfer, and optimizer work after loader setup.",
                    }
                    if family != "tinystories":
                        del data_model, data_optimizer, data_scaler
                        gc.collect()
                        torch.cuda.empty_cache()
            if base_model_alive:
                del model, optimizer, scaler
            torch.cuda.empty_cache()

    parallel_training = (
        _run_parallel_qualification(profile, settings, runs, output, device, options, started_at=started_at)
        if settings.parallel_scales and settings.parallel_worker_counts and settings.data_fed else
        {
            "requested": False,
            "reason": (
                "Parallel qualification requires data-fed single-worker baselines."
                if settings.parallel_scales and settings.parallel_worker_counts and not settings.data_fed
                else "No parallel pilot scales/workers selected."
            ),
            "scales": {},
        }
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "v100_training_pilot",
        "settings": asdict(settings),
        "preflight": preflight.to_dict(),
        "recurrence": {**recurrence, "resolved_backend": options.backend},
        "runs": runs,
        "parallel_training": parallel_training,
        "data_fed": {
            "requested": settings.data_fed,
            "families": ["tinystories", "mqar", "overwrite"] if settings.data_fed else [],
            "status": "paired_per_family_and_run" if settings.data_fed else "not_requested",
            "note": "Loader setup is excluded; measured updates include batch fetch or generation, H2D transfer, and optimizer work." if settings.data_fed else None,
        },
        "trace": trace,
        "interpretation": "Use GPU versus wall time to locate launch/host overhead; this artifact does not select a training budget.",
    }
    estimate = _training_time_estimate(payload, settings)
    payload["training_time_estimate"] = estimate
    atomic_write_json(output / "pilot_result.json", payload)
    _write_markdown(output / "pilot_summary.md", payload)
    _write_training_time_estimate(output / "training_time_estimate.md", estimate)
    atomic_write_json(output / "training_time_estimate.json", estimate)
    parallel_worker_statuses = {
        scale: {
            family: {
                str(trial["workers"]): trial["status"]
                for trial in family_result.get("trials", [])
            }
            for family, family_result in family_results.items()
        }
        for scale, family_results in parallel_training.get("scales", {}).items()
    }
    _pilot_progress(
        f"complete in {time.perf_counter() - started_at:,.1f}s; reports: "
        f"{output / 'pilot_summary.md'} and {output / 'training_time_estimate.md'}",
        started_at=started_at,
    )
    return {
        "status": "complete", "output_dir": str(output), "runs": len(runs),
        "training_time_estimate": str(output / "training_time_estimate.md"),
        "parallel_worker_statuses": parallel_worker_statuses,
    }
