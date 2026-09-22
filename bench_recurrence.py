"""Resident-tensor recurrence microbenchmarks (OpenSpec 3.2).

Benchmarks the canonical recurrence for both alpha layouts
(``[B,T,H,1]`` scalar and ``[B,T,H,D]`` channel-wise), every production
head dimension derived from ``PRODUCTION_MODELS``, sequence lengths
16/128/512, and forward/backward/combined execution.

Design notes:
- Resident tensors only: inputs are created once per case (fixed seed)
  outside all timed regions; no dataloader, tokenization, download,
  compilation, or diagnostics inside warmup/measured windows.
- Warmup samples are executed but never recorded; only measured samples
  contribute to p50/p95/mean/std.
- GPU timing uses CUDA events with device synchronization; CPU timing
  uses wall time. CUDA-only paths are skipped with an explicit reason.
- This module never accepts or mutates ``ExperimentConfig``/budgets, so
  ``max_optimizer_steps`` cannot be changed here.
"""

from __future__ import annotations

import math
import time
from statistics import mean, median, pstdev
from typing import Any, Literal

import torch

from config import PRODUCTION_MODELS
from energy import EnergyProvider, normalized_energy
from recurrence import (
    BackendUnavailableError,
    RecurrenceOptions,
    backend_metadata,
    ensure_backend_available,
    run_recurrence,
)

RECURRENCE_SEQ_LENS: tuple[int, ...] = (16, 128, 512)
ALPHA_LAYOUTS: tuple[str, ...] = ("scalar", "channel-wise")
RECURRENCE_MODES: tuple[str, ...] = ("forward", "backward", "combined")
DEFAULT_WARMUP = 25
DEFAULT_MEASURED = 200

AlphaLayout = Literal["scalar", "channel-wise"]
RecurrenceMode = Literal["forward", "backward", "combined"]


def production_head_dims() -> dict[str, int]:
    """Derive production head dims from PRODUCTION_MODELS (never hardcoded)."""
    return {scale: model.d_head for scale, model in PRODUCTION_MODELS.items()}


def summary(values: list[float]) -> dict[str, float | int | None]:
    """p50/p95/mean/std summary; warmup values must never be passed here."""
    if not values:
        return {"count": 0, "mean": None, "std": None, "p50": None, "p95": None}
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "p50": median(values),
        "p95": ordered[p95_index],
    }


def _timed_call(work: Any, device: torch.device) -> float:
    """Time one callable in seconds; CUDA events on GPU, wall time on CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        work()
        end_event.record()
        end_event.synchronize()
        return start_event.elapsed_time(end_event) / 1000.0
    started = time.perf_counter()
    work()
    return time.perf_counter() - started


def _peak_memory(device: torch.device) -> dict[str, Any]:
    if device.type == "cuda":
        return {
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
        }
    return {
        "peak_allocated_gib": 0.0,
        "peak_reserved_gib": 0.0,
        "note": "CPU device has no CUDA allocated/reserved counters",
    }


def _make_inputs(
    *, batch: int, heads: int, d_head: int, seq_len: int,
    layout: AlphaLayout, device: torch.device, requires_grad: bool, seed: int,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(
            shape, generator=generator, device=device, dtype=torch.float32,
            requires_grad=requires_grad,
        )
    alpha_width = d_head if layout == "channel-wise" else 1
    q, k, v = rand(batch, seq_len, heads, d_head), rand(batch, seq_len, heads, d_head), rand(batch, seq_len, heads, d_head)
    alpha = torch.sigmoid(rand(batch, seq_len, heads, alpha_width)).detach().requires_grad_(requires_grad)
    beta = torch.sigmoid(rand(batch, seq_len, heads, 1)).detach().requires_grad_(requires_grad)
    return q, k, v, alpha, beta


def benchmark_case(
    *,
    backend: str = "reference",
    d_head: int,
    seq_len: int,
    alpha_layout: AlphaLayout = "scalar",
    mode: RecurrenceMode = "combined",
    batch: int = 2,
    heads: int = 4,
    warmup: int = DEFAULT_WARMUP,
    measured: int = DEFAULT_MEASURED,
    device: torch.device | None = None,
    seed: int = 91_337,
) -> dict[str, Any]:
    """Benchmark one (backend, head-dim, seq-len, layout, mode) case.

    Returns measured-sample statistics plus peak memory; warmup is
    executed but excluded. Unavailable backends return
    ``status: unavailable`` with a reason and no fabricated samples.
    """
    if warmup < 0 or measured < 1:
        raise ValueError("warmup must be >= 0 and measured must be >= 1")
    if alpha_layout not in ALPHA_LAYOUTS:
        raise ValueError(f"Unknown alpha layout {alpha_layout!r}")
    if mode not in RECURRENCE_MODES:
        raise ValueError(f"Unknown recurrence mode {mode!r}")
    resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    options = RecurrenceOptions(backend=backend)  # type: ignore[arg-type]
    try:
        provenance = ensure_backend_available(options, device=resolved)
    except (BackendUnavailableError, ValueError) as exc:
        return {
            "status": "unavailable",
            "backend": backend,
            "resolved_backend": None,
            "reason": f"{type(exc).__name__}: {exc}",
            "backend_metadata": backend_metadata(options, device=resolved),
            "d_head": d_head, "seq_len": seq_len, "alpha_layout": alpha_layout, "mode": mode,
            "warmup": warmup, "measured": measured, "samples": [],
            "timing_seconds": summary([]),
        }
    needs_grad = mode in ("backward", "combined")
    q, k, v, alpha, beta = _make_inputs(
        batch=batch, heads=heads, d_head=d_head, seq_len=seq_len,
        layout=alpha_layout, device=resolved, requires_grad=needs_grad, seed=seed,
    )
    tensors = (q, k, v, alpha, beta)

    def forward_once() -> torch.Tensor:
        output, _ = run_recurrence(q, k, v, alpha, beta, None, options=options)
        return output

    def combined_once() -> None:
        output = forward_once()
        output.square().mean().backward()
        for tensor in tensors:
            if tensor.grad is not None:
                tensor.grad = None

    def backward_once() -> None:
        output = forward_once()
        output.square().mean().backward()

    def clear_grads() -> None:
        for tensor in tensors:
            if tensor.grad is not None:
                tensor.grad = None

    # Warmup (untimed, excluded from statistics).
    with torch.no_grad() if mode == "forward" else torch.enable_grad():
        for _ in range(warmup):
            if mode == "forward":
                with torch.no_grad():
                    forward_once()
            elif mode == "combined":
                combined_once()
            else:
                backward_once()
                clear_grads()
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
        torch.cuda.reset_peak_memory_stats(resolved)

    provider = EnergyProvider.create(resolved)
    samples: list[float] = []
    with provider.region(f"recurrence_{backend}_d{d_head}_t{seq_len}_{alpha_layout}_{mode}") as region:
        if mode == "forward":
            with torch.no_grad():
                for _ in range(measured):
                    samples.append(_timed_call(forward_once, resolved))
        elif mode == "combined":
            for _ in range(measured):
                samples.append(_timed_call(combined_once, resolved))
        else:
            for _ in range(measured):
                output = forward_once()  # rebuild autograd graph outside timing

                def _backward(out: torch.Tensor = output) -> None:
                    out.square().mean().backward()

                samples.append(_timed_call(_backward, resolved))
                clear_grads()
    assert region.result is not None
    stats = summary(samples)
    p50 = stats["p50"] or 0.0
    elements = batch * seq_len * heads * d_head
    return {
        "status": "measured",
        "backend": backend,
        "resolved_backend": provenance.get("resolved_backend"),
        "backend_metadata": backend_metadata(options, device=resolved),
        "d_head": d_head, "seq_len": seq_len, "alpha_layout": alpha_layout, "mode": mode,
        "batch": batch, "heads": heads,
        "warmup": warmup, "measured": measured, "samples": samples,
        "timing_seconds": stats,
        "throughput": {
            "elements_per_second": elements / p50 if p50 else 0.0,
            "tokens_per_second": (batch * seq_len) / p50 if p50 else 0.0,
        },
        "memory": _peak_memory(resolved),
        "energy": normalized_energy(
            region.result, tokens=batch * seq_len * measured, unit="recurrence_tokens",
        ),
        "device": {"device_type": resolved.type},
    }


def benchmark_recurrence_matrix(
    *,
    backends: tuple[str, ...] = ("reference", "cuda-fused"),
    seq_lens: tuple[int, ...] = RECURRENCE_SEQ_LENS,
    modes: tuple[RecurrenceMode, ...] = ("forward", "backward", "combined"),  # type: ignore[assignment]
    warmup: int = DEFAULT_WARMUP,
    measured: int = DEFAULT_MEASURED,
    device: torch.device | None = None,
    batch: int = 2,
    heads: int = 4,
) -> dict[str, Any]:
    """Benchmark every production head dim x seq len x layout x mode x backend.

    cuda-fused cases that cannot load record ``status: unavailable`` with
    the loader reason instead of raising. No ``ExperimentConfig`` is
    touched, so ``max_optimizer_steps`` is never mutated here.
    """
    resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    head_dims = production_head_dims()
    cases: list[dict[str, Any]] = []
    for scale, d_head in sorted(head_dims.items()):
        for seq_len in seq_lens:
            for layout in ALPHA_LAYOUTS:
                for mode in modes:
                    for backend in backends:
                        case = benchmark_case(
                            backend=backend, d_head=d_head, seq_len=seq_len,
                            alpha_layout=layout, mode=mode, batch=batch,  # type: ignore[arg-type]
                            heads=heads, warmup=warmup, measured=measured,
                            device=resolved,
                        )
                        case["scale"] = scale
                        cases.append(case)
    # Paired speedups where both backends measured the same shape/mode.
    speedups: list[dict[str, Any]] = []
    by_key: dict[tuple, dict[str, dict[str, Any]]] = {}
    for case in cases:
        key = (case["scale"], case["seq_len"], case["alpha_layout"], case["mode"])
        by_key.setdefault(key, {})[case["backend"]] = case
    for key, pair in sorted(by_key.items()):
        ref, fused = pair.get("reference"), pair.get("cuda-fused")
        if ref is None or fused is None:
            continue
        if ref.get("status") != "measured" or fused.get("status") != "measured":
            speedups.append({
                "scale": key[0], "seq_len": key[1], "alpha_layout": key[2], "mode": key[3],
                "speedup_vs_reference_p50": None,
                "reason": str(fused.get("reason", "cuda-fused unavailable")),
            })
            continue
        ref_p50 = (ref["timing_seconds"]["p50"] or 0.0)
        fused_p50 = (fused["timing_seconds"]["p50"] or 0.0)
        speedups.append({
            "scale": key[0], "seq_len": key[1], "alpha_layout": key[2], "mode": key[3],
            "speedup_vs_reference_p50": (ref_p50 / fused_p50) if fused_p50 else None,
            "reference_p50_seconds": ref_p50, "fused_p50_seconds": fused_p50,
        })
    return {
        "schema_version": 1,
        "kind": "recurrence_microbenchmark",
        "settings": {
            "backends": list(backends), "seq_lens": list(seq_lens),
            "alpha_layouts": list(ALPHA_LAYOUTS), "modes": list(modes),
            "warmup": warmup, "measured": measured, "batch": batch, "heads": heads,
            "device_type": resolved.type, "head_dims": head_dims,
        },
        "cases": cases,
        "speedups": speedups,
        "interpretation": (
            "Warmup samples are excluded from all statistics. "
            "Compare speedups only for identical scale/seq/layout/mode/device."
        ),
    }
