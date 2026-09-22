"""Canonical Gated DeltaNet recurrence backends and provenance helpers.

The eager implementation is intentionally kept as the numerical oracle.  CUDA
extensions are loaded lazily so CPU smoke tests never require a CUDA toolkit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

import torch


RecurrenceBackend = Literal["reference", "cuda-fused"]
StateAccumulatorDtype = Literal["input", "float32"]
RECURRENCE_SEMANTIC_VERSION = "gdn-recurrence-v1"


class BackendUnavailableError(RuntimeError):
    """Raised when an explicitly requested backend cannot be used safely."""


@dataclass(frozen=True)
class RecurrenceOptions:
    backend: RecurrenceBackend = "reference"
    input_dtype: Literal["autocast", "float32"] = "autocast"
    state_accumulator_dtype: StateAccumulatorDtype = "input"
    chunk_size: int = 64
    qualification_artifact: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[int, int, int, int]:
    if q.ndim != 4:
        raise ValueError("q must have shape [B,T,H,D]")
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, and v must have identical [B,T,H,D] shapes")
    batch, time, heads, width = q.shape
    if alpha.shape[:3] != (batch, time, heads) or alpha.shape[-1] not in (1, width):
        raise ValueError("alpha must have shape [B,T,H,1] or [B,T,H,D]")
    if beta.shape != (batch, time, heads, 1):
        raise ValueError("beta must have shape [B,T,H,1]")
    if any(tensor.device != q.device for tensor in (k, v, alpha, beta)):
        raise ValueError("q, k, v, alpha, and beta must share a device")
    expected_state = (batch, heads, width, width)
    if initial_state is not None and tuple(initial_state.shape) != expected_state:
        raise ValueError(f"Invalid recurrent state shape {tuple(initial_state.shape)}; expected {expected_state}")
    if initial_state is not None and initial_state.device != q.device:
        raise ValueError("Initial recurrent state must match Q device")
    return batch, time, heads, width


def _state_dtype(q: torch.Tensor, options: RecurrenceOptions) -> torch.dtype:
    input_dtype = torch.float32 if options.input_dtype == "float32" else q.dtype
    return torch.float32 if options.state_accumulator_dtype == "float32" else input_dtype


def _validate_options(options: RecurrenceOptions) -> None:
    if options.backend not in ("reference", "cuda-fused"):
        raise ValueError(f"Unknown recurrence backend {options.backend!r}")
    if options.input_dtype not in ("autocast", "float32"):
        raise ValueError(f"Unknown recurrence input dtype mode {options.input_dtype!r}")
    if options.state_accumulator_dtype not in ("input", "float32"):
        raise ValueError(f"Unknown state accumulator dtype mode {options.state_accumulator_dtype!r}")
    if options.chunk_size < 1:
        raise ValueError("recurrence chunk_size must be positive")


def reference_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    options: RecurrenceOptions = RecurrenceOptions(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical sequential update for normalized Q/K inputs.

    Scalar alpha broadcasts as ``[B,H,1,1]`` and channel alpha as
    ``[B,H,D,1]``.  The returned output follows the input dtype; the state
    follows the explicitly configured accumulator dtype.
    """

    _validate_options(options)
    batch, time, heads, width = _validate_inputs(q, k, v, alpha, beta, initial_state)
    state_dtype = _state_dtype(q, options)
    q_work, k_work, v_work = (tensor.to(state_dtype) for tensor in (q, k, v))
    alpha_work, beta_work = alpha.to(state_dtype), beta.to(state_dtype)
    expected_state = (batch, heads, width, width)
    state = (
        torch.zeros(expected_state, device=q.device, dtype=state_dtype)
        if initial_state is None
        else initial_state.to(state_dtype)
    )
    outputs: list[torch.Tensor] = []
    for step in range(time):
        q_t = q_work[:, step].unsqueeze(-1)
        k_t = k_work[:, step].unsqueeze(-1)
        v_t = v_work[:, step].unsqueeze(-1)
        alpha_t = alpha_work[:, step].unsqueeze(-1)
        beta_t = beta_work[:, step].unsqueeze(-1)
        error = v_t - state.matmul(k_t)
        state = alpha_t * state + beta_t * error.matmul(k_t.transpose(-1, -2))
        outputs.append(state.matmul(q_t).squeeze(-1))
    return torch.stack(outputs, dim=1).to(q.dtype), state


def _fused_load_reason() -> str:
    try:
        import gdn_cuda  # noqa: PLC0415 - lazy so CPU imports never need a toolkit

        ok, reason = gdn_cuda.load_extension()
        if ok:
            return "cuda-fused extension loaded but has not yet been qualified"
        return reason or "cuda-fused extension is unavailable"
    except Exception as exc:  # pragma: no cover - defensive; loader must not break CPU
        return f"cuda-fused extension is unavailable: {type(exc).__name__}: {exc}"


def backend_metadata(options: RecurrenceOptions, *, device: torch.device | None = None) -> dict[str, object]:
    """Return serializable provenance without forcing CUDA extension imports."""

    try:
        import gdn_cuda  # noqa: PLC0415 - pure-Python scaffold; CPU-safe

        extension_identity = gdn_cuda.build_identity()
        decode_version = gdn_cuda.DECODE_BACKEND_VERSION
        extension_version = gdn_cuda.EXTENSION_SEMANTIC_VERSION
        target_arch = gdn_cuda.TARGET_ARCH
    except Exception:
        extension_identity = "unavailable"
        decode_version = "decode-v1-fused-stub"
        extension_version = "gdn-cuda-v0-scaffold"
        target_arch = "sm_70"
    metadata: dict[str, object] = {
        "requested_backend": options.backend,
        "semantic_version": RECURRENCE_SEMANTIC_VERSION,
        "implementation_source_hash": sha256(Path(__file__).read_bytes()).hexdigest(),
        "extension_semantic_version": extension_version,
        "extension_build_identity": extension_identity,
        "target_architecture": target_arch,
        "decode_backend_version": decode_version,
        "input_dtype": options.input_dtype,
        "state_accumulator_dtype": options.state_accumulator_dtype,
        "chunk_size": options.chunk_size,
        "qualification_artifact": options.qualification_artifact,
        "qualification_hash": qualification_hash(options.qualification_artifact),
        "compiled_architectures": ["sm_70"] if options.backend == "cuda-fused" else [],
        "supported_input_dtypes": ["float32", "float16", "bfloat16"],
        "supported_alpha_layouts": ["[B,T,H,1]", "[B,T,H,D]"],
        "state_layout": "[B,H,D,D]",
        "chunk_policy": "sequential-reference" if options.backend == "reference" else "fixed-checkpoint-chunks",
        "resolved_backend": options.backend if options.backend == "reference" else None,
        "load_failure_reason": None,
    }
    if device is not None:
        metadata["device_type"] = device.type
    if options.backend == "cuda-fused":
        metadata["load_failure_reason"] = _fused_load_reason()
    return metadata


def qualification_hash(path: str | None) -> str | None:
    if not path:
        return None
    target = Path(path)
    if not target.is_file():
        return None
    return sha256(target.read_bytes()).hexdigest()


def ensure_backend_available(options: RecurrenceOptions, *, device: torch.device) -> dict[str, object]:
    _validate_options(options)
    metadata = backend_metadata(options, device=device)
    if options.backend == "reference":
        return metadata
    if device.type != "cuda":
        raise BackendUnavailableError("cuda-fused recurrence requires a CUDA device")
    if torch.cuda.get_device_capability(device) != (7, 0):
        raise BackendUnavailableError(
            "cuda-fused recurrence is currently qualified only for Volta sm_70; "
            f"found sm_{torch.cuda.get_device_capability(device)[0]}{torch.cuda.get_device_capability(device)[1]}"
        )
    if qualification_hash(options.qualification_artifact) is None:
        raise BackendUnavailableError(
            "cuda-fused production requires a matching qualification artifact "
            f"({options.qualification_artifact!r} missing or unreadable); "
            "qualify on the V100S first or run reference explicitly for diagnostics"
        )
    reason = str(metadata["load_failure_reason"])
    raise BackendUnavailableError(reason)


def run_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    options: RecurrenceOptions = RecurrenceOptions(),
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_options(options)
    if options.backend == "reference":
        return reference_recurrence(q, k, v, alpha, beta, initial_state, options=options)
    ensure_backend_available(options, device=q.device)
    import gdn_cuda  # noqa: PLC0415 - lazy so CPU import never needs a toolkit

    gdn_cuda.validate_fused_inputs(q, k, v, alpha, beta, initial_state, chunk_size=options.chunk_size)
    return gdn_cuda.fused_forward(q, k, v, alpha, beta, initial_state, chunk_size=options.chunk_size)
