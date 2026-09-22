"""CPU-safe scaffolding for the Volta (sm_70) fused GDN recurrence extension.

OpenSpec change ``optimize-gdn-recurrence`` section 4 (tasks 4.1-4.8), scaffolded
without the full V100 build: importing this package must NEVER require a CUDA
toolkit, nvcc, or a GPU. The real fused kernels land once the pilot/toolchain
is confirmed on the college V100S. Until then every numeric entry point raises
instead of returning placeholder values (no silent fallback, no faked numerics).

Linux-only note: compilation (see ``setup_gdn_cuda.py``) targets ``sm_70`` and
is supported on Linux with the pinned cu126 toolchain. Windows/CPU checkouts
use this package for loader reasons, validation messages, and provenance only.
"""

from __future__ import annotations

import hashlib
import importlib
import pathlib
import platform
import sys
from typing import Any

import torch

# 4.1: build target identity. The compiled extension must carry an sm_70 path.
TARGET_ARCH = "sm_70"
SM_70 = True
EXTENSION_SEMANTIC_VERSION = "gdn-cuda-v0-scaffold"
DECODE_BACKEND_VERSION = "decode-v1-fused-stub"
CHUNK_POLICY = "fixed-checkpoint-chunks"
COMPILED_MODULE_NAME = "gdn_cuda._fused"

# Production head dimensions (10M/30M d_head=40, 100M d_head=48, 55M d_head=56).
SUPPORTED_HEAD_DIMS = (40, 48, 56)
SUPPORTED_INPUT_DTYPES = ("float32", "float16", "bfloat16")
SUPPORTED_ACCUMULATOR_MODES = ("input", "float32")
SUPPORTED_ALPHA_LAYOUTS = ("[B,T,H,1]", "[B,T,H,D]")

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent


def _source_files() -> list[pathlib.Path]:
    files = [pathlib.Path(__file__)]
    csrc = _PACKAGE_ROOT / "csrc"
    if csrc.is_dir():
        files.extend(sorted(csrc.rglob("*")))
    return [p for p in files if p.is_file()]


def build_identity() -> str:
    """Deterministic source/build identity hash for provenance (task 4.8)."""
    digest = hashlib.sha256()
    digest.update(EXTENSION_SEMANTIC_VERSION.encode())
    digest.update(b"|" + TARGET_ARCH.encode())
    for path in _source_files():
        try:
            digest.update(b"|" + path.name.encode() + b"=" + hashlib.sha256(path.read_bytes()).hexdigest().encode())
        except OSError:
            digest.update(b"|" + path.name.encode() + b"=unreadable")
    return "sha256:" + digest.hexdigest()


def support_matrix() -> dict[str, Any]:
    """Qualified support matrix stub (task 4.8); real kernel may narrow it."""
    return {
        "target_architecture": TARGET_ARCH,
        "sm_70": SM_70,
        "semantic_version": EXTENSION_SEMANTIC_VERSION,
        "decode_backend_version": DECODE_BACKEND_VERSION,
        "chunk_policy": CHUNK_POLICY,
        "supported_head_dims": list(SUPPORTED_HEAD_DIMS),
        "supported_input_dtypes": list(SUPPORTED_INPUT_DTYPES),
        "supported_accumulator_modes": list(SUPPORTED_ACCUMULATOR_MODES),
        "supported_alpha_layouts": list(SUPPORTED_ALPHA_LAYOUTS),
        "state_layout": "[B,H,D,D]",
        "build_identity": build_identity(),
    }


def load_extension() -> tuple[bool, str | None]:
    """Lazily load the compiled fused extension (task 4.1).

    Returns ``(True, None)`` only when the compiled module imports. Otherwise
    returns ``(False, reason)`` with an actionable message. Never raises for a
    missing toolchain/GPU and never triggers a build on import.
    """
    if platform.system() != "Linux":
        return (
            False,
            "cuda-fused extension is unavailable: Linux-only build "
            f"(running on {platform.system()}); build on the V100S host with "
            "setup_gdn_cuda.py targeting sm_70.",
        )
    try:
        module = importlib.import_module(COMPILED_MODULE_NAME)
    except Exception as exc:  # noqa: BLE001 - reason must surface any failure
        return (
            False,
            "cuda-fused extension is unavailable: compiled module "
            f"{COMPILED_MODULE_NAME!r} could not be imported "
            f"({type(exc).__name__}: {exc}); build it on the Linux V100S host with "
            "'python setup_gdn_cuda.py build_ext --inplace' (sm_70, cu126 toolchain).",
        )
    try:
        arches = getattr(module, "COMPILED_ARCHES", None)
        if arches is not None and TARGET_ARCH not in arches:
            return (
                False,
                f"cuda-fused extension is unavailable: compiled arches {arches} "
                f"lack the required {TARGET_ARCH} path; rebuild with sm_70 enabled.",
            )
    except Exception as exc:  # noqa: BLE001
        return (False, f"cuda-fused extension arch check failed: {type(exc).__name__}: {exc}")
    return (True, None)


def _fail_unavailable(detail: str) -> Any:
    from recurrence import BackendUnavailableError  # lazy: avoids import cycles

    raise BackendUnavailableError(detail)


def validate_fused_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = 64,
) -> None:
    """Extension-side validation stub (task 4.7): actionable errors, no launch."""
    from recurrence import BackendUnavailableError  # lazy: avoids import cycles

    tensors = {"q": q, "k": k, "v": v, "alpha": alpha, "beta": beta}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise BackendUnavailableError(f"cuda-fused validation failed: {name} must be a torch.Tensor")
        if tensor.device.type != "cuda":
            raise BackendUnavailableError(
                f"cuda-fused validation failed: {name} must be on a CUDA device, got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise BackendUnavailableError(
                f"cuda-fused validation failed: {name} must be contiguous; call .contiguous() first"
            )
    if q.ndim != 4:
        raise BackendUnavailableError(f"cuda-fused validation failed: q must be [B,T,H,D], got {tuple(q.shape)}")
    batch, time, heads, width = q.shape
    if k.shape != q.shape or v.shape != q.shape:
        raise BackendUnavailableError("cuda-fused validation failed: q, k, v must share [B,T,H,D] shapes")
    if alpha.shape[:3] != (batch, time, heads) or alpha.shape[-1] not in (1, width):
        raise BackendUnavailableError(
            f"cuda-fused validation failed: alpha must be [B,T,H,1] or [B,T,H,{width}], got {tuple(alpha.shape)}"
        )
    if beta.shape != (batch, time, heads, 1):
        raise BackendUnavailableError(
            f"cuda-fused validation failed: beta must be [B,T,H,1], got {tuple(beta.shape)}"
        )
    if q.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise BackendUnavailableError(
            f"cuda-fused validation failed: input dtype {q.dtype} not in {SUPPORTED_INPUT_DTYPES}"
        )
    if width not in SUPPORTED_HEAD_DIMS:
        raise BackendUnavailableError(
            f"cuda-fused validation failed: head dim {width} not in supported {list(SUPPORTED_HEAD_DIMS)}"
        )
    if time < 1:
        raise BackendUnavailableError("cuda-fused validation failed: sequence length must be >= 1")
    if chunk_size < 1:
        raise BackendUnavailableError("cuda-fused validation failed: chunk_size must be positive")
    if initial_state is not None:
        if tuple(initial_state.shape) != (batch, heads, width, width):
            raise BackendUnavailableError(
                f"cuda-fused validation failed: initial state must be [B,H,D,D]={(batch, heads, width, width)}, "
                f"got {tuple(initial_state.shape)}"
            )
        if initial_state.device != q.device:
            raise BackendUnavailableError("cuda-fused validation failed: initial state must match Q device")
        if not initial_state.is_contiguous():
            raise BackendUnavailableError("cuda-fused validation failed: initial state must be contiguous")
    if torch.cuda.is_available():
        try:
            capability = torch.cuda.get_device_capability(q.device)
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailableError(f"cuda-fused validation failed: cannot query arch: {exc}") from exc
        if capability != (7, 0):
            raise BackendUnavailableError(
                "cuda-fused validation failed: qualified only for Volta sm_70, "
                f"found sm_{capability[0]}{capability[1]}"
            )


def fused_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused forward kernel stub (task 4.2): raises until the real kernel lands."""
    ok, reason = load_extension()
    if not ok:
        return _fail_unavailable(reason or "cuda-fused extension is unavailable")
    validate_fused_inputs(q, k, v, alpha, beta, initial_state, chunk_size=chunk_size)
    return _fail_unavailable(
        "cuda-fused forward kernel is not yet implemented (scaffold only): "
        "extension loaded but the sm_70 tiled forward path is deferred until "
        "the V100 pilot/toolchain is confirmed; refusing to return placeholder numerics."
    )


def fused_backward(*args: Any, **kwargs: Any) -> Any:
    """Custom backward kernel stub (task 4.4): raises, never fakes gradients."""
    ok, reason = load_extension()
    if not ok:
        return _fail_unavailable(reason or "cuda-fused extension is unavailable")
    return _fail_unavailable(
        "cuda-fused backward kernels are not yet implemented (scaffold only): "
        "reverse-time Q/K/V/alpha/beta/state kernels deferred until the V100 "
        "pilot/toolchain is confirmed; refusing to return placeholder gradients."
    )


class GdnFusedFunc(torch.autograd.Function):
    """Custom autograd operation stub (task 4.5).

    Correct signature ``(q, k, v, alpha, beta, initial_state, options)`` with
    FP32 validation and FP16-autocast input modes plus an explicit accumulator
    dtype carried by ``options``. Raises until the real kernel lands.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        options: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from recurrence import BackendUnavailableError  # lazy: avoids import cycles

        input_mode = getattr(options, "input_dtype", "?")
        accumulator_mode = getattr(options, "state_accumulator_dtype", "?")
        if input_mode not in ("float32", "autocast"):
            raise BackendUnavailableError(
                f"cuda-fused autograd rejected input_dtype mode {input_mode!r}: expected 'float32' or 'autocast'"
            )
        if accumulator_mode not in ("input", "float32"):
            raise BackendUnavailableError(
                f"cuda-fused autograd rejected accumulator mode {accumulator_mode!r}: expected 'input' or 'float32'"
            )
        chunk_size = int(getattr(options, "chunk_size", 64))
        validate_fused_inputs(q, k, v, alpha, beta, initial_state, chunk_size=chunk_size)
        raise BackendUnavailableError(
            "cuda-fused autograd forward is not yet implemented (scaffold only): "
            "sm_70 fused forward/backward deferred until the V100 pilot/toolchain "
            "is confirmed; refusing to return placeholder numerics."
        )

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[None, ...]:  # type: ignore[override]
        from recurrence import BackendUnavailableError  # lazy: avoids import cycles

        raise BackendUnavailableError(
            "cuda-fused autograd backward is not yet implemented (scaffold only)."
        )


__all__ = [
    "CHUNK_POLICY",
    "COMPILED_MODULE_NAME",
    "DECODE_BACKEND_VERSION",
    "EXTENSION_SEMANTIC_VERSION",
    "SM_70",
    "SUPPORTED_ACCUMULATOR_MODES",
    "SUPPORTED_ALPHA_LAYOUTS",
    "SUPPORTED_HEAD_DIMS",
    "SUPPORTED_INPUT_DTYPES",
    "TARGET_ARCH",
    "GdnFusedFunc",
    "build_identity",
    "fused_backward",
    "fused_forward",
    "load_extension",
    "support_matrix",
    "validate_fused_inputs",
]
