"""CUDA/NVML discovery and fail-closed production launch checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import torch

from nvml_utils import nvml_text


CUDA_12_MIN_DRIVER = (525, 60, 13)

# Fail-closed cuda-fused support matrix ( Volta V100S target + production scales ).
SUPPORTED_HEAD_DIMS = (40, 48, 56)
SUPPORTED_INPUT_DTYPES = ("autocast", "float32")
SUPPORTED_STATE_ACCUMULATOR_DTYPES = ("input", "float32")


def _version_tuple(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    match = re.search(r"(\d+(?:\.\d+)+)", value)
    return tuple(int(part) for part in match.group(1).split(".")) if match else None


def _at_least(actual: tuple[int, ...] | None, minimum: tuple[int, ...]) -> bool:
    if actual is None:
        return False
    padded = actual + (0,) * (len(minimum) - len(actual))
    return padded >= minimum


@dataclass(frozen=True)
class CudaPreflight:
    python: str
    torch: str
    torch_cuda: str | None
    cuda_available: bool
    gpu_name: str | None
    gpu_uuid: str | None
    driver: str | None
    nvml_available: bool
    energy_counter_supported: bool
    power_sampling_supported: bool
    power_limit_watts: float | None
    compute_capability: tuple[int, int] | None
    extension_toolchain: dict[str, Any]
    compatible: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extension_toolchain() -> dict[str, Any]:
    """Discover, but never invoke, the optional local CUDA build toolchain."""

    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:
        CUDA_HOME = None
    cuda_home = Path(CUDA_HOME) if CUDA_HOME else None
    nvcc = shutil.which("nvcc")
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    return {
        "cuda_home": str(cuda_home) if cuda_home else None,
        "nvcc": nvcc,
        "cxx": compiler,
        "cuda_headers_present": bool(cuda_home and (cuda_home / "include" / "cuda.h").is_file()),
        "ready_for_extension_build": bool(nvcc and compiler and cuda_home),
    }


def collect_cuda_preflight() -> CudaPreflight:
    driver = gpu_name = gpu_uuid = None
    nvml_available = energy_counter_supported = power_sampling_supported = False
    power_limit_watts = None
    handle = None
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        nvml_available = True
        driver = nvml_text(pynvml.nvmlSystemGetDriverVersion(), "ascii")
        if torch.cuda.is_available():
            handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            gpu_name = nvml_text(pynvml.nvmlDeviceGetName(handle))
            gpu_uuid = nvml_text(pynvml.nvmlDeviceGetUUID(handle), "ascii")
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                energy_counter_supported = True
            except pynvml.NVMLError:
                pass
            try:
                pynvml.nvmlDeviceGetPowerUsage(handle)
                power_sampling_supported = True
            except pynvml.NVMLError:
                pass
            try:
                power_limit_watts = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
            except pynvml.NVMLError:
                pass
    except Exception:
        pass
    finally:
        if nvml_available:
            try:
                import pynvml  # type: ignore[import-not-found]
                pynvml.nvmlShutdown()
            except Exception:
                pass

    toolchain = _extension_toolchain()
    if not torch.cuda.is_available():
        return CudaPreflight(
            python=sys.version.split()[0], torch=torch.__version__, torch_cuda=torch.version.cuda,
            cuda_available=False, gpu_name=None, gpu_uuid=None, driver=driver,
            nvml_available=nvml_available, energy_counter_supported=False, power_sampling_supported=False,
            power_limit_watts=None, compute_capability=None, extension_toolchain=toolchain,
            compatible=False, reason="CUDA is not available to PyTorch",
        )
    cuda_ver = torch.version.cuda or ""
    if not cuda_ver or not (cuda_ver.startswith("12") or cuda_ver.startswith("13")):
        reason = f"PyTorch CUDA runtime {torch.version.cuda!r} is not a supported CUDA channel (expected 12.x or 13.x)"
        compatible = False
    elif not _at_least(_version_tuple(driver), CUDA_12_MIN_DRIVER):
        reason = "NVIDIA driver is unknown or below CUDA 12.x minimum 525.60.13; select a compatible official PyTorch wheel channel"
        compatible = False
    else:
        reason, compatible = "compatible", True
    return CudaPreflight(
        python=sys.version.split()[0], torch=torch.__version__, torch_cuda=torch.version.cuda,
        cuda_available=True, gpu_name=gpu_name or torch.cuda.get_device_name(), gpu_uuid=gpu_uuid,
        driver=driver, nvml_available=nvml_available, energy_counter_supported=energy_counter_supported,
        power_sampling_supported=power_sampling_supported, power_limit_watts=power_limit_watts,
        compute_capability=torch.cuda.get_device_capability(), extension_toolchain=toolchain,
        compatible=compatible, reason=reason,
    )


def _qualification_file_hash(path: str | None) -> str | None:
    """Hash a qualification artifact without importing recurrence (CPU-safe)."""
    if not path:
        return None
    try:
        target = Path(path)
        if not target.is_file():
            return None
        from hashlib import sha256
        return sha256(target.read_bytes()).hexdigest()
    except OSError:
        return None


def _check_qualification_coverage(
    *, qualification_artifact: str | None, input_dtype: str,
    state_accumulator_dtype: str, chunk_size: int, head_dims: tuple[int, ...] | None,
) -> None:
    """Fail closed when the qualification artifact is missing, unreadable, or
    explicitly does not cover the requested configuration.

    Artifacts without explicit coverage fields are accepted here on presence +
    hash alone; exact backend/build/dtype/chunk/qualification drift is rejected
    by manifest fingerprint compatibility at resumption time.
    """
    import json

    if not qualification_artifact:
        raise RuntimeError(
            "cuda-fused production requires --qualification-artifact pointing at the "
            "accepted V100 qualification JSON; re-run qualification or select "
            "--recurrence-backend reference explicitly for diagnostics."
        )
    digest = _qualification_file_hash(qualification_artifact)
    if digest is None:
        raise RuntimeError(
            f"Qualification artifact {qualification_artifact!r} is missing or unreadable "
            f"(sha256 unavailable). Provide the immutable qualification JSON from the "
            "successful V100 qualification run; refusing cuda-fused startup."
        )
    try:
        payload = json.loads(Path(qualification_artifact).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Non-JSON artifacts: presence + hash is the compatibility identity.
        return
    if not isinstance(payload, dict):
        return
    problems: list[str] = []
    coverage_dtypes = payload.get("supported_input_dtypes", payload.get("input_dtypes"))
    if isinstance(coverage_dtypes, list) and input_dtype not in coverage_dtypes:
        problems.append(f"input dtype {input_dtype!r} not in qualified {coverage_dtypes}")
    coverage_accum = payload.get("supported_state_accumulator_dtypes", payload.get("state_accumulator_dtypes"))
    if isinstance(coverage_accum, list) and state_accumulator_dtype not in coverage_accum:
        problems.append(f"state accumulator {state_accumulator_dtype!r} not in qualified {coverage_accum}")
    coverage_chunk = payload.get("supported_chunk_sizes", payload.get("chunk_sizes", payload.get("chunk_size")))
    if isinstance(coverage_chunk, list) and chunk_size not in coverage_chunk:
        problems.append(f"chunk_size {chunk_size} not in qualified {coverage_chunk}")
    elif coverage_chunk is not None and not isinstance(coverage_chunk, list) and chunk_size != coverage_chunk:
        problems.append(f"chunk_size {chunk_size} != qualified {coverage_chunk}")
    coverage_heads = payload.get("supported_head_dims", payload.get("head_dims"))
    if isinstance(coverage_heads, list) and head_dims is not None:
        uncovered = [h for h in head_dims if h not in coverage_heads]
        if uncovered:
            problems.append(f"head dims {uncovered} not in qualified {coverage_heads}")
    if problems:
        raise RuntimeError(
            "Qualification artifact "
            f"{qualification_artifact!r} (sha256 {digest[:12]}) does not cover the requested "
            "cuda-fused configuration: " + "; ".join(problems) +
            ". Qualify this exact backend/build/dtype/chunk/shape combination first."
        )


def require_production_cuda(
    *,
    recurrence_backend: str = "reference",
    input_dtype: str = "autocast",
    state_accumulator_dtype: str = "input",
    chunk_size: int = 64,
    qualification_artifact: str | None = None,
    head_dims: tuple[int, ...] | list[int] | None = None,
    recurrence_options: Any | None = None,
) -> CudaPreflight:
    if recurrence_options is not None:
        recurrence_backend = recurrence_options.backend
        input_dtype = recurrence_options.input_dtype
        state_accumulator_dtype = recurrence_options.state_accumulator_dtype
        chunk_size = recurrence_options.chunk_size
        qualification_artifact = recurrence_options.qualification_artifact
    result = collect_cuda_preflight()
    if not result.compatible:
        raise RuntimeError(
            f"Production CUDA preflight failed: {result.reason}. Run nvidia-smi, record the driver version, "
            "and choose a compatible official PyTorch CUDA wheel before launching production training."
        )
    if recurrence_backend == "reference":
        # Explicitly recorded reference production run for diagnostics; no
        # optimized qualification is claimed. CUDA compatibility above still applies.
        return result
    if recurrence_backend != "cuda-fused":
        raise RuntimeError(
            f"Unknown recurrence backend {recurrence_backend!r}; production requires "
            "'reference' (explicit diagnostics) or 'cuda-fused' (qualified V100 path)."
        )
    if result.compute_capability != (7, 0):
        raise RuntimeError(
            "cuda-fused recurrence is targeted at the college V100S (sm_70); "
            f"found compute capability {result.compute_capability}. "
            "Run on the qualified V100S or select --recurrence-backend reference explicitly."
        )
    if input_dtype not in SUPPORTED_INPUT_DTYPES:
        raise RuntimeError(
            f"cuda-fused input dtype {input_dtype!r} is outside qualified {list(SUPPORTED_INPUT_DTYPES)}; "
            "qualify this dtype combination first instead of approximating."
        )
    if state_accumulator_dtype not in SUPPORTED_STATE_ACCUMULATOR_DTYPES:
        raise RuntimeError(
            f"cuda-fused state accumulator {state_accumulator_dtype!r} is outside qualified "
            f"{list(SUPPORTED_STATE_ACCUMULATOR_DTYPES)}; qualify this accumulator mode first."
        )
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise RuntimeError(
            f"cuda-fused chunk_size {chunk_size!r} is invalid; provide the qualified positive "
            "checkpoint chunk size (production default 64)."
        )
    if head_dims is not None:
        unsupported = [h for h in head_dims if h not in SUPPORTED_HEAD_DIMS]
        if unsupported:
            raise RuntimeError(
                f"cuda-fused head dims {unsupported} are outside qualified {list(SUPPORTED_HEAD_DIMS)}; "
                "qualify these production shapes before training instead of approximating."
            )
    if not result.extension_toolchain["ready_for_extension_build"]:
        raise RuntimeError(
            "cuda-fused recurrence requires a local CUDA toolkit, nvcc, and C++ compiler. "
            f"Discovered toolchain: {result.extension_toolchain}. "
            "Install CUDA_HOME/nvcc plus a C++ compiler with headers on the V100S host."
        )
    try:
        import gdn_cuda  # type: ignore[import-not-found]  # noqa: F401

        _ = gdn_cuda
    except Exception as exc:
        raise RuntimeError(
            "cuda-fused extension is unloadable; build the repository-owned sm_70 extension "
            f"on the V100 host before production startup ({type(exc).__name__}: {exc}). "
            f"Toolchain seen: {result.extension_toolchain}"
        ) from exc
    _check_qualification_coverage(
        qualification_artifact=qualification_artifact, input_dtype=input_dtype,
        state_accumulator_dtype=state_accumulator_dtype, chunk_size=chunk_size,
        head_dims=tuple(head_dims) if head_dims is not None else None,
    )
    return result
