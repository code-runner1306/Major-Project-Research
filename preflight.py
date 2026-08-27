"""CUDA/NVML discovery and fail-closed production launch checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import torch


CUDA_12_MIN_DRIVER = (525, 60, 13)


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
        driver = pynvml.nvmlSystemGetDriverVersion().decode("ascii")
        if torch.cuda.is_available():
            handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            gpu_name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
            gpu_uuid = pynvml.nvmlDeviceGetUUID(handle).decode("ascii")
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
    if not torch.version.cuda or not torch.version.cuda.startswith("12"):
        reason = f"PyTorch CUDA runtime {torch.version.cuda!r} is not the configured CUDA 12.x channel"
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


def require_production_cuda(*, recurrence_backend: str = "reference") -> CudaPreflight:
    result = collect_cuda_preflight()
    if not result.compatible:
        raise RuntimeError(
            f"Production CUDA preflight failed: {result.reason}. Run nvidia-smi, record the driver version, "
            "and choose a compatible official PyTorch CUDA wheel before launching production training."
        )
    if recurrence_backend == "cuda-fused":
        if result.compute_capability != (7, 0):
            raise RuntimeError(
                "cuda-fused recurrence is targeted at the college V100S (sm_70); "
                f"found compute capability {result.compute_capability}."
            )
        if not result.extension_toolchain["ready_for_extension_build"]:
            raise RuntimeError(
                "cuda-fused recurrence requires a local CUDA toolkit, nvcc, and C++ compiler. "
                f"Discovered toolchain: {result.extension_toolchain}"
            )
    return result
