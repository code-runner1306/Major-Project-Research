"""Best-effort GPU/device, power, and activity observations for research runs."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import platform
import subprocess
import threading
import time
from typing import Any

import torch

from energy import EnergyProvider


def _text(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


class DeviceResourceCapture:
    """Capture whole-device telemetry; inability to query it never fails model work."""

    def __init__(self, device: torch.device, *, activity_interval_seconds: float = 1.0):
        self.device = device
        self.activity_interval_seconds = activity_interval_seconds
        self.energy_provider = EnergyProvider.create(device)
        self.energy_region = self.energy_provider.region("resource_capture")
        self.pynvml = None
        self.handle = None
        self.nvml_error: str | None = None
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.started_utc: str | None = None
        self.finished_utc: str | None = None
        self._start_time = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.result: dict[str, Any] | None = None
        self._torch_memory_before: dict[str, int] | None = None
        self._open_nvml()

    def _open_nvml(self) -> None:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            self.nvml_error = "CUDA/NVML unavailable"
            return
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            self.pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception as exc:
            self.nvml_error = f"NVML activity query unavailable: {type(exc).__name__}: {exc}"

    def runtime_metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": str(torch.__version__),
            "torch_cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_type": self.device.type,
            "device_index": self.device.index,
            "process_id": os.getpid(),
        }
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                result["torch_gpu_name"] = torch.cuda.get_device_name(self.device)
                result["compute_capability"] = list(torch.cuda.get_device_capability(self.device))
            except Exception as exc:
                result["torch_gpu_query_error"] = f"{type(exc).__name__}: {exc}"
            try:
                result["cudnn_version"] = torch.backends.cudnn.version()
            except Exception:
                result["cudnn_version"] = None
        if self.pynvml is not None and self.handle is not None:
            try:
                p = self.pynvml
                result.update({
                    "nvml_gpu_name": _text(p.nvmlDeviceGetName(self.handle)),
                    "gpu_uuid": _text(p.nvmlDeviceGetUUID(self.handle)),
                    "driver_version": _text(p.nvmlSystemGetDriverVersion()),
                    "nvml_available": True,
                })
            except Exception as exc:
                result["nvml_metadata_error"] = f"{type(exc).__name__}: {exc}"
        else:
            result["nvml_available"] = False
            result["nvml_unavailable_reason"] = self.nvml_error
        if self.device.type == "cuda" and ("driver_version" not in result or "nvml_gpu_name" not in result):
            try:
                probe = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True, timeout=3,
                )
                first = probe.stdout.strip().splitlines()[0].split(",")
                if len(first) == 3:
                    result.setdefault("nvidia_smi_gpu_name", first[0].strip())
                    result.setdefault("gpu_uuid", first[1].strip())
                    result.setdefault("driver_version", first[2].strip())
            except Exception as exc:
                result["nvidia_smi_fallback_error"] = f"{type(exc).__name__}: {exc}"
        return result

    def _sample(self) -> dict[str, Any]:
        sample: dict[str, Any] = {"timestamp_utc": datetime.now(timezone.utc).isoformat()}
        p, handle = self.pynvml, self.handle
        if p is None or handle is None:
            sample.update({"available": False, "reason": self.nvml_error})
            return sample
        sample["available"] = True
        for field, getter in (
            ("gpu_utilization_percent", lambda: p.nvmlDeviceGetUtilizationRates(handle).gpu),
            ("memory_utilization_percent", lambda: p.nvmlDeviceGetUtilizationRates(handle).memory),
            ("memory_used_bytes", lambda: p.nvmlDeviceGetMemoryInfo(handle).used),
            ("memory_total_bytes", lambda: p.nvmlDeviceGetMemoryInfo(handle).total),
            ("power_watts", lambda: p.nvmlDeviceGetPowerUsage(handle) / 1000.0),
            ("temperature_celsius", lambda: p.nvmlDeviceGetTemperature(handle, p.NVML_TEMPERATURE_GPU)),
            ("sm_clock_mhz", lambda: p.nvmlDeviceGetClockInfo(handle, p.NVML_CLOCK_SM)),
            ("memory_clock_mhz", lambda: p.nvmlDeviceGetClockInfo(handle, p.NVML_CLOCK_MEM)),
        ):
            try:
                sample[field] = getter()
            except Exception as exc:
                sample[field] = None
                sample.setdefault("field_errors", {})[field] = f"{type(exc).__name__}: {exc}"
        try:
            processes = p.nvmlDeviceGetComputeRunningProcesses(handle)
            pids = {int(process.pid) for process in processes}
            sample["compute_process_count"] = len(pids)
            sample["current_process_visible"] = os.getpid() in pids
            sample["other_compute_process_count"] = len(pids - {os.getpid()})
            sample["process_visibility"] = "nvml_compute_process_list"
        except Exception as exc:
            sample["compute_process_count"] = None
            sample["current_process_visible"] = None
            sample["other_compute_process_count"] = None
            sample["process_visibility"] = "unavailable"
            sample.setdefault("field_errors", {})["compute_processes"] = f"{type(exc).__name__}: {exc}"
        return sample

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.activity_interval_seconds):
            try:
                self.samples.append(self._sample())
            except Exception as exc:  # Defensive: monitoring must never interrupt training/evaluation.
                self.errors.append(f"Activity sample failed: {type(exc).__name__}: {exc}")

    def __enter__(self) -> "DeviceResourceCapture":
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self._start_time = time.perf_counter()
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                self._torch_memory_before = {
                    "allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
                }
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception as exc:
                self.errors.append(f"PyTorch CUDA memory telemetry unavailable: {type(exc).__name__}: {exc}")
        self.energy_region.__enter__()
        try:
            self.samples.append(self._sample())
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        except Exception as exc:
            self.errors.append(f"GPU activity sampling could not start: {type(exc).__name__}: {exc}")
            self._thread = None
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.activity_interval_seconds * 2 + 1)
        try:
            self.samples.append(self._sample())
        except Exception as sample_exc:
            self.errors.append(f"Final activity sample failed: {type(sample_exc).__name__}: {sample_exc}")
        self.energy_region.__exit__(exc_type, exc, traceback)
        self.finished_utc = datetime.now(timezone.utc).isoformat()
        torch_memory: dict[str, Any] = {"available": False, "scope": "current process PyTorch CUDA allocator"}
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                torch_memory = {
                    "available": True,
                    "scope": "current process PyTorch CUDA allocator",
                    "allocated_before_bytes": (self._torch_memory_before or {}).get("allocated_bytes"),
                    "reserved_before_bytes": (self._torch_memory_before or {}).get("reserved_bytes"),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
                    "allocated_after_bytes": int(torch.cuda.memory_allocated(self.device)),
                    "reserved_after_bytes": int(torch.cuda.memory_reserved(self.device)),
                }
            except Exception as exc:
                torch_memory["reason"] = f"{type(exc).__name__}: {exc}"
        numeric_fields = (
            "gpu_utilization_percent", "memory_utilization_percent", "memory_used_bytes",
            "power_watts", "temperature_celsius", "sm_clock_mhz", "memory_clock_mhz",
            "other_compute_process_count",
        )
        aggregates: dict[str, Any] = {}
        for field in numeric_fields:
            values = [float(sample[field]) for sample in self.samples
                      if isinstance(sample.get(field), (int, float))]
            aggregates[field] = {
                "mean": sum(values) / len(values) if values else None,
                "max": max(values) if values else None,
                "min": min(values) if values else None,
                "samples": len(values),
            }
        process_counts = [sample.get("other_compute_process_count") for sample in self.samples
                          if isinstance(sample.get("other_compute_process_count"), int)]
        if process_counts:
            other_activity = {
                "status": "visible",
                "other_compute_process_seen": any(count > 0 for count in process_counts),
                "other_compute_process_count_max": max(process_counts),
                "interpretation": "NVML-visible compute processes only; zero does not prove the GPU was otherwise idle",
            }
        else:
            other_activity = {
                "status": "unknown",
                "other_compute_process_seen": None,
                "reason": "NVML did not expose the device compute-process list",
            }
        self.result = {
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "duration_wall_seconds": time.perf_counter() - self._start_time,
            "runtime": self.runtime_metadata(),
            "energy": self.energy_region.result.to_dict() if self.energy_region.result else None,
            "torch_memory": torch_memory,
            "activity_summary": aggregates,
            "other_gpu_activity": other_activity,
            "activity_sample_interval_seconds": self.activity_interval_seconds,
            "activity_sample_count": len(self.samples),
            "activity_samples": self.samples,
            "monitor_errors": self.errors,
        }
        return False
