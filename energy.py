"""NVML energy measurement with a CPU-safe unsupported provider."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
import threading
import time
from typing import Any

import torch


@dataclass
class EnergyResult:
    supported: bool
    method: str | None
    joules: float | None
    duration_seconds: float | None
    average_watts: float | None
    peak_watts: float | None
    sampling_interval_seconds: float | None
    reason: str | None
    device: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EnergyRegion(AbstractContextManager["EnergyRegion"]):
    def __init__(self, provider: "EnergyProvider", name: str):
        self.provider = provider
        self.name = name
        self.result: EnergyResult | None = None
        self._start_counter: int | None = None
        self._start_time: float | None = None
        self._samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "EnergyRegion":
        self._start_time = time.perf_counter()
        if not self.provider.supported:
            return self
        if self.provider.energy_counter_supported:
            try:
                self._start_counter = self.provider.read_energy_millijoules()
            except Exception as exc:
                if not self.provider.power_sampling_supported:
                    self.provider.supported = False
                    self.provider.reason = f"NVML energy counter became unavailable: {exc}"
                    return self
                self.provider.energy_counter_supported = False
                self.provider.reason = f"NVML energy counter failed; falling back to sampled power: {exc}"
        if not self.provider.energy_counter_supported:
            try:
                self._sample_once()
                self._thread = threading.Thread(target=self._sample_loop, daemon=True)
                self._thread.start()
            except Exception as exc:
                self.provider.supported = False
                self.provider.reason = f"NVML power sampling became unavailable: {exc}"
        return self

    def _sample_once(self) -> None:
        self._samples.append((time.perf_counter(), self.provider.read_power_watts()))

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.provider.sampling_interval_seconds):
            try:
                self._sample_once()
            except Exception as exc:
                self.provider.supported = False
                self.provider.reason = f"NVML power sampling failed during measurement: {exc}"
                self._stop.set()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        end_time = time.perf_counter()
        duration = end_time - (self._start_time or end_time)
        if not self.provider.supported:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=self.provider.sampling_interval_seconds * 2 + 1)
            self.result = EnergyResult(
                supported=False, method=None, joules=None, duration_seconds=duration,
                average_watts=None, peak_watts=None, sampling_interval_seconds=None,
                reason=self.provider.reason, device=self.provider.device_metadata,
            )
            return False
        if self.provider.energy_counter_supported:
            assert self._start_counter is not None
            try:
                joules = max(0.0, (self.provider.read_energy_millijoules() - self._start_counter) / 1000.0)
            except Exception as exc:
                self.result = EnergyResult(
                    supported=False, method=None, joules=None, duration_seconds=duration,
                    average_watts=None, peak_watts=None, sampling_interval_seconds=None,
                    reason=f"NVML energy counter failed during measurement: {exc}",
                    device=self.provider.device_metadata,
                )
                return False
            try:
                self._sample_once()
            except Exception:
                pass
            sampled_watts = [sample[1] for sample in self._samples]
            self.result = EnergyResult(
                supported=True, method="nvml_total_energy_counter", joules=joules,
                duration_seconds=duration, average_watts=joules / duration if duration else None,
                peak_watts=max(sampled_watts) if sampled_watts else None, sampling_interval_seconds=None, reason=None,
                device=self.provider.device_metadata,
            )
        else:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=self.provider.sampling_interval_seconds * 2 + 1)
            try:
                self._sample_once()
            except Exception as exc:
                self.result = EnergyResult(
                    supported=False, method=None, joules=None, duration_seconds=duration,
                    average_watts=None, peak_watts=None, sampling_interval_seconds=None,
                    reason=f"NVML power sampling failed at measurement end: {exc}",
                    device=self.provider.device_metadata,
                )
                return False
            joules = 0.0
            for (previous_t, previous_w), (current_t, current_w) in zip(self._samples, self._samples[1:]):
                joules += (current_t - previous_t) * (previous_w + current_w) / 2
            watts = [sample[1] for sample in self._samples]
            self.result = EnergyResult(
                supported=True, method="nvml_sampled_power", joules=joules,
                duration_seconds=duration, average_watts=sum(watts) / len(watts) if watts else None,
                peak_watts=max(watts) if watts else None,
                sampling_interval_seconds=self.provider.sampling_interval_seconds, reason=None,
                device=self.provider.device_metadata,
            )
        return False


class EnergyProvider:
    """A small abstraction so metrics never fabricate energy on CPU hardware."""

    def __init__(
        self, *, supported: bool, reason: str | None, device_metadata: dict[str, Any],
        energy_counter_supported: bool = False, power_sampling_supported: bool = False,
        sampling_interval_seconds: float = 0.05,
    ):
        self.supported = supported
        self.reason = reason
        self.device_metadata = device_metadata
        self.energy_counter_supported = energy_counter_supported
        self.power_sampling_supported = power_sampling_supported
        self.sampling_interval_seconds = sampling_interval_seconds

    @classmethod
    def create(cls, device: torch.device, sampling_interval_seconds: float = 0.05) -> "EnergyProvider":
        if device.type != "cuda" or not torch.cuda.is_available():
            return cls(
                supported=False, reason="CUDA/NVML is unavailable on this device",
                device_metadata={"device_type": device.type, "cuda_runtime": torch.version.cuda},
            )
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            index = device.index if device.index is not None else torch.cuda.current_device()
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            metadata = {
                "device_type": "cuda",
                "name": pynvml.nvmlDeviceGetName(handle).decode("utf-8"),
                "uuid": pynvml.nvmlDeviceGetUUID(handle).decode("ascii"),
                "driver": pynvml.nvmlSystemGetDriverVersion().decode("ascii"),
                "cuda_runtime": torch.version.cuda,
            }
            try:
                metadata["power_limit_watts"] = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
            except pynvml.NVMLError:
                metadata["power_limit_watts"] = None
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                counter = True
            except pynvml.NVMLError:
                counter = False
            try:
                pynvml.nvmlDeviceGetPowerUsage(handle)
                power_sampling = True
            except pynvml.NVMLError:
                power_sampling = False
            supported = counter or power_sampling
            provider = cls(
                supported=supported,
                reason=None if supported else "NVML exposes neither a total-energy counter nor GPU power samples",
                device_metadata=metadata, energy_counter_supported=counter,
                power_sampling_supported=power_sampling,
                sampling_interval_seconds=sampling_interval_seconds,
            )
            provider._pynvml, provider._handle = pynvml, handle  # type: ignore[attr-defined]
            return provider
        except Exception as exc:
            return cls(
                supported=False, reason=f"NVML initialization failed: {exc}",
                device_metadata={"device_type": "cuda", "cuda_runtime": torch.version.cuda},
            )

    def read_energy_millijoules(self) -> int:
        return int(self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle))  # type: ignore[attr-defined]

    def read_power_watts(self) -> float:
        return float(self._pynvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0  # type: ignore[attr-defined]

    def region(self, name: str) -> EnergyRegion:
        return EnergyRegion(self, name)


def normalized_energy(result: EnergyResult, *, tokens: int, unit: str) -> dict[str, Any]:
    data = result.to_dict()
    data["token_count"] = tokens
    if not result.supported or result.joules is None or tokens <= 0:
        data[f"joules_per_{unit}"] = None
    else:
        data[f"joules_per_{unit}"] = result.joules / tokens
    return data
