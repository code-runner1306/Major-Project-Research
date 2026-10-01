from __future__ import annotations

import time

from energy import EnergyProvider
from resource_telemetry import DeviceResourceCapture
from retrieval import RetrievalDataset


def test_retrieval_is_deterministic_masked_and_split_distinct(smoke_config):
    train = RetrievalDataset(task="mqar", examples=4, config=smoke_config.retrieval, seed=17, split="train")
    held_out = RetrievalDataset(task="mqar", examples=4, config=smoke_config.retrieval, seed=17, split="held_out")
    first, repeat = train.example(0), train.example(0)
    assert first.tokens.equal(repeat.tokens)
    assert (first.labels != -100).any()
    assert train.manifest()["hash"] != held_out.manifest()["hash"]
    train_cases = {tuple(train.example(index).tokens.tolist()) for index in range(len(train))}
    held_cases = {tuple(held_out.example(index).tokens.tolist()) for index in range(len(held_out))}
    assert train_cases.isdisjoint(held_cases)


def test_overwrite_targets_latest_value_and_marks_stale_values(smoke_config):
    dataset = RetrievalDataset(task="overwrite", examples=4, config=smoke_config.retrieval, seed=17, split="held_out")
    example = dataset.example(1)
    latest = [answer for answer in example.answers if answer.kind == "latest"]
    assert len(latest) == 1
    assert latest[0].stale_values
    assert int(example.labels[latest[0].label_position]) == latest[0].target


def test_cpu_energy_is_structured_unsupported():
    provider = EnergyProvider.create(__import__("torch").device("cpu"))
    with provider.region("test") as region:
        pass
    assert region.result is not None
    assert region.result.supported is False
    assert region.result.joules is None


class _CounterEnergy(EnergyProvider):
    def __init__(self):
        super().__init__(supported=True, reason=None, device_metadata={"name": "fake"}, energy_counter_supported=True)
        self.calls = 0

    def read_energy_millijoules(self):
        self.calls += 1
        return 1_000 if self.calls == 1 else 1_750


class _SampledEnergy(EnergyProvider):
    def __init__(self):
        super().__init__(supported=True, reason=None, device_metadata={"name": "fake"}, sampling_interval_seconds=0.001)

    def read_power_watts(self):
        return 42.0


class _UnavailablePower(EnergyProvider):
    def __init__(self):
        super().__init__(supported=True, reason=None, device_metadata={"name": "mock"})

    def read_power_watts(self):
        raise RuntimeError("power telemetry denied")


def test_nvml_counter_and_sampling_paths():
    counter = _CounterEnergy()
    with counter.region("counter") as region:
        pass
    assert region.result is not None
    assert region.result.method == "nvml_total_energy_counter"
    assert region.result.joules == 0.75
    sampled = _SampledEnergy()
    with sampled.region("sampling") as region:
        time.sleep(0.003)
    assert region.result is not None
    assert region.result.method == "nvml_sampled_power"
    assert region.result.joules is not None and region.result.joules >= 0


def test_unavailable_power_telemetry_is_recordable_without_failing_work():
    provider = _UnavailablePower()
    with provider.region("unavailable") as region:
        pass
    assert region.result is not None
    assert region.result.supported is False
    assert region.result.joules is None
    assert "power sampling" in (region.result.reason or "")


def test_cpu_resource_capture_records_unknown_gpu_telemetry_without_failure():
    import torch

    capture = DeviceResourceCapture(torch.device("cpu"), activity_interval_seconds=0.001)
    with capture:
        time.sleep(0.002)
    assert capture.result is not None
    assert capture.result["energy"]["supported"] is False
    assert capture.result["activity_sample_count"] >= 2
    assert capture.result["other_gpu_activity"]["status"] == "unknown"
