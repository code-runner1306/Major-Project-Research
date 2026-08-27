from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from config import production_profile
from pilot import PilotSettings, _pilot_output_dir, _summary, run_v100_pilot
from recurrence import (
    BackendUnavailableError,
    RecurrenceOptions,
    backend_metadata,
    ensure_backend_available,
    reference_recurrence,
)


def _inputs(*, channel_wise: bool, requires_grad: bool = False):
    torch.manual_seed(4)
    q = torch.randn(2, 5, 3, 4, requires_grad=requires_grad)
    k = torch.randn(2, 5, 3, 4, requires_grad=requires_grad)
    v = torch.randn(2, 5, 3, 4, requires_grad=requires_grad)
    alpha_width = 4 if channel_wise else 1
    alpha = torch.sigmoid(torch.randn(2, 5, 3, alpha_width)).detach().requires_grad_(requires_grad)
    beta = torch.sigmoid(torch.randn(2, 5, 3, 1)).detach().requires_grad_(requires_grad)
    return q, k, v, alpha, beta


@pytest.mark.parametrize("channel_wise", [False, True])
def test_reference_recurrence_shapes_initial_state_and_continuation(channel_wise):
    q, k, v, alpha, beta = _inputs(channel_wise=channel_wise)
    output, state = reference_recurrence(q, k, v, alpha, beta)
    assert output.shape == q.shape
    assert state.shape == (2, 3, 4, 4)

    initial = torch.randn_like(state)
    with_initial, final = reference_recurrence(q, k, v, alpha, beta, initial)
    assert not torch.allclose(output, with_initial)
    first, middle = reference_recurrence(q[:, :2], k[:, :2], v[:, :2], alpha[:, :2], beta[:, :2], initial)
    second, continued = reference_recurrence(q[:, 2:], k[:, 2:], v[:, 2:], alpha[:, 2:], beta[:, 2:], middle)
    assert torch.allclose(torch.cat((first, second), dim=1), with_initial)
    assert torch.allclose(continued, final)


@pytest.mark.parametrize("channel_wise", [False, True])
def test_reference_recurrence_has_gradients_for_every_input(channel_wise):
    q, k, v, alpha, beta = _inputs(channel_wise=channel_wise, requires_grad=True)
    initial = torch.randn(2, 3, 4, 4, requires_grad=True)
    output, state = reference_recurrence(q, k, v, alpha, beta, initial)
    (output.square().mean() + state.square().mean()).backward()
    for tensor in (q, k, v, alpha, beta, initial):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_reference_recurrence_validates_inputs_and_explicit_float32_accumulator():
    q, k, v, alpha, beta = _inputs(channel_wise=False)
    with pytest.raises(ValueError, match="alpha"):
        reference_recurrence(q, k, v, alpha[..., :0], beta)
    with pytest.raises(ValueError, match="state shape"):
        reference_recurrence(q, k, v, alpha, beta, torch.zeros(1, 1, 1, 1))
    output, state = reference_recurrence(
        q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
        alpha.to(torch.bfloat16), beta.to(torch.bfloat16),
        options=RecurrenceOptions(input_dtype="float32", state_accumulator_dtype="input"),
    )
    assert output.dtype == torch.bfloat16
    assert state.dtype == torch.float32


def test_backend_metadata_and_cpu_fused_rejection_are_explicit():
    metadata = backend_metadata(RecurrenceOptions())
    assert metadata["resolved_backend"] == "reference"
    assert metadata["state_layout"] == "[B,H,D,D]"
    with pytest.raises(BackendUnavailableError, match="CUDA device"):
        ensure_backend_available(RecurrenceOptions(backend="cuda-fused"), device=torch.device("cpu"))


def test_pilot_statistics_and_cpu_skip_reason(tmp_path):
    assert _summary([1.0, 2.0, 5.0]) == {"count": 3, "mean": pytest.approx(8 / 3), "std": pytest.approx(1.699673171197595), "p50": 2.0, "p95": 5.0}
    profile = production_profile(tmp_path / "pilot")
    settings = PilotSettings(scales=("55M",), warmup_steps=1, measured_steps=1)
    first = _pilot_output_dir(profile, settings, "20260827T000000Z")
    second = _pilot_output_dir(profile, settings, "20260827T000000Z")
    assert first.parent == second.parent
    assert first != second
    with pytest.raises(RuntimeError, match="Production CUDA preflight failed"):
        run_v100_pilot(replace(profile, recurrence_backend="reference"), settings)
    assert not profile.output_dir.exists()
