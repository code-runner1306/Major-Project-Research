"""CUDA qualification tests for OpenSpec change ``optimize-gdn-recurrence`` (tasks 5.1-5.6).

What this module is
-------------------
* CUDA-only qualification cases (``test_cuda_*``) encode the spec gates from
  ``specs/recurrence-optimization-qualification/spec.md`` so the V100S host can
  execute them verbatim:
    - 5.1 FP32 forward/final-state equivalence, max-abs ``1e-4``, finite.
    - 5.2 backward-gradient equivalence, normalized-L2 ``1e-3`` + max-abs
      ``1e-3``, including unreduced channel-wise alpha gradients.
    - 5.3 split-sequence continuation within the dtype tolerance.
    - 5.4 prefill/cached-decode vs full-prefix per variant.
    - 5.5 paired 5-step FP32 trajectory (per-step loss diff ``1e-3``) plus a
      mixed-precision stability guard (``2e-2``) that fails on non-finite
      values, missing gradients, or silent fallback.
* CPU-runnable harness checks (``test_*_on_cpu``) verify the same metric and
  trajectory helpers against the ``reference`` backend so ``tasks 5.x`` logic
  is proven on the CPU laptop without a GPU.
* Task 5.6: every ``test_cuda_*`` carries an explicit
  ``pytest.mark.skipif(torch.cuda.is_available() is False, ...)`` marker AND a
  runtime ``_require_cuda_device()`` / ``_require_fused_backend()`` gate, so a
  CPU run SKIPS (never fake-passes) with an actionable reason while reference
  smoke stays mandatory.

Task 7.3 note (pinned deps / build helpers): no dependency change is made by
this module. ``requirements.txt`` keeps the ``cu126`` PyTorch channel and the
CUDA extension stays an *optional* lazy import (``gdn_cuda`` via
``recurrence._fused_load_reason``); nothing here makes CUDA build tools
mandatory for CPU smoke. If a future extension build needs helpers (ninja /
setuptools / CUDA toolkit), they must be added as optional extras only.

Until change tasks 4.x land, ``ensure_backend_available`` always raises
``BackendUnavailableError`` for ``cuda-fused`` (extension not yet built), so
the ``test_cuda_*`` cases skip even on a CUDA host with an actionable
"extension unavailable" reason instead of passing fictitiously.
"""

from __future__ import annotations

import pytest
import torch

from config import PRODUCTION_MODELS, ModelConfig
from controls import construct_model
from eval import cache_equivalence_error
from recurrence import (
    BackendUnavailableError,
    RecurrenceOptions,
    ensure_backend_available,
    reference_recurrence,
    run_recurrence,
)
from train import seed_everything

# ---------------------------------------------------------------------------
# Tolerances (mirror the qualification spec)
# ---------------------------------------------------------------------------

FP32_FWD_MAX_ABS = 1e-4
MIXED_MAX_ABS = 2e-2
GRAD_NORM_L2_TOL = 1e-3
GRAD_MAX_ABS_TOL = 1e-3
TRAJECTORY_LOSS_TOL = 1e-3

PRODUCTION_HEAD_DIMS = sorted({m.d_head for m in PRODUCTION_MODELS.values()})
assert PRODUCTION_HEAD_DIMS, "production models must define head dims"
# Expected today: 10M/30M -> 40, 100M -> 48, 55M -> 56.
assert set(PRODUCTION_HEAD_DIMS) == {40, 48, 56}, PRODUCTION_HEAD_DIMS

SEQ_LENS_FWD = (16, 128, 512)
SEQ_LEN_GRAD = 32
SPLIT_LEN = 64

CUDA_SKIP_REASON = (
    "Requires a CUDA Volta sm_70 (V100S) device with the built gdn_cuda "
    "extension for cuda-fused qualification; CPU laptop path uses the "
    "reference backend (run the CPU smoke suite instead)."
)

needs_cuda = pytest.mark.skipif(
    torch.cuda.is_available() is False, reason=CUDA_SKIP_REASON
)


# ---------------------------------------------------------------------------
# Skip / metric helpers
# ---------------------------------------------------------------------------


def _require_cuda_device() -> torch.device:
    """Skip unless a Volta sm_70 CUDA device is present; else return it."""
    if torch.cuda.is_available() is False:
        pytest.skip(CUDA_SKIP_REASON)
    device = torch.device("cuda")
    try:
        capability = torch.cuda.get_device_capability(device)
    except Exception as exc:  # pragma: no cover - CUDA-host only
        pytest.skip(f"CUDA device capability unreadable ({exc}); V100S sm_70 required.")
    if tuple(capability) != (7, 0):
        pytest.skip(
            f"cuda-fused is qualified only for Volta sm_70; found sm_{capability[0]}{capability[1]}. "
            "Run the CPU reference smoke suite on this host."
        )
    return device


def _require_fused_backend(device: torch.device) -> RecurrenceOptions:
    """Skip unless the cuda-fused extension loads on *device*."""
    options = RecurrenceOptions(
        backend="cuda-fused", input_dtype="float32", state_accumulator_dtype="float32"
    )
    try:
        ensure_backend_available(options, device=device)
    except BackendUnavailableError as exc:
        pytest.skip(f"cuda-fused backend unavailable, skipping qualification: {exc}")
    return options


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _normalized_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(b.pow(2).mean().item()) ** 0.5
    if denom == 0.0:
        return float(a.pow(2).mean().item() ** 0.5)
    return float(((a - b).pow(2).mean().item() ** 0.5) / (denom + 1e-12))


def _assert_all_finite(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        assert torch.isfinite(tensor).all(), "non-finite values are never acceptable"


def _make_fp32_inputs(
    *,
    batch: int = 2,
    heads: int = 2,
    head_dim: int,
    seq: int,
    channel_wise: bool,
    seed: int = 1234,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(
            shape, generator=generator, dtype=torch.float32, requires_grad=requires_grad
        ).to(device)

    q = rand(batch, seq, heads, head_dim)
    k = rand(batch, seq, heads, head_dim)
    v = rand(batch, seq, heads, head_dim)
    width = head_dim if channel_wise else 1
    alpha = torch.sigmoid(rand(batch, seq, heads, width)).detach().to(device)
    beta = torch.sigmoid(rand(batch, seq, heads, 1)).detach().to(device)
    if requires_grad:
        alpha.requires_grad_()
        beta.requires_grad_()
    return q, k, v, alpha, beta


def _tiny_qualification_config() -> ModelConfig:
    """Small pure-GDN backbone whose head dim (40) is a production dim."""
    return ModelConfig(
        "qual-tiny", d_model=80, n_layer=2, n_head=2, d_ff=128,
        attention_layout="pure_gdn",
    )


def _run_optimizer_steps(
    model, batches: list[tuple[torch.Tensor, torch.Tensor]], *, steps: int = 5
) -> list[float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
    losses: list[float] = []
    for step in range(steps):
        tokens, targets = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(tokens, targets)
        assert loss is not None
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"missing gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"
        optimizer.step()
        value = float(loss.item())
        assert value == value and value != float("inf"), "non-finite loss"
        losses.append(value)
    return losses


def _synthetic_batches(
    *, count: int = 5, batch: int = 2, seq: int = 16, vocab: int = 128,
    seed: int = 999, device: torch.device | str = "cpu",
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for _ in range(count):
        tokens = torch.randint(0, vocab, (batch, seq), generator=generator).to(device)
        out.append((tokens, tokens.clone()))
    return out


# ---------------------------------------------------------------------------
# 5.1 FP32 forward / final-state equivalence (CUDA-only)
# ---------------------------------------------------------------------------


@needs_cuda
@pytest.mark.parametrize("channel_wise", [False, True])
@pytest.mark.parametrize("head_dim", PRODUCTION_HEAD_DIMS)
@pytest.mark.parametrize("seq_len", SEQ_LENS_FWD)
@pytest.mark.parametrize("with_init", [False, True])
def test_cuda_fp32_forward_final_state_equivalence(
    channel_wise: bool, head_dim: int, seq_len: int, with_init: bool
) -> None:
    device = _require_cuda_device()
    fused_options = _require_fused_backend(device)
    ref_options = RecurrenceOptions(
        backend="reference", input_dtype="float32", state_accumulator_dtype="float32"
    )
    q, k, v, alpha, beta = _make_fp32_inputs(
        head_dim=head_dim, seq=seq_len, channel_wise=channel_wise, device=device
    )
    initial = None
    if with_init:
        generator = torch.Generator(device="cpu").manual_seed(77)
        initial = torch.randn(
            2, 2, head_dim, head_dim, generator=generator, dtype=torch.float32
        ).to(device)
    ref_out, ref_state = run_recurrence(q, k, v, alpha, beta, initial, options=ref_options)
    fused_out, fused_state = run_recurrence(
        q, k, v, alpha, beta, initial, options=fused_options
    )
    _assert_all_finite(ref_out, ref_state, fused_out, fused_state)
    assert _max_abs(ref_out, fused_out) <= FP32_FWD_MAX_ABS
    assert _max_abs(ref_state, fused_state) <= FP32_FWD_MAX_ABS


# ---------------------------------------------------------------------------
# 5.2 Backward-gradient equivalence incl. unreduced channel-wise (CUDA-only)
# ---------------------------------------------------------------------------


@needs_cuda
@pytest.mark.parametrize("channel_wise", [False, True])
@pytest.mark.parametrize("head_dim", PRODUCTION_HEAD_DIMS)
def test_cuda_fp32_backward_gradient_equivalence(channel_wise: bool, head_dim: int) -> None:
    device = _require_cuda_device()
    fused_options = _require_fused_backend(device)
    ref_options = RecurrenceOptions(
        backend="reference", input_dtype="float32", state_accumulator_dtype="float32"
    )
    seed = 2024
    ref_inputs = _make_fp32_inputs(
        head_dim=head_dim, seq=SEQ_LEN_GRAD, channel_wise=channel_wise,
        seed=seed, device=device, requires_grad=True,
    )
    fused_inputs = _make_fp32_inputs(
        head_dim=head_dim, seq=SEQ_LEN_GRAD, channel_wise=channel_wise,
        seed=seed, device=device, requires_grad=True,
    )
    gen = torch.Generator(device="cpu").manual_seed(5)
    ref_init = torch.randn(
        2, 2, head_dim, head_dim, generator=gen, dtype=torch.float32,
        device=device, requires_grad=True,
    )
    gen = torch.Generator(device="cpu").manual_seed(5)
    fused_init = torch.randn(
        2, 2, head_dim, head_dim, generator=gen, dtype=torch.float32,
        device=device, requires_grad=True,
    )
    ref_out, ref_state = run_recurrence(*ref_inputs, ref_init, options=ref_options)
    fused_out, fused_state = run_recurrence(*fused_inputs, fused_init, options=fused_options)
    ref_loss = ref_out.square().mean() + ref_state.square().mean()
    fused_loss = fused_out.square().mean() + fused_state.square().mean()
    ref_loss.backward()
    fused_loss.backward()
    names = ("q", "k", "v", "alpha", "beta", "initial_state")
    for name, ref_t, fused_t in zip(names, (*ref_inputs, ref_init), (*fused_inputs, fused_init)):
        assert ref_t.grad is not None, f"reference missing grad for {name}"
        assert fused_t.grad is not None, f"fused missing grad for {name} (no silent fallback)"
        _assert_all_finite(ref_t.grad, fused_t.grad)
        assert _normalized_l2(fused_t.grad, ref_t.grad) <= GRAD_NORM_L2_TOL, name
        assert _max_abs(fused_t.grad, ref_t.grad) <= GRAD_MAX_ABS_TOL, name


@needs_cuda
def test_cuda_channelwise_alpha_gradients_unreduced() -> None:
    device = _require_cuda_device()
    fused_options = _require_fused_backend(device)
    ref_options = RecurrenceOptions(
        backend="reference", input_dtype="float32", state_accumulator_dtype="float32"
    )
    head_dim = PRODUCTION_HEAD_DIMS[0]
    ref_inputs = _make_fp32_inputs(
        head_dim=head_dim, seq=SEQ_LEN_GRAD, channel_wise=True,
        seed=31337, device=device, requires_grad=True,
    )
    fused_inputs = _make_fp32_inputs(
        head_dim=head_dim, seq=SEQ_LEN_GRAD, channel_wise=True,
        seed=31337, device=device, requires_grad=True,
    )
    ref_out, _ = run_recurrence(*ref_inputs, None, options=ref_options)
    fused_out, _ = run_recurrence(*fused_inputs, None, options=fused_options)
    ref_loss = ref_out.square().mean()
    fused_loss = fused_out.square().mean()
    ref_loss.backward()
    fused_loss.backward()
    ref_alpha, fused_alpha = ref_inputs[3], fused_inputs[3]
    assert ref_alpha.grad is not None and fused_alpha.grad is not None
    # Unreduced: one gradient per [B,T,H,D] element, never head-wise reduced.
    assert tuple(fused_alpha.grad.shape) == tuple(ref_alpha.grad.shape) == (2, SEQ_LEN_GRAD, 2, head_dim)
    assert fused_alpha.grad.shape[-1] == head_dim != 1
    # Per-channel variation must survive (a head-scalar reduction would collapse it).
    assert float(ref_alpha.grad.var().item()) > 0.0
    assert _normalized_l2(fused_alpha.grad, ref_alpha.grad) <= GRAD_NORM_L2_TOL
    assert _max_abs(fused_alpha.grad, ref_alpha.grad) <= GRAD_MAX_ABS_TOL


# ---------------------------------------------------------------------------
# 5.3 Split-sequence continuation (CUDA-only)
# ---------------------------------------------------------------------------


@needs_cuda
@pytest.mark.parametrize("channel_wise", [False, True])
@pytest.mark.parametrize("head_dim", PRODUCTION_HEAD_DIMS)
def test_cuda_split_sequence_continuation(channel_wise: bool, head_dim: int) -> None:
    device = _require_cuda_device()
    fused_options = _require_fused_backend(device)
    ref_options = RecurrenceOptions(
        backend="reference", input_dtype="float32", state_accumulator_dtype="float32"
    )
    split = SPLIT_LEN // 2
    for options in (ref_options, fused_options):
        q, k, v, alpha, beta = _make_fp32_inputs(
            head_dim=head_dim, seq=SPLIT_LEN, channel_wise=channel_wise, device=device
        )
        full_out, full_state = run_recurrence(q, k, v, alpha, beta, None, options=options)
        first_out, mid_state = run_recurrence(
            q[:, :split], k[:, :split], v[:, :split], alpha[:, :split], beta[:, :split],
            None, options=options,
        )
        second_out, cont_state = run_recurrence(
            q[:, split:], k[:, split:], v[:, split:], alpha[:, split:], beta[:, split:],
            mid_state, options=options,
        )
        joined = torch.cat((first_out, second_out), dim=1)
        _assert_all_finite(full_out, full_state, joined, cont_state)
        assert _max_abs(joined, full_out) <= FP32_FWD_MAX_ABS
        assert _max_abs(cont_state, full_state) <= FP32_FWD_MAX_ABS


# ---------------------------------------------------------------------------
# 5.4 Prefill / cached-decode equivalence (CUDA-only)
# ---------------------------------------------------------------------------


@needs_cuda
@pytest.mark.parametrize("variant", ["scalar", "channel-wise", "scalar-matched"])
def test_cuda_prefill_cached_decode_equivalence(variant: str) -> None:
    device = _require_cuda_device()
    _require_fused_backend(device)  # gates on extension; model uses fused options below
    backbone = _tiny_qualification_config()
    fused_options = RecurrenceOptions(
        backend="cuda-fused", input_dtype="float32", state_accumulator_dtype="float32"
    )
    model, _, _ = construct_model(
        backbone, variant, 128, recurrence_options=fused_options
    )
    model.to(device).eval()
    generator = torch.Generator(device="cpu").manual_seed(11)
    tokens = torch.randint(0, 128, (2, 16), generator=generator).to(device)
    error = cache_equivalence_error(model, tokens)
    assert error == error and error != float("inf"), "non-finite cache equivalence"
    assert error <= FP32_FWD_MAX_ABS


# ---------------------------------------------------------------------------
# 5.5 Paired 5-step FP32 trajectory + mixed-precision stability (CUDA-only)
# ---------------------------------------------------------------------------


@needs_cuda
def test_cuda_paired_five_step_fp32_trajectory() -> None:
    device = _require_cuda_device()
    fused_options = _require_fused_backend(device)
    ref_options = RecurrenceOptions(
        backend="reference", input_dtype="float32", state_accumulator_dtype="float32"
    )
    backbone = _tiny_qualification_config()
    seed_everything(12345)
    ref_model, _, _ = construct_model(backbone, "scalar", 128, recurrence_options=ref_options)
    state = {k: v.clone() for k, v in ref_model.state_dict().items()}
    fused_model, _, _ = construct_model(
        backbone, "scalar", 128, recurrence_options=fused_options
    )
    fused_model.load_state_dict(state)
    ref_model.to(device)
    fused_model.to(device)
    batches = _synthetic_batches(device=device)
    ref_losses = _run_optimizer_steps(ref_model, batches, steps=5)
    fused_losses = _run_optimizer_steps(fused_model, batches, steps=5)
    assert len(ref_losses) == len(fused_losses) == 5
    for step, (ref_loss, fused_loss) in enumerate(zip(ref_losses, fused_losses)):
        assert abs(ref_loss - fused_loss) <= TRAJECTORY_LOSS_TOL, f"step {step} diverged"


@needs_cuda
@pytest.mark.parametrize("variant", ["scalar", "channel-wise"])
def test_cuda_mixed_precision_stability_guard(variant: str) -> None:
    device = _require_cuda_device()
    fused_options = _require_fused_backend(device)
    backbone = _tiny_qualification_config()
    seed_everything(777)
    model, _, _ = construct_model(
        backbone, variant, 128,
        recurrence_options=RecurrenceOptions(
            backend="cuda-fused",
            input_dtype=fused_options.input_dtype,
            state_accumulator_dtype=fused_options.state_accumulator_dtype,
        ),
    )
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
    batches = _synthetic_batches(count=2, device=device)
    with torch.autocast(device.type, dtype=torch.float16):
        for tokens, targets in batches:
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(tokens, targets)
            assert loss is not None, "missing loss under autocast (fallback?)"
            loss.backward()
            for name, param in model.named_parameters():
                assert param.grad is not None, f"missing grad for {name} under autocast"
                assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"
            assert torch.isfinite(loss).all(), "non-finite mixed-precision loss"
            optimizer.step()
    # Cached decode must also stay within the versioned mixed-precision tolerance.
    model.eval()
    with torch.autocast(device.type, dtype=torch.float16):
        tokens = batches[0][0]
        assert cache_equivalence_error(model, tokens) <= MIXED_MAX_ABS


# ---------------------------------------------------------------------------
# CPU-side verification of the same harnesses (tasks 5.x logic + 7.5)
# These MUST pass on the laptop; CUDA tests above MUST skip there.
# ---------------------------------------------------------------------------


def test_metric_helpers_behave_on_cpu() -> None:
    a = torch.ones(2, 3, dtype=torch.float64)
    b = torch.ones(2, 3, dtype=torch.float64) + 1e-5
    assert _max_abs(a, b) == pytest.approx(1e-5)
    assert _normalized_l2(a, b) < 1e-4
    assert _normalized_l2(a, a) == pytest.approx(0.0)


def test_reference_split_sequence_harness_on_cpu() -> None:
    for channel_wise in (False, True):
        q, k, v, alpha, beta = _make_fp32_inputs(
            head_dim=8, seq=32, channel_wise=channel_wise, device="cpu"
        )
        full_out, full_state = reference_recurrence(q, k, v, alpha, beta)
        first, mid = reference_recurrence(q[:, :12], k[:, :12], v[:, :12], alpha[:, :12], beta[:, :12])
        second, cont = reference_recurrence(
            q[:, 12:], k[:, 12:], v[:, 12:], alpha[:, 12:], beta[:, 12:], mid
        )
        assert _max_abs(torch.cat((first, second), dim=1), full_out) <= FP32_FWD_MAX_ABS
        assert _max_abs(cont, full_state) <= FP32_FWD_MAX_ABS


@pytest.mark.parametrize("variant", ["scalar", "channel-wise", "scalar-matched"])
def test_reference_prefill_cached_decode_harness_on_cpu(variant: str) -> None:
    backbone = _tiny_qualification_config()
    model, _, _ = construct_model(backbone, variant, 128)
    model.eval()
    tokens = torch.randint(0, 128, (2, 9))
    assert cache_equivalence_error(model, tokens) < 1e-5


def test_paired_trajectory_harness_reproducible_on_cpu() -> None:
    backbone = _tiny_qualification_config()
    seed_everything(4242)
    first, _, _ = construct_model(backbone, "scalar", 128)
    state = {k: v.clone() for k, v in first.state_dict().items()}
    second, _, _ = construct_model(backbone, "scalar", 128)
    second.load_state_dict(state)
    batches = _synthetic_batches()
    first_losses = _run_optimizer_steps(first, batches, steps=5)
    second_losses = _run_optimizer_steps(second, batches, steps=5)
    assert len(first_losses) == 5
    for a, b in zip(first_losses, second_losses):
        assert abs(a - b) <= TRAJECTORY_LOSS_TOL


def test_cpu_mixed_precision_stability_guard_documents_tolerance() -> None:
    backbone = _tiny_qualification_config()
    model, _, _ = construct_model(backbone, "channel-wise", 128)
    model.eval()
    tokens = torch.randint(0, 128, (1, 8))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert cache_equivalence_error(model, tokens) < MIXED_MAX_ABS


def test_cuda_tests_skip_with_actionable_reason_on_cpu() -> None:
    """Task 5.6: CUDA cases skip (never fake-pass) on CPU; reference stays mandatory."""
    if torch.cuda.is_available():
        pytest.skip("CUDA present; skip-behavior check applies to the CPU laptop only.")
    assert "sm_70" in CUDA_SKIP_REASON
    assert "gdn_cuda" in CUDA_SKIP_REASON
    assert "reference" in CUDA_SKIP_REASON
    cuda_tests = [
        obj for name, obj in globals().items()
        if name.startswith("test_cuda_") and callable(obj)
        and name != "test_cuda_tests_skip_with_actionable_reason_on_cpu"
    ]
    assert len(cuda_tests) >= 7, "expected all 5.x CUDA qualification cases to exist"
    for fn in cuda_tests:
        marks = getattr(fn, "pytestmark", [])
        assert any(m.name == "skipif" for m in marks), f"{fn.__name__} lacks skipif marker"
    # Reference smoke remains runnable and mandatory on CPU.
    q, k, v, alpha, beta = _make_fp32_inputs(
        head_dim=8, seq=8, channel_wise=False, device="cpu"
    )
    out, state = reference_recurrence(q, k, v, alpha, beta)
    _assert_all_finite(out, state)
