"""Compare the real decay-before-residual update and every input gradient."""
import copy
import pytest
import torch
from torch.nn import functional as F

from experimental_chunked_recurrence import chunked_recurrence
from recurrence import RecurrenceOptions, reference_recurrence


@pytest.mark.parametrize('gate_width', [1, 5])
@pytest.mark.parametrize('chunk_size', [1, 4, 16])
@pytest.mark.parametrize('zero_gate', [False, True])
def test_chunked_matches_output_state_and_all_gradients(gate_width, chunk_size, zero_gate):
    torch.manual_seed(781)
    q, k = [F.normalize(torch.randn(2, 9, 2, 5, dtype=torch.float64), dim=-1) for _ in range(2)]
    v = torch.randn_like(q)
    alpha = torch.rand(2, 9, 2, gate_width, dtype=torch.float64)
    if zero_gate:
        alpha[:, 3] = 0
    beta = torch.rand(2, 9, 2, 1, dtype=torch.float64)
    state = torch.randn(2, 2, 5, 5, dtype=torch.float64)
    options = RecurrenceOptions(chunk_size=chunk_size, state_accumulator_dtype='input')
    inputs = [x.requires_grad_() for x in (q, k, v, alpha, beta, state)]
    ref_output, ref_state = reference_recurrence(*inputs, options=options)
    output, final = chunked_recurrence(*inputs, options=options)
    torch.testing.assert_close(output, ref_output, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(final, ref_state, atol=1e-11, rtol=1e-11)
    weights = torch.randn_like(output)
    state_weights = torch.randn_like(final)
    reference_grads = torch.autograd.grad((ref_output * weights).sum() + (ref_state * state_weights).sum(), inputs)
    grads = torch.autograd.grad((output * weights).sum() + (final * state_weights).sum(), inputs)
    for result, expected in zip(grads, reference_grads):
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('gate_width', [1, 8])
def test_chunked_fp32_and_split_continuation(gate_width):
    torch.manual_seed(5)
    q, k = [F.normalize(torch.randn(2, 33, 2, 8), dim=-1) for _ in range(2)]
    v = torch.randn_like(q)
    alpha = torch.rand(2, 33, 2, gate_width)
    beta = torch.rand(2, 33, 2, 1)
    options = RecurrenceOptions(chunk_size=8, state_accumulator_dtype='float32')
    inputs = (q, k, v, alpha, beta)
    expected, expected_state = reference_recurrence(*inputs, options=options)
    output, state = chunked_recurrence(*inputs, options=options)
    prefix, prefix_state = chunked_recurrence(*(x[:, :13] for x in inputs), options=options)
    suffix, split_state = chunked_recurrence(*(x[:, 13:] for x in inputs), prefix_state, options=options)
    for actual, reference in ((output, expected), (state, expected_state),
                              (torch.cat([prefix, suffix], dim=1), output), (split_state, state)):
        torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('variant', ['scalar', 'channel-wise'])
def test_real_model_optimizer_trajectory_and_cache(variant):
    from config import ModelConfig
    from model import GatedDeltaNetLM
    from experimental_chunked_recurrence import install_candidate

    torch.manual_seed(177)
    model = GatedDeltaNetLM(ModelConfig('candidate-test', 16, 2, 2, 32, 'pure_gdn'),
                           variant, 64, RecurrenceOptions(state_accumulator_dtype='float32'),
                           study_init_seed=177)
    candidate = copy.deepcopy(model)
    install_candidate(candidate, chunk_size=4)
    assert set(candidate.state_dict()) == set(model.state_dict())
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3) for m in (model, candidate)]
    for _ in range(5):
        tokens = torch.randint(0, 64, (2, 13))
        labels = torch.randint(0, 64, (2, 13))
        for m, optimizer in zip((model, candidate), optimizers):
            optimizer.zero_grad(set_to_none=True)
            _, loss = m(tokens, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
            optimizer.step()
        for actual, expected in zip(candidate.parameters(), model.parameters()):
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    tokens = torch.randint(0, 64, (2, 13))
    with torch.no_grad():
        full, _ = candidate(tokens)
        prefix, _, cache = candidate(tokens[:, :7], use_cache=True)
        suffix, _, _ = candidate(tokens[:, 7:], cache=cache, use_cache=True)
    torch.testing.assert_close(torch.cat((prefix, suffix), 1), full, atol=2e-6, rtol=2e-5)
