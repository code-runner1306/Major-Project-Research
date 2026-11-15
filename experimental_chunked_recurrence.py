"""Unqualified PyTorch chunked GDN candidate; never selected by production.

Uses triangular solves rather than one host-dispatched update per token. The
state remains value-by-key and decay acts on key columns. Interval products
avoid division by cumulative decay, including when a decay is exactly zero.
"""
from __future__ import annotations

import torch

from recurrence import RecurrenceOptions, _validate_inputs, _validate_options, _state_dtype


def chunked_recurrence(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    alpha: torch.Tensor, beta: torch.Tensor,
    initial_state: torch.Tensor | None = None, *,
    options: RecurrenceOptions = RecurrenceOptions(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact-arithmetic equivalent of reference_recurrence, with autograd.

    Within a chunk, write residuals satisfy a unit lower-triangular system.
    For i >= j, P[i,j] is the product of decay gates j+1 through i. Thus
    S_i = S_start * prod(alpha[:i+1]) + sum_j beta_j e_j (k_j*P[i,j])^T.
    Substitution into e_i = v_i - (alpha_i*S_{i-1}) k_i yields the solve.
    All algebra runs in the declared state dtype; outputs use the Q dtype.
    """
    _validate_options(options)
    batch, length, heads, width = _validate_inputs(q, k, v, alpha, beta, initial_state)
    if length == 0:
        raise ValueError('Chunked recurrence requires a nonempty sequence')
    dtype = _state_dtype(q, options)
    qh, kh, vh, ah, bh = (
        x.to(dtype).transpose(1, 2) for x in (q, k, v, alpha, beta)
    )
    state = (torch.zeros(batch, heads, width, width, dtype=dtype, device=q.device)
             if initial_state is None else initial_state.to(dtype))
    outputs = []
    for start in range(0, length, options.chunk_size):
        stop = min(start + options.chunk_size, length)
        qc, kc, vc, ac, bc = (x[:, :, start:stop] for x in (qh, kh, vh, ah, bh))
        size = stop - start
        index = torch.arange(size, device=q.device)
        lower = index[:, None] > index[None, :]
        causal = index[:, None] >= index[None, :]
        # [B,H,destination,source,gate_width]. No ratios or inverse gates.
        factors = torch.where(lower[None, None, :, :, None], ac.unsqueeze(-2), 1.)
        intervals = factors.cumprod(dim=2)
        cumulative = ac.cumprod(dim=2)
        # Scalar gates use ordinary matmul, avoiding a width-sized expansion.
        if ac.size(-1) == 1:
            coupling = kc @ kc.transpose(-1, -2) * intervals.squeeze(-1)
            readout = qc @ kc.transpose(-1, -2) * intervals.squeeze(-1)
        else:
            coupling = (kc.unsqueeze(-2) * kc.unsqueeze(-3) * intervals).sum(-1)
            readout = (qc.unsqueeze(-2) * kc.unsqueeze(-3) * intervals).sum(-1)
        coupling = coupling * bc.squeeze(-1).unsqueeze(-2)
        matrix = torch.eye(size, device=q.device, dtype=dtype) + coupling.masked_fill(~lower, 0.)
        rhs = vc - (kc * cumulative) @ state.transpose(-1, -2)
        residual = torch.linalg.solve_triangular(matrix, rhs, upper=False, unitriangular=True)
        writes = residual * bc
        output = (qc * cumulative) @ state.transpose(-1, -2) + readout.masked_fill(~causal, 0.) @ writes
        state = state * cumulative[:, :, -1].unsqueeze(-2) + writes.transpose(-1, -2) @ (kc * intervals[:, :, -1])
        outputs.append(output)
    return torch.cat(outputs, dim=2).transpose(1, 2).to(q.dtype), state


def install_candidate(model: torch.nn.Module, *, chunk_size: int = 16) -> None:
    """Replace GDN mixers for an isolated diagnostic, preserving every weight.

    Does not register or qualify a production backend. The caller must record
    the candidate source hash and all numerical/performance results.
    """
    from dataclasses import replace
    from model import GatedDeltaNetAttention

    class ChunkedAttention(GatedDeltaNetAttention):
        def _recurrent_update(self, q, k, v, alpha, beta, initial_state, *, options):
            return chunked_recurrence(q, k, v, alpha, beta, initial_state,
                                      options=replace(options, chunk_size=chunk_size))

    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    for block in model.blocks:
        old = block.attn
        if not isinstance(old, GatedDeltaNetAttention):
            continue
        if old._gate_observer is not None or old._runtime_gate_groups is not None:
            raise ValueError('Candidate installation requires uninstrumented diagnostic models')
        with torch.random.fork_rng(devices=[]):
            candidate = ChunkedAttention(old.d_model, old.n_head, old.d_head, old.variant,
                old.recurrence_options, gate_groups=old.gate_groups,
                calibrated_initialization=old.alpha_proj.bias is not None)
        parameter = next(old.parameters())
        candidate.to(device=parameter.device, dtype=parameter.dtype)
        candidate.load_state_dict(old.state_dict())
        candidate.layer_idx = old.layer_idx
        candidate.train(old.training)
        block.attn = candidate
