"""Forward pass with captured tensors: per-layer gates, token-by-token recurrent state, layer-4 crops.

Ported from legacy_prototype/export_data.py (mixer_inputs, mixer_tensors) and the serial recurrence in
recipe_model.serial_rectangular. The token loop here must agree with serial_rectangular; this is asserted.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recipe_model import published_l2norm, serial_rectangular  # noqa: E402

RECURRENCE_TOLERANCE = 1e-5
CROP = 32


def mixer_inputs(model, tokens):
    """Input x of every GDN sequence mixer, captured with pre-hooks during one network forward."""
    captured = {}
    hooks = [blk.sequence_mixer.register_forward_pre_hook(
        lambda m, a, idx=idx: captured.__setitem__(idx, a[0].detach()))
        for idx, blk in enumerate(model.network.backbone.layers)]
    try:
        logits = model.network(tokens)
    finally:
        for h in hooks:
            h.remove()
    return captured, logits


def mixer_tensors(mixer, x):
    """q, k, v, alpha, beta exactly as RecipeGDN.forward computes them (q already scaled by head_k**-.5)."""
    def projected(projection, convolution, width):
        raw = projection(x).transpose(1, 2)
        return F.silu(convolution(F.pad(raw, (3, 0))).transpose(1, 2)).reshape(x.size(0), x.size(1), mixer.heads, width)
    q = published_l2norm(projected(mixer.q_proj, mixer.q_conv, mixer.head_k)) * mixer.head_k ** -.5
    k = published_l2norm(projected(mixer.k_proj, mixer.k_conv, mixer.head_k))
    v = projected(mixer.v_proj, mixer.v_conv, mixer.head_v)
    alpha = mixer.decay(x)
    beta = mixer.b_proj(x).view(x.size(0), x.size(1), mixer.heads, 1).sigmoid()
    return q, k, v, alpha, beta


def _r(x, n):
    return [round(float(v), n) for v in x]


@torch.no_grad()
def trace(model, tokens):
    """Run the model on one (answer-free) token sequence [T] and record its recurrence.

    Returns dict(layers=[{layer, alpha [T][1|128], beta [T], state_norm [T]}], state_crops [T][32][32]
    (layer 4), recurrence_check_max_abs_diff). Raises RuntimeError if the token loop disagrees with
    serial_rectangular by more than RECURRENCE_TOLERANCE.
    """
    captured, _ = mixer_inputs(model, tokens[None])
    steps = tokens.numel()
    layers, crops, worst = [], None, 0.0
    for idx, blk in enumerate(model.network.backbone.layers):
        mixer = blk.sequence_mixer
        q, k, v, alpha, beta = mixer_tensors(mixer, captured[idx])
        S = q.new_zeros(v.size(-1), k.size(-1))            # [Dv, Dk] for the single head
        norms, layer_crops = [], []
        for t in range(steps):
            S = S * alpha[0, t, 0].unsqueeze(0)              # decay each key column
            k_t = k[0, t, 0]
            residual = v[0, t, 0] - S @ k_t
            S = S + beta[0, t, 0, 0] * torch.outer(residual, k_t)
            norms.append(float(S.norm()))
            if idx == len(model.network.backbone.layers) - 1:
                layer_crops.append(S[:CROP, :CROP].tolist())
        _, final = serial_rectangular(q, k, v, alpha, beta)
        worst = max(worst, float((S - final[0, 0]).abs().max()))
        layers.append(dict(
            layer=idx + 1,
            alpha=[_r(row, 4) for row in alpha[0, :, 0].tolist()],
            beta=_r(beta[0, :, 0, 0].tolist(), 4),
            state_norm=_r(norms, 4),
        ))
        if idx == len(model.network.backbone.layers) - 1:
            crops = [[_r(row, 3) for row in mat] for mat in layer_crops]
    if worst >= RECURRENCE_TOLERANCE:
        raise RuntimeError(f'Token-by-token recurrence differs from serial_rectangular by {worst:.3g}')
    return dict(layers=layers, state_crops=crops, recurrence_check_max_abs_diff=worst)
