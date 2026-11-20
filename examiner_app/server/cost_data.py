"""Live parameter and per-token cost of the two recipe models, built from recipe_model.RecipeLM on CPU.

Nothing is trained. Models are constructed once (seed 42, same settings as the recipe timing runs) and cached.
The global torch RNG state is saved and restored, because RecipeLM reseeds torch when it is built.
"""
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from recipe_model import RecipeLM, RecipeGDN  # noqa: E402

ARCHITECTURES = ('scalar', 'channel-wise')
RECIPE = dict(vocabulary=8192, length=64, width=128, backend='reference', seed=42)
FP32_BYTES = 4


def _build(architecture):
    rng = torch.get_rng_state()
    try:
        return RecipeLM(architecture=architecture, **RECIPE)
    finally:
        torch.set_rng_state(rng)


def _architecture_facts(architecture):
    model = _build(architecture)
    total = sum(p.numel() for p in model.parameters())  # each tensor counted once
    layers = []
    for name, module in model.named_modules():
        if not isinstance(module, RecipeGDN):
            continue
        a_proj = module.a_proj.weight
        state_shape = (module.heads, module.head_v, module.head_k)
        state_values = module.heads * module.head_v * module.head_k
        layers.append({
            'layer': module.layer_idx + 1,
            'gate_width': module.gate_width,
            'a_proj_parameters': a_proj.numel(),
            'a_proj_shape': list(a_proj.shape),
            'A_log_parameters': module.A_log.numel(),
            'dt_bias_parameters': module.dt_bias.numel(),
            'gate_parameters': a_proj.numel() + module.A_log.numel() + module.dt_bias.numel(),
            'gate_macs_per_token': module.width * module.heads * module.gate_width,
            'state_shape': list(state_shape),
            'state_bytes_fp32': state_values * FP32_BYTES,
        })
    linear_macs = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            linear_macs += module.in_features * module.out_features
    return {
        'architecture': architecture,
        'total_parameters': total,
        'gate_parameters': sum(layer['gate_parameters'] for layer in layers),
        'layers': layers,
        'gate_macs_per_token': sum(layer['gate_macs_per_token'] for layer in layers),
        'linear_macs_per_token': linear_macs,
        'state_bytes_fp32': sum(layer['state_bytes_fp32'] for layer in layers),
    }


@lru_cache(maxsize=1)
def parameters():
    """Live numbers for the Cost page: totals, the per-layer gate, the difference, and per-token gate work."""
    facts = {arch: _architecture_facts(arch) for arch in ARCHITECTURES}
    scalar, channel = facts['scalar'], facts['channel-wise']
    delta = channel['total_parameters'] - scalar['total_parameters']
    extra_macs = channel['gate_macs_per_token'] - scalar['gate_macs_per_token']
    return {
        'architectures': facts,
        'delta': {
            'parameters': delta,
            'percent_of_scalar': 100.0 * delta / scalar['total_parameters'],
            'per_layer_parameters': delta // len(channel['layers']),
            'extra_gate_macs_per_token': extra_macs,
            'extra_gate_macs_share_of_linear': extra_macs / scalar['linear_macs_per_token'],
            'state_bytes_difference': channel['state_bytes_fp32'] - scalar['state_bytes_fp32'],
        },
        'settings': {**RECIPE, 'layers': len(scalar['layers']), 'heads': 1},
        'source': 'live: recipe_model.RecipeLM built on CPU (seed 42), counted from its tensors',
    }
