"""Online gate and static-cost diagnostics kept outside benchmark timing regions."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable

import torch

from controls import architecture_contract, channel_wise_gate_overhead, gate_capacity_delta, gdn_layer_count
from model import GatedDeltaNetLM, parameter_breakdown


class GateDiagnostics:
    def __init__(self, reservoir_size: int = 4096, saturation_threshold: float = 0.05):
        self.reservoir_size = reservoir_size
        self.saturation_threshold = saturation_threshold
        self.stats: dict[int, dict[str, Any]] = defaultdict(lambda: {
            "count": 0, "sum": 0.0, "sum_sq": 0.0, "near_zero": 0, "near_one": 0,
            "half_life_values": [], "values": [], "channel_diversity_sum": 0.0, "channel_diversity_count": 0,
            "variant": None,
        })

    def observe(self, alpha: torch.Tensor, variant: str, layer: int) -> None:
        values = alpha.detach().float().reshape(-1).cpu()
        entry = self.stats[layer]
        entry["variant"] = variant
        entry["count"] += values.numel()
        entry["sum"] += values.sum().item()
        entry["sum_sq"] += values.square().sum().item()
        entry["near_zero"] += (values <= self.saturation_threshold).sum().item()
        entry["near_one"] += (values >= 1 - self.saturation_threshold).sum().item()
        capped = values.clamp(1e-6, 1 - 1e-6)
        half_life = (math.log(0.5) / torch.log(capped)).clamp(max=1_000_000)
        self._append_reservoir(entry["values"], values)
        self._append_reservoir(entry["half_life_values"], half_life)
        if variant == "channel-wise":
            diversity = alpha.detach().float().var(dim=-1, unbiased=False).mean().item()
            entry["channel_diversity_sum"] += diversity
            entry["channel_diversity_count"] += 1

    def _append_reservoir(self, reservoir: list[float], values: torch.Tensor) -> None:
        remaining = self.reservoir_size - len(reservoir)
        if remaining <= 0:
            return
        if values.numel() > remaining:
            indices = torch.linspace(0, values.numel() - 1, remaining).long()
            values = values[indices]
        reservoir.extend(values.tolist())

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for layer, entry in sorted(self.stats.items()):
            count = entry["count"]
            mean = entry["sum"] / count if count else None
            variance = max(0.0, entry["sum_sq"] / count - (mean or 0.0) ** 2) if count else None
            values = torch.tensor(entry["values"], dtype=torch.float32)
            half_lives = torch.tensor(entry["half_life_values"], dtype=torch.float32)
            result[str(layer)] = {
                "variant": entry["variant"], "count": count, "mean": mean,
                "std": math.sqrt(variance) if variance is not None else None,
                "quantiles": {str(q): torch.quantile(values, q).item() if values.numel() else None for q in (0.05, 0.5, 0.95)},
                "near_zero_fraction": entry["near_zero"] / count if count else None,
                "near_one_fraction": entry["near_one"] / count if count else None,
                "half_life_quantiles": {str(q): torch.quantile(half_lives, q).item() if half_lives.numel() else None for q in (0.05, 0.5, 0.95)},
                "channel_diversity": (
                    entry["channel_diversity_sum"] / entry["channel_diversity_count"]
                    if entry["channel_diversity_count"] else None
                ),
            }
        return result


@torch.no_grad()
def collect_gate_diagnostics(
    model: GatedDeltaNetLM, batches: Iterable[torch.Tensor], device: torch.device
) -> dict[str, Any]:
    collector = GateDiagnostics()
    model.set_gate_observer(collector.observe)
    was_training = model.training
    model.eval()
    try:
        for tokens in batches:
            model(tokens.to(device))
    finally:
        model.set_gate_observer(None)
        model.train(was_training)
    return {"layers": collector.summary(), "diagnostic_hooks_in_timed_regions": False}


def static_cost_report(model: GatedDeltaNetLM, *, batch_size: int, context_length: int) -> dict[str, Any]:
    cfg = model.model_config
    width = cfg.d_head
    element_bytes = next(model.parameters()).element_size()
    recurrent_state_bytes = gdn_layer_count(cfg) * batch_size * cfg.n_head * width * width * element_bytes
    recurrent_conv_bytes = gdn_layer_count(cfg) * 3 * batch_size * 3 * cfg.d_model * element_bytes
    attention_layers = cfg.n_layer - gdn_layer_count(cfg)
    kv_cache_bytes = attention_layers * 2 * batch_size * cfg.n_head * context_length * width * element_bytes
    return {
        "parameter_components": parameter_breakdown(model),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "tied_output_parameter_count": 0,
        "architecture_contract": architecture_contract(
            cfg, variant=model.variant, gate_initialization=model.gate_initialization,
        ),
        "gate_parameter_overhead": gate_capacity_delta(
            cfg, model.variant, gate_groups=model.gate_groups,
            gate_initialization=model.gate_initialization,
        ),
        "theoretical_gate_projection_ops_per_token": channel_wise_gate_overhead(cfg) // max(gdn_layer_count(cfg), 1),
        "recurrent_state_bytes": recurrent_state_bytes,
        "recurrent_convolution_cache_bytes": recurrent_conv_bytes,
        "attention_kv_cache_bytes_at_context": kv_cache_bytes,
        "cache_formula": "GDN: layers * batch * heads * head_dim^2; attention: layers * 2 * batch * heads * context * head_dim",
    }
