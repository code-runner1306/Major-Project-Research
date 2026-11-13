"""Calibration utilities for matched geometric decay initialization."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from hashlib import sha256
from typing import Any

import torch
from torch import nn

from model import GatedDeltaNetAttention, GatedDeltaNetLM


def _tokens(batch: Any) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, Mapping):
        value = batch.get("tokens", batch.get("input_ids"))
    elif isinstance(batch, (tuple, list)):
        value = batch[0]
    else:
        value = None
    if not isinstance(value, torch.Tensor):
        raise ValueError("Calibration batches must contain token tensors")
    return value


def _calibration_mask(batch: Any, tokens: torch.Tensor) -> torch.Tensor | None:
    """Return an explicit non-padding mask when the caller supplies one."""
    if isinstance(batch, Mapping):
        value = batch.get("padding_mask", batch.get("attention_mask", batch.get("mask")))
        if isinstance(value, torch.Tensor):
            if value.shape != tokens.shape:
                raise ValueError("Calibration padding masks must match token shape")
            return value.to(dtype=torch.bool)
    return None


@torch.no_grad()
def calibrate_gate_initialization(
    model: GatedDeltaNetLM,
    batches: Iterable[Any],
    device: torch.device | str,
    min_half_life: float,
    max_half_life: float,
) -> dict[str, Any]:
    """Fit alpha biases so calibration-set mean retention matches geometric targets.

    Half-life targets are spaced geometrically over recurrent heads. Each head's
    groups/channels share identical initial projection functions and fitted bias.
    """
    if model.gate_initialization != "calibrated-geometric":
        raise ValueError("Model must be constructed with gate_initialization='calibrated-geometric'")
    if not (0 < min_half_life <= max_half_life):
        raise ValueError("Half-life bounds must satisfy 0 < min_half_life <= max_half_life")
    layers = [b.attn for b in model.blocks if isinstance(b.attn, GatedDeltaNetAttention)]
    if not layers:
        raise ValueError("Model has no recurrent gate layers")
    raw_batches = list(batches)
    token_batches = [_tokens(batch).to(device) for batch in raw_batches]
    if not token_batches:
        raise ValueError("At least one calibration token is required")
    was_training = model.training
    model.eval()
    explicit_masks = [_calibration_mask(batch, tokens) for batch, tokens in zip(raw_batches, token_batches)]
    if any(mask is not None for mask in explicit_masks) and any(mask is None for mask in explicit_masks):
        raise ValueError("Calibration requires a padding mask for every batch or none")
    if explicit_masks and explicit_masks[0] is not None:
        calibration_masks = [mask.to(device) for mask in explicit_masks if mask is not None]
        count = sum(int(mask.sum().item()) for mask in calibration_masks)
    else:
        calibration_masks = None
        count = sum(batch.size(0) * batch.size(1) for batch in token_batches)
    if count == 0:
        raise ValueError("At least one calibration token is required")

    head_count = layers[0].n_head
    targets = torch.logspace(
        torch.log10(torch.tensor(min_half_life, device=device)),
        torch.log10(torch.tensor(max_half_life, device=device)), head_count, device=device,
    )
    desired = torch.exp(-torch.log(torch.tensor(2.0, device=device)) / targets)
    report_layers = []
    try:
      for layer in layers:
        collected: list[torch.Tensor] = []
        def capture(module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            collected.append(args[0].detach().reshape(-1, args[0].size(-1)).to(device))
        handle = layer.alpha_proj.register_forward_pre_hook(capture)
        try:
            # Recompute fresh activations after calibrating each preceding layer.
            for batch_index, token_batch in enumerate(token_batches):
                model(token_batch)
        finally:
            handle.remove()
        inputs = torch.cat(collected, dim=0).to(layer.alpha_proj.weight)
        if calibration_masks is not None:
            flat_mask = torch.cat([mask.reshape(-1) for mask in calibration_masks])
            inputs = inputs[flat_mask]
        logits = F_linear_no_bias(layer.alpha_proj, inputs)
        n_gate = layer.alpha_proj.out_features // head_count
        logits = logits.view(-1, head_count, n_gate)
        biases = torch.empty((head_count, n_gate), device=logits.device, dtype=logits.dtype)
        realized = torch.empty_like(biases)
        for head in range(head_count):
            for gate_idx in range(n_gate):
                z = logits[:, head, gate_idx]
                target = desired[head].to(z)
                lo, hi = -40.0, 40.0
                for _ in range(64):
                    mid = (lo + hi) / 2
                    if torch.sigmoid(z + mid).mean() < target:
                        lo = mid
                    else:
                        hi = mid
                bias = (lo + hi) / 2
                biases[head, gate_idx] = bias
                realized[head, gate_idx] = torch.sigmoid(z + bias).mean()
        layer.alpha_proj.bias.copy_(biases.reshape(-1))
        realized_by_head = realized.mean(-1).clamp(1e-8, 1 - 1e-8)
        realized_half_life = -torch.log(torch.tensor(2.0, device=realized.device, dtype=realized.dtype)) / realized_by_head.log()
        realized_gate = torch.sigmoid(logits + biases.view(1, head_count, n_gate))
        report_layers.append({
            "layer": layer.layer_idx,
            "target_half_life": targets.cpu().tolist(),
            "realized_half_life": realized_half_life.cpu().tolist(),
            "target_mean_retention": desired.cpu().tolist(),
            "realized_mean_retention": realized_by_head.cpu().tolist(),
            "retention_residual": (realized_by_head - desired.to(realized)).cpu().tolist(),
            "pre_bias_logit_std": logits.std(dim=0, unbiased=False).view(head_count, n_gate)[:, 0].cpu().tolist(),
            "realized_gate_std": realized_gate.std(dim=0, unbiased=False).mean(-1).cpu().tolist(),
            "mean_abs_dretention_dlogit": (realized_gate * (1 - realized_gate)).mean(dim=0).mean(-1).cpu().tolist(),
        })
    finally:
        model.train(was_training)
    token_digest = sha256()
    for token_batch in token_batches:
        token_digest.update(token_batch.detach().cpu().numpy().tobytes())
    return {
        "method": "calibrated-geometric-package",
        "treatment_components": {
            "alpha_bias_fitting": True,
            "within_head_projection_row_matching": True,
            "causal_claim": "package effect; not timescale-only",
        },
        "data_provenance": {
            "split": "train",
            "selection": "first configured calibration batches",
            "weighting": "uniform over explicitly included tokens",
            "padding_mask": "explicit mask supplied" if calibration_masks is not None else "none supplied; all input positions included",
            "masking_limit": "write/query/filler categories are not inferred without phase masks",
            "token_batches_sha256": token_digest.hexdigest(),
        },
        "calibration_tokens": count,
        "min_half_life": float(min_half_life),
        "max_half_life": float(max_half_life),
        "target_half_life_by_head": targets.cpu().tolist(),
        "layers": report_layers,
        "half_life_interpretation": "decay diagnostic derived from mean alpha; not a measured full-state memory lifetime",
    }


def F_linear_no_bias(layer: nn.Linear, inputs: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(inputs, layer.weight, bias=None)
