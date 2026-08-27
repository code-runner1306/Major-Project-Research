"""Construction and accounting for scientifically distinct model variants."""

from __future__ import annotations

from dataclasses import replace

from config import ModelConfig
from model import GatedDeltaNetLM, parameter_breakdown
from recurrence import RecurrenceOptions, backend_metadata


def gdn_layer_count(model_config: ModelConfig) -> int:
    if model_config.attention_layout == "pure_gdn":
        return model_config.n_layer
    if model_config.attention_layout == "hybrid":
        return sum(index % 4 != 3 for index in range(model_config.n_layer))
    raise ValueError(f"Unknown attention layout {model_config.attention_layout!r}")


def channel_wise_gate_overhead(model_config: ModelConfig) -> int:
    """Extra trainable alpha-projection weights relative to scalar gating."""

    return gdn_layer_count(model_config) * model_config.d_model * (model_config.d_model - model_config.n_head)


def expected_parameter_count(model_config: ModelConfig, variant: str, vocab_size: int) -> int:
    """Exact trainable-storage count without instantiating a production model."""

    if variant == "scalar-matched":
        model_config, variant = matched_scalar_config(model_config), "scalar"
    if variant not in {"scalar", "channel-wise"}:
        raise ValueError(f"Unknown variant {variant!r}")
    d_model, heads = model_config.d_model, model_config.n_head
    gdn_layers = gdn_layer_count(model_config)
    attention_layers = model_config.n_layer - gdn_layers
    embeddings = vocab_size * d_model  # Output head is tied, not additional storage.
    norms = (2 * model_config.n_layer + 1) * d_model
    ffn = 3 * d_model * sum(model_config.ffn_widths)
    # Q/K/V, output gate and output projection; three depthwise conv weights
    # plus bias; beta and alpha projections.
    gdn = gdn_layers * (5 * d_model * d_model + 15 * d_model + 2 * d_model * heads)
    if variant == "channel-wise":
        gdn += channel_wise_gate_overhead(model_config)
    full_attention = attention_layers * 4 * d_model * d_model
    return embeddings + norms + ffn + gdn + full_attention


def matched_scalar_config(model_config: ModelConfig) -> ModelConfig:
    """Widen scalar FFNs by useful parameters until the count matches exactly.

    A SwiGLU width increment adds ``3 * d_model`` parameters. The channel-wise
    alpha overhead must be divisible by that amount; otherwise this backbone
    cannot be exactly capacity matched without dummy parameters and is rejected.
    """

    gate_overhead = channel_wise_gate_overhead(model_config)
    if gate_overhead <= 0:
        raise ValueError("A matched scalar control requires at least one Gated DeltaNet layer")
    parameters_per_width = 3 * model_config.d_model
    if gate_overhead % parameters_per_width:
        raise ValueError(
            "Exact useful scalar matching is impossible for this configuration: "
            f"gate overhead {gate_overhead} is not divisible by {parameters_per_width} "
            "parameters per distributed SwiGLU width increment."
        )
    increments = gate_overhead // parameters_per_width
    per_layer, remainder = divmod(increments, model_config.n_layer)
    baseline = model_config.ffn_widths
    # Assign the remainder to early layers deterministically; layer ordering is
    # persisted in every manifest rather than hidden behind a rounded average.
    schedule = tuple(width + per_layer + (1 if index < remainder else 0) for index, width in enumerate(baseline))
    return replace(model_config, d_ff=schedule)


def construct_model(
    model_config: ModelConfig,
    variant: str,
    vocab_size: int,
    *,
    recurrence_options: RecurrenceOptions = RecurrenceOptions(),
) -> tuple[GatedDeltaNetLM, ModelConfig, dict]:
    """Build a variant and attach the reproducible control-accounting metadata."""

    resolved_config = matched_scalar_config(model_config) if variant == "scalar-matched" else model_config
    actual_variant = "scalar" if variant == "scalar-matched" else variant
    model = GatedDeltaNetLM(
        resolved_config, variant=actual_variant, vocab_size=vocab_size,
        recurrence_options=recurrence_options,
    )
    total = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    expected = expected_parameter_count(model_config, variant, vocab_size)
    if total != expected:
        raise AssertionError(f"Model accounting mismatch: expected {expected}, constructed {total}")
    metadata = {
        "variant": variant,
        "control_strategy": "expanded_scalar_ffn_exact_match" if variant == "scalar-matched" else "same_backbone",
        "parameter_count": total,
        "gate_parameter_overhead": channel_wise_gate_overhead(model_config),
        "per_layer_ffn_widths": list(resolved_config.ffn_widths),
        "parameter_components": parameter_breakdown(model),
        "recurrence": backend_metadata(recurrence_options),
    }
    if variant == "scalar-matched":
        channel_model = GatedDeltaNetLM(model_config, variant="channel-wise", vocab_size=vocab_size)
        target = sum(parameter.numel() for parameter in channel_model.parameters() if parameter.requires_grad)
        if total != target:
            raise AssertionError(f"scalar-matched count {total} does not equal channel-wise count {target}")
        metadata["matched_target_parameter_count"] = target
        metadata["parameter_equality_asserted"] = True
    return model, resolved_config, metadata


def documented_hybrid_schedules() -> dict[str, tuple[int, ...]]:
    """The exact capacity-control schedules quoted in the approved design."""

    return {
        "10M": (470, 470, 469, 469, 469, 469),
        "30M": (937, 937, 936, 936, 936, 936, 936, 936, 936, 936),
        "55M": (1304,) * 12,
        "100M": (1677,) * 16,
    }
