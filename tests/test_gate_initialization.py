from __future__ import annotations

import torch

from config import ModelConfig
from controls import construct_model, expected_parameter_count
from gate_initialization import calibrate_gate_initialization
from model import GatedDeltaNetLM, GatedDeltaNetAttention


def test_grouped_gates_expand_and_backpropagate_independent_outputs():
    attention = GatedDeltaNetAttention(32, 4, 8, variant="grouped-4")
    x = torch.randn(2, 5, 32, requires_grad=True)
    output = attention(x)
    output.square().mean().backward()
    assert output.shape == x.shape
    assert attention.alpha_proj.out_features == 16
    assert attention.alpha_proj.weight.grad is not None
    assert torch.count_nonzero(attention.alpha_proj.weight.grad).item() > 0


def test_calibration_matches_geometric_head_retention_and_within_head_functions():
    torch.manual_seed(17)
    config = ModelConfig("tiny", d_model=32, n_layer=2, n_head=4, d_ff=64, attention_layout="pure_gdn")
    model = GatedDeltaNetLM(
        config, variant="grouped-4", vocab_size=97, gate_initialization="calibrated-geometric",
    )
    # Before calibration, corresponding group functions are identical within each head.
    for block in model.blocks:
        layer = block.attn
        assert isinstance(layer, GatedDeltaNetAttention)
        w = layer.alpha_proj.weight.view(4, 4, 32)
        assert torch.equal(w, w[:, :1].expand_as(w))

    report = calibrate_gate_initialization(
        model, [torch.randint(0, 97, (2, 8)), torch.randint(0, 97, (1, 5))], "cpu", 2.0, 32.0,
    )
    assert report["calibration_tokens"] == 21
    assert len(report["target_half_life_by_head"]) == 4
    for layer_report in report["layers"]:
        assert max(abs(x) for x in layer_report["retention_residual"]) < 1e-5
    for block in model.blocks:
        layer = block.attn
        w = layer.alpha_proj.weight.view(4, 4, 32)
        b = layer.alpha_proj.bias.view(4, 4)
        assert torch.equal(w, w[:, :1].expand_as(w))
        assert torch.allclose(b, b[:, :1].expand_as(b))
    tokens = torch.randint(0, 97, (2, 8))
    logits, _ = model(tokens)
    logits.sum().backward()
    assert model.blocks[0].attn.alpha_proj.weight.grad is not None


def test_study_seed_matches_backbone_across_gate_granularities():
    config = ModelConfig("tiny", d_model=32, n_layer=1, n_head=4, d_ff=64, attention_layout="pure_gdn")
    grouped = GatedDeltaNetLM(config, "grouped-4", 97, study_init_seed=23)
    channel = GatedDeltaNetLM(config, "channel-wise", 97, study_init_seed=23)
    grouped_params = dict(grouped.named_parameters())
    channel_params = dict(channel.named_parameters())
    for name in grouped_params.keys() & channel_params.keys():
        if name.startswith("blocks.") and ".attn.alpha_proj." in name:
            continue
        assert torch.equal(grouped_params[name], channel_params[name]), name


def test_study_gate_arms_have_matched_trainable_biases_and_legacy_stays_bias_free():
    config = ModelConfig("tiny", d_model=32, n_layer=2, n_head=4, d_ff=64, attention_layout="pure_gdn")
    for variant in ("scalar", "grouped-4", "channel-wise"):
        default, _, default_meta = construct_model(config, variant, 97, study_init_seed=41)
        calibrated, _, calibrated_meta = construct_model(
            config, variant, 97, gate_initialization="calibrated-geometric", study_init_seed=41,
        )
        default_count = sum(parameter.numel() for parameter in default.parameters() if parameter.requires_grad)
        calibrated_count = sum(parameter.numel() for parameter in calibrated.parameters() if parameter.requires_grad)
        assert default_count == calibrated_count
        assert default_count == expected_parameter_count(config, variant, 97, study_init_seed=41)
        assert default_meta["parameter_count"] == calibrated_meta["parameter_count"] == default_count
        for model in (default, calibrated):
            for block in model.blocks:
                assert block.attn.alpha_proj.bias is not None
                assert torch.count_nonzero(block.attn.alpha_proj.bias).item() == 0
        # The seed changes initialization, not non-gate trainable storage.
        default_params = dict(default.named_parameters())
        calibrated_params = dict(calibrated.named_parameters())
        for name in default_params.keys() & calibrated_params.keys():
            if "alpha_proj" not in name:
                assert torch.equal(default_params[name], calibrated_params[name]), name

    legacy = GatedDeltaNetLM(config, "scalar", 97)
    assert all(block.attn.alpha_proj.bias is None for block in legacy.blocks)
    assert all(not name.endswith("alpha_proj.bias") for name in legacy.state_dict())
