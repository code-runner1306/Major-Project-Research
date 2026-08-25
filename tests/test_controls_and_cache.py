from __future__ import annotations

import pytest
import torch

from config import ModelConfig, PRODUCTION_MODELS
from dataset import toy_dataloader
from controls import construct_model, documented_hybrid_schedules, expected_parameter_count, matched_scalar_config
from eval import cache_equivalence_error
from model import cache_num_bytes


def test_documented_hybrid_counts_and_schedules():
    expected = {
        "10M": (10_030_880, 10_155_680),
        "30M": (29_272_320, 30_071_040),
        "55M": (53_348_288, 55_122_368),
        "100M": (96_918_912, 100_817_280),
    }
    for scale, backbone in PRODUCTION_MODELS.items():
        scalar_count = expected_parameter_count(backbone, "scalar", 50_257)
        channel_count = expected_parameter_count(backbone, "channel-wise", 50_257)
        assert (scalar_count, channel_count) == expected[scale]
        assert matched_scalar_config(backbone).ffn_widths == documented_hybrid_schedules()[scale]
        assert expected_parameter_count(backbone, "scalar-matched", 50_257) == channel_count


def test_pure_gdn_smoke_control_and_cache_equivalence(smoke_config):
    backbone = smoke_config.model_for("tiny")
    assert matched_scalar_config(backbone).ffn_widths == (74, 73, 73)
    tokens = torch.randint(0, smoke_config.vocab_size, (2, 9))
    for variant in ("scalar", "channel-wise", "scalar-matched"):
        model, _, info = construct_model(backbone, variant, smoke_config.vocab_size)
        assert cache_equivalence_error(model, tokens) < 1e-5
        if variant == "scalar-matched":
            assert info["parameter_equality_asserted"]


def test_invalid_control_fails_explicitly():
    impossible = ModelConfig("impossible", d_model=10, n_layer=1, n_head=3, d_ff=20, attention_layout="pure_gdn")
    with pytest.raises(ValueError, match="Exact useful scalar matching"):
        matched_scalar_config(impossible)


def test_mixed_precision_cache_and_kv_growth(smoke_config):
    hybrid = ModelConfig("hybrid", d_model=32, n_layer=4, n_head=4, d_ff=64, attention_layout="hybrid")
    model, _, _ = construct_model(hybrid, "channel-wise", smoke_config.vocab_size)
    tokens = torch.randint(0, smoke_config.vocab_size, (1, 8))
    # bfloat16 autocast is the CPU-safe mixed-precision path. Full and cached
    # decode must agree to a looser dtype-appropriate tolerance.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert cache_equivalence_error(model, tokens) < 2e-2
    _, small_cache = model.prefill(tokens[:, :2])
    _, large_cache = model.prefill(tokens)
    assert cache_num_bytes(large_cache) > cache_num_bytes(small_cache)


def test_paired_toy_order_and_fixed_budget(smoke_config):
    first = list(toy_dataloader(smoke_config, seed=107, split="train"))
    second = list(toy_dataloader(smoke_config, seed=107, split="train"))
    assert smoke_config.budget.max_optimizer_steps == 2
    assert all(torch.equal(x1, x2) and torch.equal(y1, y2) for (x1, y1), (x2, y2) in zip(first, second))
