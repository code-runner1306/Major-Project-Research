"""[Major-Project-Research patch] Tests for the decay-granularity switch in DeltaNetRefLayer.

1. The published arms are unchanged: parameter counts equal the `params` recorded in the authors' own result files.
2. Single knob: a channel-wise cell whose d_key decay rows are tied (identical weights and bias) produces exactly the
   output of the scalar cell holding that one row, with every other weight shared (to float rounding, <= 1e-6).
3. The parameter difference is exactly (d_key - 1) * (d_model + 1) per recurrence (two per bidirectional mixer).
"""
import json
from pathlib import Path

import torch

from dualgoose.mixers import DeltaNetRefLayer
from dualgoose.model import build_tiny_denoiser

UPSTREAM = Path(__file__).resolve().parents[1] / "upstream_results" / "mqar_probe"


def _count(model):
    return sum(p.numel() for p in model.parameters())


def _build(model_type, K, short_conv):
    seq_len = 2 * K + 8  # mqar_probe layout; parameter count does not depend on it beyond the vocab
    return build_tiny_denoiser(64, seq_len, d_model=32, n_layers=1, model_type=model_type, short_conv=short_conv,
                               merge="static", direction="bidirectional")


def test_published_arms_match_author_parameter_counts():
    for label, model_type, short_conv in [("armed_gdn", "gated_deltanet", 4), ("gated_deltanet", "gated_deltanet", 0)]:
        for K in (16, 32):
            files = sorted(UPSTREAM.glob(f"{label}_k{K}_d32l1_*seed0.json"))
            assert files, f"no upstream result for {label} K={K}"
            recorded = json.loads(files[0].read_text())
            model = build_tiny_denoiser(64, recorded["seq_len"], d_model=32, n_layers=1, model_type=model_type,
                                        short_conv=short_conv, merge="static", direction="bidirectional")
            assert _count(model) == recorded["params"], (label, K, _count(model), recorded["params"])


def test_tied_channel_decay_equals_scalar_decay():
    torch.manual_seed(0)
    for short_conv in (0, 4):
        scalar = DeltaNetRefLayer(32, gated=True, short_conv=short_conv, decay="scalar")
        channel = DeltaNetRefLayer(32, gated=True, short_conv=short_conv, decay="channel")
        d = channel.d_key
        with torch.no_grad():
            channel.proj.weight[:3 * d + 1].copy_(scalar.proj.weight[:3 * d + 1])
            channel.proj.bias[:3 * d + 1].copy_(scalar.proj.bias[:3 * d + 1])
            channel.proj.weight[3 * d + 1:].copy_(scalar.proj.weight[3 * d + 1:].expand(d, -1))
            channel.proj.bias[3 * d + 1:].copy_(scalar.proj.bias[3 * d + 1:].expand(d))
            channel.out.load_state_dict(scalar.out.state_dict())
            if short_conv:
                channel.qkv_conv.load_state_dict(scalar.qkv_conv.state_dict())
        x = torch.randn(3, 20, 32)
        # Identical up to float rounding: the two projections have different output widths, so the matrix
        # multiply is blocked differently. A real granularity difference is orders of magnitude larger.
        gap = (channel(x) - scalar(x)).abs().max().item()
        assert gap <= 1e-6, f"tied channel decay must reproduce the scalar cell (max difference {gap:.2e})"


def test_untied_channel_decay_differs_from_scalar():
    torch.manual_seed(1)
    channel = DeltaNetRefLayer(32, gated=True, decay="channel")
    x = torch.randn(2, 12, 32)
    with torch.no_grad():
        channel.proj.weight[-32:].normal_()
    tied = DeltaNetRefLayer(32, gated=True, decay="channel")
    tied.load_state_dict(channel.state_dict())
    with torch.no_grad():
        tied.proj.weight[-32:].copy_(channel.proj.weight[-32:][:1].expand(32, -1))
        tied.proj.bias[-32:].copy_(channel.proj.bias[-32:][:1].expand(32))
    assert not torch.allclose(channel(x), tied(x)), "per-channel decay must be able to differ from a shared decay"


def test_parameter_difference_is_only_the_decay_rows():
    for short_conv in (0, 4):
        channel = _build("gated_deltanet", 32, short_conv)
        scalar = _build("gated_deltanet_scalar", 32, short_conv)
        assert _count(channel) - _count(scalar) == 2 * (32 - 1) * (32 + 1)  # forward and backward streams
        names_c = {n for n, _ in channel.named_parameters()}
        names_s = {n for n, _ in scalar.named_parameters()}
        assert names_c == names_s
        for (n, pc), (_, ps) in zip(sorted(channel.named_parameters()), sorted(scalar.named_parameters())):
            if pc.shape != ps.shape:
                assert n.endswith(("proj.weight", "proj.bias")) and "merge" not in n, n


def test_default_decay_is_published_channel_wise():
    layer = DeltaNetRefLayer(32, gated=True)
    assert layer.decay == "channel" and layer.proj.out_features == 3 * 32 + 1 + 32
