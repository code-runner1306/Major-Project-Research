"""Export decay-product histograms for the two mean-decay channel-wise checkpoints."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_paper_filler_decay_diagnostic import (
    ROOT, EXAMPLES, OFFSET, PREFIX, RecipeLM, atomic_write_json, checkpoint_digest,
    evaluate_with_predictions, extended_cases, mqar, same_manifest, select_runs,
)


def run(output: Path) -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    selected = select_runs(ROOT, ["mqar"], [43, 44], [], "explore")
    records = []
    for seed in (43, 44):
        summary, record = selected["mqar", seed, "calibrated", "channel-wise"]
        path = summary.parent / "latest_checkpoint.pt"
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            checkpoint = torch.load(path, map_location=device, weights_only=True)
        if not same_manifest(checkpoint["settings"], record["settings"]):
            raise ValueError(f"Checkpoint metadata mismatch: {path}")
        model = RecipeLM("channel-wise", 8192, 64, 128, "chunked", seed)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        data = extended_cases(*mqar(EXAMPLES, length=64, seed=OFFSET + seed))[1024]
        products = []

        @torch.no_grad()
        def capture(module, args):
            alpha = module.decay(args[0])[:, PREFIX:PREFIX + 960]
            products.append(alpha.double().clamp_min(torch.finfo(torch.float32).tiny).log().sum(dim=1).exp().cpu())

        hook = model.network.backbone.layers[-1].sequence_mixer.register_forward_pre_hook(capture)
        try:
            metrics = evaluate_with_predictions(model, data, device)
        finally:
            hook.remove()
        values = torch.cat(products).reshape(EXAMPLES, 128).detach().numpy()
        counts, edges = np.histogram(values, bins=np.linspace(0, 1, 21))
        records.append({
            "seed": seed, "checkpoint_sha256": checkpoint_digest(path), "metrics": metrics,
            "bin_edges": edges.tolist(), "counts": counts.tolist(),
            "per_example_channel_products": values.tolist(),
        })
        print(seed, metrics["correct"], counts.tolist(), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, {"length": 1024, "layer": 4, "examples_per_seed": EXAMPLES,
        "initializer": "mean decay", "filler_id": 0, "inserted_tokens": 960, "records": records})


def plot(records: list[dict], figure: Path) -> None:
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 9}):
        fig, axes = plt.subplots(1, 2, figsize=(4.7, 2.1), sharex=True, sharey=True,
                                 layout="constrained")
        for ax, row in zip(axes, records):
            edges = np.asarray(row["bin_edges"])
            counts = np.asarray(row["counts"])
            ax.bar(edges[:-1], 100 * counts / counts.sum(), width=np.diff(edges),
                   align="edge", color="#526B96", edgecolor="white", linewidth=0.45)
            ax.set_title(f"Seed {row['seed']}: {row['metrics']['correct']}/64 correct", fontsize=9)
            ax.set_xticks((0, 0.25, 0.5, 0.75, 1.0))
            ax.set_xlim(0, 1)
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("Products (%)")
        fig.supxlabel("Decay-only product across 960 filler tokens", fontsize=9)
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure, format="pdf")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Histogram data JSON.")
    parser.add_argument("--figure", type=Path, help="Optional histogram PDF.")
    parser.add_argument("--plot-only", action="store_true", help="Read precomputed histogram data.")
    args = parser.parse_args()
    if not args.plot_only:
        run(args.output)
    if args.figure:
        plot(json.loads(args.output.read_text(encoding="utf-8"))["records"], args.figure)
