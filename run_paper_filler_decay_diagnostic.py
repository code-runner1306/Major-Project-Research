"""Log GDN decay gates on the post-training MQAR filler-extension probe.

This is a frozen-checkpoint diagnostic, not a second training experiment.  It
uses the same 64-token MQAR examples and insertion rule as
``run_paper_mqar_length_extension.py`` and records the decay-only alpha gates
on the inserted token-0 filler span.  The recurrence can still write to state,
so alpha products are reported as a mechanism diagnostic rather than as a
retention guarantee.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from artifacts import atomic_write_json
from confirm_recipe_study import same_manifest, select_runs
from recipe_data import RecipeData, audit_mqar, manifest, mqar
from recipe_model import RecipeLM, RecipeGDN


ROOT = Path("output/calibrated_recipe_study")
OUT = Path("research_results/filler_decay_diagnostic_20261119")
FIGURE = Path("indicon_gate_paper/figures/filler_decay.pdf")
LENGTHS = (256, 1024)
SEEDS = (43, 44)
INITS = ("default", "calibrated")
VARIANTS = ("scalar", "channel-wise")
EXAMPLES = 16
OFFSET = 1950000
PREFIX = 8
BATCH = 8


def extended_cases(tokens: torch.Tensor, targets: torch.Tensor) -> dict[int, RecipeData]:
    """Make the exact evaluation-only length extrapolation cases."""
    cases: dict[int, RecipeData] = {}
    for length in LENGTHS:
        extra = length - 64
        extended = torch.cat(
            (tokens[:, :PREFIX], tokens.new_zeros((len(tokens), extra)), tokens[:, PREFIX:]),
            dim=1,
        )
        labels = torch.cat(
            (targets[:, :PREFIX], targets.new_full((len(targets), extra), -100), targets[:, PREFIX:]),
            dim=1,
        )
        audit = audit_mqar(extended, labels)
        if not audit["passed"]:
            raise ValueError(f"MQAR audit failed at length {length}: {audit}")
        cases[length] = RecipeData(
            extended,
            labels,
            torch.where(labels != -100, 0, -1),
            torch.full_like(labels, -1),
            torch.zeros_like(extended, dtype=torch.bool),
            manifest(
                extended,
                labels,
                seed=OFFSET,
                injected_filler=extra,
                filler_token=0,
                filler_start=PREFIX,
                source_length=64,
            ),
        )
    return cases


def checkpoint_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_with_predictions(model: RecipeLM, data: RecipeData, device: torch.device) -> dict:
    """Match recipe_evaluation.evaluate and retain a compact prediction audit."""
    was_training = model.training
    model.eval()
    correct = total = complete = 0
    confidences: list[float] = []
    wrong_confidences: list[float] = []
    prediction_bytes: list[bytes] = []
    target_bytes: list[bytes] = []
    try:
        for start in range(0, len(data.tokens), BATCH):
            stop = min(start + BATCH, len(data.tokens))
            tokens = data.tokens[start:stop].clone()
            if data.erase_answers[start:stop].any():
                tokens[data.erase_answers[start:stop]] = 1
            labels = data.labels[start:stop]
            mask = labels != -100
            selected = model.scored_logits(tokens.to(device), mask.to(device))
            targets = labels[mask].to(device)
            predictions = selected.argmax(-1)
            matches = predictions == targets
            correct += int(matches.sum())
            total += int(matches.numel())
            counts = mask.sum(1)
            offset = 0
            for count in counts.tolist():
                complete += int(matches[offset : offset + count].all())
                offset += count
            confidence = selected.float().log_softmax(-1).exp().amax(-1)
            confidences.extend(confidence.detach().cpu().tolist())
            wrong_confidences.extend(confidence[~matches].detach().cpu().tolist())
            prediction_bytes.append(predictions.detach().cpu().numpy().astype("<i8").tobytes())
            target_bytes.append(targets.detach().cpu().numpy().astype("<i8").tobytes())
    finally:
        model.train(was_training)
    pred_digest = hashlib.sha256(b"".join(prediction_bytes)).hexdigest()
    target_digest = hashlib.sha256(b"".join(target_bytes)).hexdigest()
    return {
        "exact_query_accuracy": correct / total,
        "correct": correct,
        "total": total,
        "example_accuracy": complete / len(data.tokens),
        "evaluated_examples": len(data.tokens),
        "mean_max_probability": float(np.mean(confidences)),
        "mean_wrong_max_probability": float(np.mean(wrong_confidences)) if wrong_confidences else None,
        "prediction_sha256": pred_digest,
        "target_sha256": target_digest,
        "history_conditioning": "answer_free",
        "data_sha256": data.identity["sha256"],
    }


def summarize_alpha(chunks: list[torch.Tensor], filler_length: int) -> dict:
    """Summarize alpha and its decay-only product over B x filler x channels."""
    alpha = torch.cat(chunks, dim=0).float()  # [examples, filler, heads, gate]
    values = alpha.reshape(-1).numpy()
    # Sum log(alpha) is numerically stable and is exactly log of the product.
    log_alpha = alpha.clamp_min(torch.finfo(torch.float32).tiny).log()
    per_channel_log_product = log_alpha.sum(dim=1).reshape(-1).numpy()
    mean_log_alpha = float(log_alpha.mean())
    half_life = float(math.log(0.5) / mean_log_alpha) if mean_log_alpha < 0 else float("inf")
    product = np.exp(np.clip(per_channel_log_product, -745.0, 0.0))
    return {
        "examples": int(alpha.shape[0]),
        "filler_tokens": filler_length,
        "heads": int(alpha.shape[2]),
        "gate_channels": int(alpha.shape[3]),
        "alpha_mean": float(values.mean()),
        "alpha_p10": float(np.percentile(values, 10)),
        "alpha_p50": float(np.percentile(values, 50)),
        "alpha_p90": float(np.percentile(values, 90)),
        "log_alpha_mean": mean_log_alpha,
        "effective_half_life_tokens": half_life,
        "log_product_mean": float(per_channel_log_product.mean()),
        "log_product_p10": float(np.percentile(per_channel_log_product, 10)),
        "log_product_p50": float(np.percentile(per_channel_log_product, 50)),
        "log_product_p90": float(np.percentile(per_channel_log_product, 90)),
        "product_mean": float(product.mean()),
        "product_p10": float(np.percentile(product, 10)),
        "product_p50": float(np.percentile(product, 50)),
        "product_p90": float(np.percentile(product, 90)),
    }


def run(output: Path = OUT, figure: Path = FIGURE) -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    selected = select_runs(ROOT, ["mqar"], list(SEEDS), [], "explore")
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "description": "Decay-only alpha probe on frozen checkpoints from the 64-token MQAR training study.",
        "checkpoint_root": str(ROOT),
        "seeds": list(SEEDS),
        "initializations": list(INITS),
        "variants": list(VARIANTS),
        "lengths": list(LENGTHS),
        "examples_per_seed": EXAMPLES,
        "batch": BATCH,
        "generator_seed_offset": OFFSET,
        "source_training_length": 64,
        "prefix_tokens_before_filler": PREFIX,
        "filler_token": 0,
        "filler_token_is_present_in_training_queries": True,
        "device": str(device),
        "torch": str(torch.__version__),
        "alpha_definition": "exp(-exp(A_log) * softplus(a_proj(x) + dt_bias))",
        "summary": "per-layer alpha values and sum(log(alpha)) product over inserted filler only",
        "interpretation_guardrail": "alpha products omit beta writes and therefore do not alone imply state retention",
    }
    atomic_write_json(output / "protocol.json", protocol)
    rows: list[dict] = []
    for seed in SEEDS:
        tokens, targets = mqar(EXAMPLES, length=64, seed=OFFSET + seed)
        cases = extended_cases(tokens, targets)
        for (_, checkpoint_seed, init, variant), (summary, record) in sorted(selected.items(), key=lambda item: item[0]):
            if checkpoint_seed != seed:
                continue
            path = summary.parent / "latest_checkpoint.pt"
            with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                checkpoint = torch.load(path, map_location=device, weights_only=True)
            if not same_manifest(checkpoint["settings"], record["settings"]):
                raise ValueError(f"Checkpoint metadata mismatch: {path}")
            model = RecipeLM(variant, 8192, 64, 128, "chunked", seed)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.to(device)
            model.eval()
            digest = checkpoint_digest(path)
            for length, data in cases.items():
                extra = length - 64
                captured: list[list[torch.Tensor]] = [[] for _ in range(len(model.network.backbone.layers))]
                hooks = []
                for layer_index, layer in enumerate(model.network.backbone.layers):
                    mixer = layer.sequence_mixer
                    if not isinstance(mixer, RecipeGDN):
                        raise TypeError(f"Unexpected mixer at layer {layer_index}: {type(mixer)}")

                    def capture(module, args, index=layer_index):
                        alpha = module.decay(args[0])[:, PREFIX : PREFIX + extra].detach().cpu()
                        captured[index].append(alpha)

                    hooks.append(mixer.register_forward_pre_hook(capture))
                try:
                    metrics = evaluate_with_predictions(model, data, device)
                finally:
                    for hook in hooks:
                        hook.remove()
                layers = [summarize_alpha(chunks, extra) for chunks in captured]
                row = {
                    "seed": seed,
                    "initialization": init,
                    "variant": variant,
                    "length": length,
                    "filler_start": PREFIX,
                    "filler_stop": PREFIX + extra,
                    "checkpoint": str(path),
                    "checkpoint_sha256": digest,
                    "metrics": metrics,
                    "layers": layers,
                }
                rows.append(row)
                print(
                    seed,
                    init,
                    variant,
                    length,
                    metrics["exact_query_accuracy"],
                    [round(x["alpha_p50"], 6) for x in layers],
                    [round(x["log_product_p50"], 2) for x in layers],
                    flush=True,
                )
                atomic_write_json(output / "results.json", {"status": "running", "protocol": protocol, "results": rows})
    atomic_write_json(output / "results.json", {"status": "complete", "protocol": protocol, "results": rows})
    plot_results(rows, figure)


def plot_results(rows: list[dict], figure: Path = FIGURE) -> None:
    """Draw all four seed and initialization pairs at length 1,024."""
    selected = {(r["seed"], r["initialization"], r["variant"]): r
                for r in rows if r["length"] == 1024}
    floor = -8
    colors = {"scalar": "#276FBF", "channel-wise": "#B63B52"}
    with plt.rc_context({"font.size": 7}):
        fig, axes = plt.subplots(2, 2, figsize=(3.48, 2.56), sharex=True, sharey=True)
        for i, init in enumerate(("default", "calibrated")):
            for j, seed in enumerate((43, 44)):
                ax = axes[i, j]
                xs = np.arange(1, 5)
                counts = []
                for variant in ("scalar", "channel-wise"):
                    row = selected[seed, init, variant]
                    counts.append(str(row["metrics"]["correct"]))
                    quantiles = np.asarray([[item[f"log_product_p{k}"] for item in row["layers"]]
                                            for k in (10, 50, 90)]) / math.log(10)
                    lo, median, hi = np.maximum(quantiles, floor)
                    color = colors[variant]
                    ax.fill_between(xs, lo, hi, color=color, alpha=0.16, linewidth=0)
                    ax.plot(xs, median, marker="o", color=color, ms=2.6, lw=0.9, label=variant)
                name = "Mean decay" if init == "calibrated" else "Default"
                ax.set_title(f"{name}, seed {seed} ({counts[0]}/{counts[1]})", fontsize=7)
                ax.set_xticks(xs)
                ax.set_yticks((-8, -4, 0))
                ax.set_ylim(-8.35, 0.5)
                ax.grid(axis="y", alpha=0.28, linewidth=0.45)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.supxlabel("GDN layer", y=0.025, fontsize=7)
        fig.supylabel(r"$\log_{10}$ decay-only product", x=0.025, fontsize=7)
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.15, top=0.84, hspace=0.44, wspace=0.20)
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure, format="pdf")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT, help="New directory for protocol and results.")
    parser.add_argument("--figure", type=Path, default=FIGURE, help="Output PDF path.")
    args = parser.parse_args()
    run(args.output, args.figure)
