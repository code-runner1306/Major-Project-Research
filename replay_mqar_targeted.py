"""Independent replay and provenance audit for the two reviewer-targeted MQAR cells.

This script deliberately reuses the pinned generator and the frozen ``explore``
checkpoint-selection contract, but does not reuse the summary metrics from the
length-extension run.  It writes per-answer predictions (including confidence,
label and prediction identities) so a zero-accuracy cell can be audited for
data or numerical failures.

Run from the repository root with::

    uv run python replay_mqar_targeted.py

Outputs are kept under ``research_results/mqar_length_extension_20261119``.
"""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

import numpy as np
import torch

from artifacts import atomic_write_json
from confirm_recipe_study import same_manifest, select_runs
from recipe_data import RecipeData, audit_mqar, manifest, mqar
from recipe_model import RecipeLM


ROOT = Path("output/calibrated_recipe_study")
OUT = Path("research_results/mqar_length_extension_20261119")
REPLAY_OUT = OUT / "targeted_replays"
SEEDS = (42, 43, 44)
TARGET_SEED = 44
EXAMPLES = 256
OFFSET = 1_950_000
TRAIN_LENGTH = 64
PREFIX = 8
VOCABULARY = 8192
BATCH = 8

# The first is the reported 301/1024 channel-wise stress cell.  In the paper's
# nomenclature, calibrated is the mean-decay initialization; retaining the
# explicit label here avoids silently changing the checkpoint condition.
TARGETS = (
    {"seed": TARGET_SEED, "initialization": "default", "variant": "channel-wise", "length": 256},
    {"seed": TARGET_SEED, "initialization": "calibrated", "variant": "scalar", "length": 1024},
)


def _all_targets():
    """Return all 12 MQAR conditions x three frozen evaluation lengths."""
    return tuple(
        {"seed": seed, "initialization": initialization, "variant": variant, "length": length}
        for seed in SEEDS
        for initialization in ("default", "calibrated")
        for variant in ("scalar", "channel-wise")
        for length in (64, 256, 1024)
    )


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _build_extended(seed: int, length: int):
    tokens, targets = mqar(EXAMPLES, vocabulary=VOCABULARY, length=TRAIN_LENGTH,
                           pairs=4, seed=OFFSET + seed)
    extra = length - TRAIN_LENGTH
    extended = torch.cat((tokens[:, :PREFIX], tokens.new_zeros((EXAMPLES, extra)),
                          tokens[:, PREFIX:]), dim=1)
    labels = torch.cat((targets[:, :PREFIX], targets.new_full((EXAMPLES, extra), -100),
                        targets[:, PREFIX:]), dim=1)
    label_audit = audit_mqar(extended, labels)
    if not label_audit["passed"]:
        raise ValueError(f"MQAR label audit failed: {label_audit}")
    data = RecipeData(
        extended,
        labels,
        torch.where(labels != -100, 0, -1),
        torch.full_like(labels, -1),
        torch.zeros_like(extended, dtype=torch.bool),
        manifest(extended, labels, seed=OFFSET + seed, injected_filler=extra),
    )
    # The training generator uses the same token 0 as the filler/query gap;
    # this records exposure without claiming that a 960-token run was trained.
    return data, label_audit, {
        "training_length": TRAIN_LENGTH,
        "evaluation_length": length,
        "inserted_filler_token": 0,
        "inserted_filler_count": extra,
        "inserted_after_prefix_tokens": PREFIX,
        "filler_token_seen_in_training_generator": True,
    }


def _checkpoint_for(selected, seed, initialization, variant):
    summary_path, record = selected[("mqar", seed, initialization, variant)]
    checkpoint_path = summary_path.parent / "latest_checkpoint.pt"
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not same_manifest(checkpoint["settings"], record["settings"]):
        raise ValueError(f"Checkpoint metadata mismatch: {checkpoint_path}")
    return checkpoint_path, digest, checkpoint


def _prediction_category(pred: int, target: int, keys: set[int], values: set[int]):
    if pred == target:
        return "correct_target"
    if pred == 0:
        return "filler_token_0"
    if pred == 1:
        return "answer_erase_token_1"
    if pred in keys:
        return "memory_key"
    if pred in values:
        return "memory_value"
    return "other_vocabulary_token"


def _token_band(pred: int):
    """MQAR vocabulary bands: 0=filler, 1..4095=keys, 4096..8191=values."""
    if pred == 0:
        return "filler_id_0"
    if 1 <= pred < VOCABULARY // 2:
        return "key_band_1_to_4095"
    if VOCABULARY // 2 <= pred < VOCABULARY:
        return "value_band_4096_to_8191"
    return "out_of_vocabulary"


@torch.no_grad()
def _collect_predictions(model, data: RecipeData):
    model.eval()
    score_rows = []
    logits_rows = []
    finite_bad = 0
    for start in range(0, EXAMPLES, BATCH):
        stop = min(start + BATCH, EXAMPLES)
        tokens = data.tokens[start:stop].clone()
        # MQAR has no answer inputs to erase, but retain the evaluation
        # contract used by recipe_evaluation for future auditability.
        tokens[data.erase_answers[start:stop]] = 1
        mask = data.labels[start:stop] != -100
        logits = model.scored_logits(tokens, mask)
        finite_bad += int((~torch.isfinite(logits)).sum().item())
        top_values, top_indices = torch.topk(logits, k=2, dim=-1)
        log_normalizer = torch.logsumexp(logits, dim=-1)
        top_probs = torch.exp(top_values[:, 0] - log_normalizer)
        positions = mask.nonzero(as_tuple=False)
        # Positions are local to this batch; retain global example indices.
        positions[:, 0] += start
        for index in range(logits.size(0)):
            score_rows.append((int(positions[index, 0]), int(positions[index, 1])))
        logits_rows.append((top_values.cpu(), top_indices.cpu(), top_probs.cpu(),
                            data.labels[start:stop][mask].cpu()))

    example_index = np.asarray([row[0] for row in score_rows], dtype=np.int32)
    sequence_position = np.asarray([row[1] for row in score_rows], dtype=np.int32)
    top_values = torch.cat([row[0] for row in logits_rows]).numpy().astype(np.float32)
    top_indices = torch.cat([row[1] for row in logits_rows]).numpy().astype(np.int64)
    top_probs = torch.cat([row[2] for row in logits_rows]).numpy().astype(np.float32)
    targets = torch.cat([row[3] for row in logits_rows]).numpy().astype(np.int64)
    predictions = top_indices[:, 0]
    categories = []
    token_bands = []
    for row, target, pred in zip(example_index, targets, predictions):
        memory = data.tokens[int(row), :PREFIX].tolist()
        keys = set(memory[0::2])
        values = set(memory[1::2])
        categories.append(_prediction_category(int(pred), int(target), keys, values))
        token_bands.append(_token_band(int(pred)))
    categories = np.asarray(categories, dtype="U32")
    token_bands = np.asarray(token_bands, dtype="U32")

    # Repeat the exact forward pass once.  An equality failure is useful evidence
    # of nondeterminism even though the model is expected to be deterministic on CPU.
    repeat_predictions = []
    for start in range(0, EXAMPLES, BATCH):
        stop = min(start + BATCH, EXAMPLES)
        tokens = data.tokens[start:stop].clone()
        tokens[data.erase_answers[start:stop]] = 1
        mask = data.labels[start:stop] != -100
        repeat_predictions.append(model.scored_logits(tokens, mask).argmax(-1).cpu())
    repeat_predictions = torch.cat(repeat_predictions).numpy().astype(np.int64)
    return {
        "example_index": example_index,
        "sequence_position": sequence_position,
        "targets": targets,
        "predictions": predictions,
        "top1_logit": top_values[:, 0],
        "top2_logit": top_values[:, 1],
        "top1_margin": top_values[:, 0] - top_values[:, 1],
        "top1_token_probability": top_probs,
        "top2_predictions": top_indices[:, 1],
        "prediction_category": categories,
        "prediction_token_band": token_bands,
        "repeat_predictions": repeat_predictions,
        "nonfinite_logit_count": np.asarray([finite_bad], dtype=np.int64),
    }


def _summary(arrays, data, target, checkpoint_path, checkpoint_sha, label_audit,
             metadata, table_row, existing_row):
    targets = arrays["targets"]
    predictions = arrays["predictions"]
    correct = predictions == targets
    categories = arrays["prediction_category"]
    token_bands = arrays["prediction_token_band"]
    wrong = ~correct
    confidence = arrays["top1_token_probability"]
    margins = arrays["top1_margin"]
    wrong_indices = np.flatnonzero(wrong)
    wrong_examples = []
    for i in wrong_indices[:32]:
        wrong_examples.append({
            "example_index": int(arrays["example_index"][i]),
            "sequence_position": int(arrays["sequence_position"][i]),
            "target": int(targets[i]),
            "prediction": int(predictions[i]),
            "category": str(categories[i]),
            "token_band": str(token_bands[i]),
            "top1_token_probability": float(confidence[i]),
            "top1_margin": float(margins[i]),
        })
    return {
        "seed": target["seed"],
        "initialization": target["initialization"],
        "variant": target["variant"],
        "length": target["length"],
        "training_length": TRAIN_LENGTH,
        "examples": EXAMPLES,
        "queries_per_example": 4,
        "answers": int(len(targets)),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "all_logits_finite": int(arrays["nonfinite_logit_count"][0]) == 0,
        "nonfinite_logit_count": int(arrays["nonfinite_logit_count"][0]),
        "predictions_repeat_exactly": bool(np.array_equal(predictions, arrays["repeat_predictions"])),
        "label_audit": label_audit,
        "data_manifest": data.identity,
        "filler_protocol": metadata,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "table_i_checkpoint_sha256": table_row.get("checkpoint_sha256"),
        "table_i_checkpoint_hash_match": checkpoint_sha == table_row.get("checkpoint_sha256"),
        "table_i_checkpoint_path_match": str(checkpoint_path).replace("/", "\\") == table_row.get("checkpoint"),
        "prior_length_extension_row": existing_row,
        "prediction_category_counts": {
            str(category): int((categories == category).sum())
            for category in np.unique(categories)
        },
        "prediction_token_band_counts": {
            str(band): int((token_bands == band).sum())
            for band in np.unique(token_bands)
        },
        "wrong_prediction_token_band_counts": {
            str(band): int((token_bands[wrong] == band).sum())
            for band in np.unique(token_bands[wrong])
        },
        "wrong_confidence": {
            "count": int(wrong.sum()),
            "top1_probability_min": float(confidence[wrong].min()) if wrong.any() else None,
            "top1_probability_median": float(np.median(confidence[wrong])) if wrong.any() else None,
            "top1_probability_max": float(confidence[wrong].max()) if wrong.any() else None,
            "margin_median": float(np.median(margins[wrong])) if wrong.any() else None,
            "wrong_with_probability_at_least_0_5": int((confidence[wrong] >= 0.5).sum()),
            "wrong_with_margin_at_least_2": int((margins[wrong] >= 2).sum()),
        },
        "wrong_prediction_examples_first_32": wrong_examples,
    }


def run(full_sweep: bool = False):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    REPLAY_OUT.mkdir(parents=True, exist_ok=True)
    selected = select_runs(ROOT, ["mqar"], list(SEEDS), [], "explore")
    prior = json.loads((OUT / "results.json").read_text())
    confirmation = json.loads((Path("research_results/three_seed_confirmation_20261117") /
                               "confirmation_results.json").read_text())
    table_rows = {
        (row["family"], row["seed"], row["initialization"], row["variant"]): row
        for row in confirmation["results"]
        if row["family"] == "mqar"
    }
    # Archive provenance for all 12 MQAR checkpoints, even when this invocation
    # only emits the two expensive raw-prediction replays.  Table I's confirmation
    # report stores the hashes; this check catches accidental replacement or path
    # drift before any model evaluation starts.
    all_hash_matches = []
    for seed in SEEDS:
        for initialization in ("default", "calibrated"):
            for variant in ("scalar", "channel-wise"):
                checkpoint_path, checkpoint_sha, _ = _checkpoint_for(selected, seed, initialization, variant)
                table_row = table_rows[("mqar", seed, initialization, variant)]
                all_hash_matches.append({
                    "seed": seed,
                    "initialization": initialization,
                    "variant": variant,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha,
                    "table_i_checkpoint_sha256": table_row.get("checkpoint_sha256"),
                    "hash_match": checkpoint_sha == table_row.get("checkpoint_sha256"),
                })
    if not all(row["hash_match"] for row in all_hash_matches):
        raise ValueError("At least one MQAR checkpoint hash differs from Table I")

    replay_root = REPLAY_OUT / "full_sweep" if full_sweep else REPLAY_OUT
    replay_root.mkdir(parents=True, exist_ok=True)
    targets = _all_targets() if full_sweep else TARGETS
    rows = []
    for target in targets:
        data, label_audit, filler_metadata = _build_extended(target["seed"], target["length"])
        checkpoint_path, checkpoint_sha, checkpoint = _checkpoint_for(
            selected, target["seed"], target["initialization"], target["variant"])
        model = RecipeLM(target["variant"], VOCABULARY, 64, 128, "chunked", target["seed"])
        model.load_state_dict(checkpoint["model"], strict=True)
        arrays = _collect_predictions(model, data)
        key = ("mqar", target["seed"], target["initialization"], target["variant"])
        table_row = table_rows[key]
        existing_row = next(row for row in prior["results"] if
                            row["seed"] == target["seed"] and
                            row["initialization"] == target["initialization"] and
                            row["variant"] == target["variant"] and
                            row["length"] == target["length"])
        summary = _summary(arrays, data, target, checkpoint_path, checkpoint_sha,
                           label_audit, filler_metadata, table_row, existing_row)
        stem = f"mqar_seed{target['seed']}_{target['initialization']}_{target['variant'].replace('-', '_')}_length{target['length']}"
        np.savez_compressed(replay_root / f"{stem}_predictions.npz",
                            tokens=data.tokens.numpy(), labels=data.labels.numpy(), **arrays)
        atomic_write_json(replay_root / f"{stem}_summary.json", summary)
        rows.append(summary)
    report = {
        "status": "complete",
        "script": "replay_mqar_targeted.py",
        "torch": str(torch.__version__),
        "device": "cpu",
        "protocol": {
            "seeds": list(SEEDS), "examples": EXAMPLES,
            "training_length": TRAIN_LENGTH, "target_lengths": [256, 1024],
            "checkpoint_stage": "explore (16 epochs / 3136 optimizer steps)",
            "history_conditioning": "answer_free", "batch": BATCH,
            "coverage": "all 36 condition-length cells" if full_sweep else "2 targeted cells (36-cell sweep available with --full-sweep)",
            "raw_prediction_output": str(replay_root),
        },
        "table_i_mqar_checkpoint_hashes": all_hash_matches,
        "targets": rows,
    }
    report_name = "full_sweep_report.json" if full_sweep else "targeted_replay_report.json"
    atomic_write_json(replay_root / report_name, report)
    print(json.dumps({
        "status": report["status"],
        "cells": [{"initialization": row["initialization"], "variant": row["variant"],
                   "length": row["length"], "correct": row["correct"],
                   "answers": row["answers"], "all_logits_finite": row["all_logits_finite"],
                   "hash_match": row["table_i_checkpoint_hash_match"],
                   "repeat_exact": row["predictions_repeat_exactly"]} for row in rows],
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-sweep", action="store_true",
                        help="Emit raw predictions for all 12 MQAR checkpoints at lengths 64/256/1024 (36 cells).")
    args = parser.parse_args()
    run(full_sweep=args.full_sweep)
