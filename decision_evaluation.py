"""Prediction, calibration, and reporting helpers for typed decisions."""

from __future__ import annotations

from collections import defaultdict
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from decision_metrics import fit_temperature, paired_case_bootstrap
from decision_model import GDNDecisionModel


def _to_list(value: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


@torch.inference_mode()
def predict_gdn(
    model: GDNDecisionModel,
    examples: Sequence[Mapping[str, Any]],
    *,
    device: torch.device | None = None,
    partition: str,
) -> list[dict[str, Any]]:
    """Return canonical-order logits and probabilities for unpadded examples."""
    if partition not in {"train", "validation", "calibration", "test"}:
        raise ValueError(f"Unsupported prediction partition: {partition}")
    device = device or next(model.parameters()).device
    model.eval()
    records: list[dict[str, Any]] = []
    for example in examples:
        input_ids = torch.as_tensor(example["input_ids"], dtype=torch.long, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        logits = model.score(input_ids, int(example["decision_index"]), example["option_spans"]).float()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latency_seconds = time.perf_counter() - started
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Model produced non-finite decision logits")
        probabilities = torch.softmax(logits, dim=-1)
        record = {
            key: example[key]
            for key in (
                "workflow", "case_id", "question_id", "primitive", "canonical_labels",
                "target", "truncated", "original_length", "retained_length", "score_values",
            )
            if key in example
        }
        record.update({
            "partition": partition,
            "variant": model.variant,
            "raw_logits": _to_list(logits),
            "raw_probabilities": _to_list(probabilities),
            "target": _to_list(torch.as_tensor(example["target"])),
            "option_order": list(example.get("option_order", example.get("canonical_labels", []))),
            "latency_seconds": latency_seconds,
            "serialized_length": int(input_ids.numel()),
            "canonical_reference_serialized_length": int(input_ids.numel()),
            "option_count": len(example["option_spans"]),
        })
        records.append(record)
    return records


def fit_primitive_temperatures(
    records: Sequence[Mapping[str, Any]],
    *,
    max_iter: int = 100,
) -> dict[str, dict[str, float | int | bool]]:
    """Fit one positive temperature per primitive over ragged candidate sets."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - project dependency
        raise RuntimeError("Temperature calibration requires PyTorch") from exc
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["primitive"])].append(record)
    result: dict[str, dict[str, float | int | bool]] = {}
    for primitive in ("choice", "score", "noul"):
        rows = grouped.get(primitive, [])
        if not rows:
            raise ValueError(f"Calibration partition has no {primitive!r} decisions")
        logits = [torch.tensor(row["raw_logits"], dtype=torch.float64) for row in rows]
        targets = [torch.tensor(row["target"], dtype=torch.float64) for row in rows]
        if any(a.ndim != 1 or a.shape != b.shape for a, b in zip(logits, targets)):
            raise ValueError(f"Invalid ragged logits/targets for {primitive}")
        if any(not torch.isfinite(a).all() for a in logits + targets):
            raise ValueError(f"Non-finite calibration values for {primitive}")
        log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)

        def objective() -> torch.Tensor:
            temperature = log_t.exp().clamp(1e-8, 1e8)
            terms = [-(target * torch.log_softmax(score / temperature, dim=-1)).sum()
                     for score, target in zip(logits, targets)]
            return torch.stack(terms).mean()

        initial_nll = float(objective().detach())
        optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = objective()
            loss.backward()
            return loss

        optimizer.step(closure)
        temperature = float(log_t.detach().exp().clamp(1e-8, 1e8))
        final_nll = float(objective().detach())
        state = optimizer.state.get(log_t, {})
        result[primitive] = {
            "temperature": temperature,
            "initial_soft_nll": initial_nll,
            "final_soft_nll": final_nll,
            "iterations": int(state.get("n_iter", 0)),
            "examples": len(rows),
        }
    return result


def apply_temperatures(
    records: Sequence[Mapping[str, Any]],
    temperatures: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy records and append probabilities calibrated by primitive."""
    output = []
    for record in records:
        primitive = str(record["primitive"])
        if primitive not in temperatures:
            raise KeyError(f"Missing calibration temperature for {primitive}")
        temperature = float(temperatures[primitive]["temperature"])
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"Invalid calibration temperature for {primitive}")
        logits = np.asarray(record["raw_logits"], dtype=np.float64) / temperature
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        copied = dict(record)
        copied["temperature"] = temperature
        copied["calibrated_probabilities"] = probs.tolist()
        output.append(copied)
    return output


def _metrics_for(records: Sequence[Mapping[str, Any]], probability_key: str) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot score an empty prediction set")
    rows = []
    confidences, correctness = [], []
    score_errors = []
    for row in records:
        prediction = np.asarray(row[probability_key], dtype=np.float64)
        target = np.asarray(row["target"], dtype=np.float64)
        if prediction.ndim != 1 or prediction.shape != target.shape:
            raise ValueError("Prediction and target option shapes differ")
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError("Non-finite prediction or target")
        if (prediction < 0).any() or (target < 0).any() or not np.isclose(prediction.sum(), 1.0, atol=1e-5):
            raise ValueError("Prediction or target is not a valid distribution")
        target_sum = target.sum()
        if not np.isclose(target_sum, 1.0, atol=1e-5):
            raise ValueError("Target is not normalized")
        chosen = int(np.argmax(prediction))
        rows.append({
            "accuracy": float(chosen == int(np.argmax(target))),
            "soft_accuracy": float(np.dot(prediction, target)),
            "soft_nll": float(-np.dot(target, np.log(np.clip(prediction, 1e-12, 1.0)))),
            "brier": float(np.sum((prediction - target) ** 2)),
        })
        confidences.append(float(prediction[chosen]))
        correctness.append(float(target[chosen]))
        if row.get("primitive") == "score" and row.get("score_values") is not None:
            score_values = np.asarray(row["score_values"], dtype=np.float64)
            if score_values.size != prediction.size:
                raise ValueError("score_values length differs from candidates")
            score_errors.append(abs(float(prediction @ score_values - target @ score_values)))
    confidence = np.asarray(confidences)
    correct_mass = np.asarray(correctness)
    bins = np.minimum(np.floor(confidence * 15).astype(int), 14)
    ece_value = 0.0
    for index in range(15):
        mask = bins == index
        if mask.any():
            ece_value += float(mask.mean() * abs(confidence[mask].mean() - correct_mass[mask].mean()))
    matrix = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    matrix["ece"] = ece_value
    matrix["score_expected_value_mae"] = float(np.mean(score_errors)) if score_errors else None
    return matrix


def summarize_predictions(
    records: Sequence[Mapping[str, Any]],
    *,
    probability_key: str = "raw_probabilities",
) -> dict[str, Any]:
    """Compute aggregate and workflow/primitive/option/truncation metrics."""
    result: dict[str, Any] = {"overall": _metrics_for(records, probability_key)}
    for field in ("workflow", "primitive", "truncated"):
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in records:
            groups[str(row.get(field, "unknown"))].append(row)
        result[field] = {key: _metrics_for(rows, probability_key) for key, rows in sorted(groups.items())}
    option_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        option_groups[str(len(row["canonical_labels"]))].append(row)
    result["option_count"] = {
        key: _metrics_for(rows, probability_key) for key, rows in sorted(option_groups.items(), key=lambda x: int(x[0]))
    }
    result["examples"] = len(records)
    result["metric_schema_version"] = 1
    result["ece_convention"] = "15 equal-width confidence bins; soft correctness mass at predicted class"
    return result


def summarize_latency(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize synchronized per-decision latency by primitive and input shape."""
    valid = [row for row in records
             if isinstance(row.get("latency_seconds"), (int, float))
             and math.isfinite(float(row["latency_seconds"])) and float(row["latency_seconds"]) >= 0]

    def stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        values = sorted(float(row["latency_seconds"]) for row in rows)
        if not values:
            return {"decisions": 0, "mean_seconds": None, "median_seconds": None,
                    "p95_seconds": None, "decisions_per_second": None}
        return {
            "decisions": len(values),
            "mean_seconds": float(np.mean(values)),
            "median_seconds": float(np.median(values)),
            "p95_seconds": float(np.percentile(values, 95, method="linear")),
            "decisions_per_second": len(values) / sum(values) if sum(values) > 0 else None,
        }

    def length_bucket(length: int) -> str:
        if length <= 128:
            return "001-128"
        if length <= 256:
            return "129-256"
        if length <= 512:
            return "257-512"
        return "513+"

    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        "primitive": defaultdict(list),
        "serialized_length_bucket": defaultdict(list),
        "option_count": defaultdict(list),
        "primitive_x_length_bucket": defaultdict(list),
    }
    for row in valid:
        primitive = str(row.get("primitive", "unknown"))
        length = int(row.get("canonical_reference_serialized_length", row.get("serialized_length", 0)))
        bucket = length_bucket(length)
        options = int(row.get("option_count", len(row.get("canonical_labels", []))))
        grouped["primitive"][primitive].append(row)
        grouped["serialized_length_bucket"][bucket].append(row)
        grouped["option_count"][str(options)].append(row)
        grouped["primitive_x_length_bucket"][f"{primitive}/{bucket}"].append(row)
    return {
        "protocol": "one synchronized forward per decision; device synchronization around each model call",
        "input_length_unit": "canonical GPT-2 serialized tokens; exact GDN input length and shared-content size proxy for Laya",
        "length_buckets": ["001-128", "129-256", "257-512", "513+"],
        "overall": stats(valid),
        **{dimension: {key: stats(rows) for key, rows in sorted(values.items())}
           for dimension, values in grouped.items()},
        "missing_latency_records": len(records) - len(valid),
    }


def paired_metric_bootstrap(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    probability_key: str = "calibrated_probabilities",
    replicates: int = 10_000,
    seed: int = 2028,
) -> dict[str, Any]:
    """Pair bootstrap over cases; both prediction inputs must have same rows."""
    key = lambda row: (row.get("workflow"), row.get("case_id"), row.get("question_id"))
    left_by_id = {key(row): row for row in left}
    right_by_id = {key(row): row for row in right}
    if left_by_id.keys() != right_by_id.keys():
        raise ValueError("Prediction artifacts do not have matching case/question coverage")
    ordered_keys = sorted(left_by_id, key=lambda value: tuple(map(str, value)))
    case_ids = [f"{workflow}/{case_id}" for workflow, case_id, _ in ordered_keys]
    result: dict[str, Any] = {}
    metric_fields = ("accuracy", "soft_accuracy", "soft_nll", "brier")
    for metric in metric_fields:
        left_values, right_values = [], []
        for row_key in ordered_keys:
            for source, target_list in ((left_by_id[row_key], left_values), (right_by_id[row_key], right_values)):
                values = _metrics_for([source], probability_key)
                target_list.append(values[metric])
        interval = paired_case_bootstrap(
            case_ids, np.asarray(left_values)[:, None], np.asarray(right_values)[:, None],
            n_resamples=replicates, seed=seed,
        )
        result[metric] = {
            "left": interval.left.__dict__,
            "right": interval.right.__dict__,
            "left_minus_right": interval.difference.__dict__,
        }
    return result


def require_test_stage_ready(
    *,
    manifest: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> None:
    """Guard official-test inference behind locked config and calibration."""
    if manifest.get("training_config_locked") is not True:
        raise RuntimeError("Official test is locked until training configuration is frozen")
    if lifecycle.get("status") not in {"calibrated", "evaluating", "complete"}:
        raise RuntimeError("Official test requires a completed calibration stage")
    if not calibration.get("temperatures"):
        raise RuntimeError("Official test requires a validated calibration artifact")


__all__ = [
    "apply_temperatures", "fit_primitive_temperatures", "paired_metric_bootstrap",
    "predict_gdn", "require_test_stage_ready", "summarize_predictions",
]
