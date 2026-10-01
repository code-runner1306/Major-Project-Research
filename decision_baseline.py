"""Optional Laya typed-decision baseline adapter (loaded only on demand)."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import math
import time
from typing import Any, Mapping, Sequence

from decision_evaluation import summarize_latency
from resource_telemetry import DeviceResourceCapture


MODEL_ID = "convaiinnovations/laya-typed-decisions"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "tolist"):
        return _jsonable(value.detach().cpu().tolist())
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    raise TypeError(f"Laya native result contains non-JSON value {type(value).__name__}")


def _probability_vector(result: Mapping[str, Any], qid: str, primitive: str,
                        labels: Sequence[str]) -> list[float]:
    """Map Laya's native probability output to the supplied canonical labels."""
    probabilities = result.get("probabilities")
    if not isinstance(probabilities, Mapping) or qid not in probabilities:
        raise ValueError(f"Laya result lacks probabilities for question {qid!r}: {list(result)}")
    native = probabilities[qid]
    if primitive == "noul" and isinstance(native, (int, float)):
        p_true = float(native)
        values = [p_true if label.lower() in {"true", "yes", "1"} else 1.0 - p_true for label in labels]
    elif isinstance(native, Mapping):
        values = [float(native[label]) for label in labels]
    elif isinstance(native, Sequence) and not isinstance(native, (str, bytes)):
        values = [float(value) for value in native]
        if len(values) != len(labels):
            raise ValueError(f"Laya returned {len(values)} probabilities for {len(labels)} labels")
    else:
        raise ValueError(f"Unsupported Laya probability shape for {qid!r}: {type(native).__name__}")
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(f"Laya returned invalid probabilities for {qid!r}")
    total = sum(values)
    if not math.isclose(total, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError(f"Laya probabilities for {qid!r} sum to {total}, not 1")
    return [value / total for value in values]


def _question(row: Mapping[str, Any]) -> dict[str, Any]:
    primitive = str(row["primitive"])
    options = row.get("options") or []
    labels = list(row["canonical_labels"])
    descriptions = {str(option["label"]): str(option["description"]) for option in options}
    if primitive == "score":
        criteria: Any = [descriptions.get(label, label) for label in labels]
    else:
        criteria = {label: descriptions.get(label, label) for label in labels}
    spec: dict[str, Any] = {
        "type": primitive,
        "instructions": str(row.get("instructions", "")),
        "criteria": criteria,
    }
    if primitive == "noul":
        # Laya's typed API defines noul probabilities semantically as P(true).
        if set(label.lower() for label in labels) != {"false", "true"}:
            raise ValueError("noul candidate labels must be false/true to preserve their semantics")
        spec["criteria"] = {"false": descriptions.get("false", "false"),
                            "true": descriptions.get("true", "true")}
    return spec


def _resolved_revision(model_id: str) -> str:
    try:
        from huggingface_hub import model_info
    except ImportError as exc:
        raise RuntimeError("Install the optional Laya baseline dependencies (laya and huggingface_hub)") from exc
    return str(model_info(model_id).sha)


def run_laya(rows: Sequence[Mapping[str, Any]], *, model_id: str = MODEL_ID,
             revision: str | None = None,
             expected_package_version: str | None = None,
             partition: str = "test", warmup_runs: int = 20) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run direct Laya typed inference at batch size one and return common rows."""
    try:
        import laya
    except ImportError as exc:
        raise RuntimeError("Laya is optional; install it in an isolated compatible environment") from exc
    package_version = version("laya")
    if expected_package_version and package_version != expected_package_version:
        raise RuntimeError(f"Expected laya=={expected_package_version}, found {package_version}")
    resolved = revision or _resolved_revision(model_id)
    agent = laya.load(model_id, revision=resolved)
    import torch

    device = next(agent.model.parameters()).device
    if rows:
        warmup_row = rows[0]
        warmup_qid = str(warmup_row["question_id"])
        for _ in range(warmup_runs):
            agent.system_one(str(warmup_row["state_text"]),
                             {warmup_qid: _question(warmup_row)})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings: list[float] = []
    predictions: list[dict[str, Any]] = []
    resource_capture = DeviceResourceCapture(device)
    with resource_capture:
        for row in rows:
            qid = str(row["question_id"])
            start = time.perf_counter()
            native = agent.system_one(str(row["state_text"]), {qid: _question(row)})
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
            labels = list(row["canonical_labels"])
            probs = _probability_vector(native, qid, str(row["primitive"]), labels)
            copied_fields = {
                key: row[key] for key in (
                    "workflow", "case_id", "question_id", "primitive", "canonical_labels",
                    "target", "truncated", "original_length", "retained_length", "score_values",
                ) if key in row
            }
            for key, value in list(copied_fields.items()):
                if hasattr(value, "detach"):
                    copied_fields[key] = value.detach().cpu().tolist()
                elif hasattr(value, "tolist"):
                    copied_fields[key] = value.tolist()
            predictions.append(copied_fields | {
                "partition": partition, "variant": "laya", "raw_probabilities": probs,
                "calibrated_probabilities": probs, "native_output": _jsonable(native),
                "latency_seconds": elapsed, "option_order": labels,
                "serialized_length": int(row["input_ids"].numel()) if hasattr(row.get("input_ids"), "numel")
                                     else len(row.get("input_ids", [])),
                "canonical_reference_serialized_length": int(row["input_ids"].numel()) if hasattr(row.get("input_ids"), "numel")
                                                               else len(row.get("input_ids", [])),
                "option_count": len(labels),
            })
    resource_measurement = resource_capture.result or {
        "started_utc": None, "finished_utc": None, "duration_wall_seconds": None,
        "runtime": {"nvml_available": False, "reason": "measurement did not complete"},
        "energy": {"supported": False, "joules": None, "reason": "measurement did not complete"},
        "activity_summary": {}, "other_gpu_activity": {"status": "unknown"},
        "activity_samples": [],
    }
    sorted_times = sorted(timings)
    max_input_length = max((row["serialized_length"] for row in predictions), default=0)
    resource_measurement["conditions"] = {
        "precision": str(next(agent.model.parameters()).dtype),
        "weight_dtype": str(next(agent.model.parameters()).dtype),
        "batch_size": 1,
        "max_canonical_reference_tokens": max_input_length,
        "laya_internal_token_count_available": False,
        "input_length_proxy": "GPT-2 canonical serialization; Laya internal prompt/token expansion is not exposed by the adapter API",
        "warmup_runs": warmup_runs if rows else 0,
        "timed_decisions": len(timings),
        "inference_passes_per_decision": 1,
        "input_serialized_length_min": min((row["serialized_length"] for row in predictions), default=0),
        "input_serialized_length_max": max_input_length,
        "option_count_min": min((row["option_count"] for row in predictions), default=0),
        "option_count_max": max((row["option_count"] for row in predictions), default=0),
    }
    summary = {
        "schema_version": 1, "model_id": model_id, "model_revision": resolved,
        "package_version": package_version, "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in agent.model.parameters()),
        "decisions": len(timings),
        "latency_median_seconds": sorted_times[len(sorted_times) // 2] if sorted_times else None,
        "latency_p95_seconds": sorted_times[min(len(sorted_times) - 1, math.ceil(.95 * len(sorted_times)) - 1)] if sorted_times else None,
        "warmup_calls": warmup_runs if rows else 0,
        "protocol": f"one state/question per system_one call; {warmup_runs} representative warmups; model load and warmup excluded",
        "inference_resources": resource_measurement,
        "latency_summary": summarize_latency(predictions),
        "timed_decisions": len(timings),
        "batch_size": 1,
        "weight_dtype": str(next(agent.model.parameters()).dtype),
        "max_canonical_reference_tokens": max((row["serialized_length"] for row in predictions), default=0),
    }
    return predictions, summary
