"""Measured E0/E1 qualification helpers for the granularity/init study runner."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_e0_worker_configs(config: dict[str, Any], output_root: Path, *, primary_size: str = "10M") -> list[dict[str, Any]]:
    """Expand easy MQAR/overwrite learnability tasks for scalar and channel endpoints."""
    qualification = config["qualification"]
    retrieval = config["retrieval"]
    budget = config["budgets"][primary_size]
    tasks = []
    conditions = [c for c in config["conditions"] if c["initialization"] == "default" and c["variant"] in {"scalar", "channel-wise"}]
    families = tuple(family for family in ("mqar", "overwrite") if family in config.get("datasets", []))
    if not families:
        raise ValueError("E0 requires at least one synthetic retrieval family")
    for family in families:
        for condition in conditions:
            task = {
                "operation": "train", "experiment_id": config["experiment_id"],
                "condition_id": condition["id"], "family": family, "scale": primary_size,
                "variant": condition["variant"], "initialization": "default", "seed": 42,
                "output_dir": str((output_root / "qualification" / "E0" / family / condition["id"] / "seed_42").resolve()),
                **budget,
                "max_optimizer_steps": int(qualification["qualification_steps"]),
                "evaluation_interval": int(qualification["validation_interval"]),
                "retrieval_config": {
                    **retrieval, "train_examples": min(int(retrieval["train_examples"]), 5000),
                    "validation_examples": 512, "held_out_examples": 512,
                    "sequence_lengths": [int(qualification["easy_task_length"])],
                    "key_counts": [int(qualification["easy_task_keys"])],
                    "query_distances": list(qualification["easy_task_query_distances"]),
                    "overwrite_counts": list(qualification["easy_overwrite_counts"]),
                },
                "qualification": True, "qualification_stage": "E0",
                "qualification_scope": qualification.get("scope", "production"),
                "validation_answer_accuracy": True,
            }
            tasks.append(task)
    return tasks


def build_e1_worker_configs(
    config: dict[str, Any], output_root: Path, *, extension: bool = False, primary_size: str = "10M",
) -> list[dict[str, Any]]:
    """Expand endpoint LR pilots on validation splits, using final task difficulty."""
    qualification = config["qualification"]
    candidates = list(qualification["learning_rate_candidates"])
    if extension:
        candidates = [float(qualification["extension_learning_rate"])]
    budget = config["budgets"][primary_size]
    endpoints = [c for c in config["conditions"] if c["initialization"] == "default" and c["variant"] in {"scalar", "channel-wise"}]
    tasks = []
    for family in config["datasets"]:
        for condition in endpoints:
            for rate in candidates:
                rate_tag = f"{rate:.0e}".replace("+", "")
                retrieval_config = None
                if family != "tinystories":
                    retrieval = config["retrieval"]
                    retrieval_config = {
                        **retrieval, "validation_examples": int(retrieval.get("validation_examples", 2000)),
                    }
                tasks.append({
                    "operation": "train", "experiment_id": config["experiment_id"],
                    "condition_id": condition["id"], "family": family, "scale": primary_size,
                    "variant": condition["variant"], "initialization": "default", "seed": 42,
                    "output_dir": str((output_root / "qualification" / "E1" / family / condition["id"] / rate_tag).resolve()),
                    **budget,
                    "max_optimizer_steps": int(qualification["pilot_steps"]),
                    "evaluation_interval": int(qualification["validation_interval"]),
                    "learning_rate": float(rate), "qualification": True,
                    "qualification_stage": "E1", "validation_answer_accuracy": family != "tinystories",
                    "qualification_scope": qualification.get("scope", "production"),
                    **({"retrieval_config": retrieval_config} if retrieval_config is not None else {}),
                })
    return tasks


def build_e1_timing_worker_configs(
    config: dict[str, Any], output_root: Path, learning_rate: float,
) -> list[dict[str, Any]]:
    """Measure the selected recipe across families, model sizes, and study conditions."""
    qualification = config["qualification"]
    rate_tag = f"{float(learning_rate):.0e}".replace("+", "")
    tasks = []
    sizes = config["qualification"].get("runtime_sizes", ["10M", "30M"])
    for size in sizes:
        budget = config["budgets"][size]
        for family in config["datasets"]:
            for condition in config["conditions"]:
                if condition.get("sizes") and size not in condition["sizes"]:
                    continue
                retrieval_config = None
                if family != "tinystories":
                    retrieval_config = dict(config["retrieval"])
                task = {
                    "operation": "train", "experiment_id": config["experiment_id"],
                    "condition_id": condition["id"], "family": family, "scale": size,
                    "variant": condition["variant"], "initialization": condition["initialization"], "seed": 42,
                    "output_dir": str((output_root / "qualification" / "E1" / "timing" / f"lr_{rate_tag}" / family / size / condition["id"]).resolve()),
                    **budget,
                    "max_optimizer_steps": int(qualification["pilot_steps"]),
                    "evaluation_interval": int(qualification["validation_interval"]),
                    "learning_rate": float(learning_rate), "qualification": True,
                    "qualification_stage": "E1-timing",
                    "validation_answer_accuracy": family != "tinystories",
                    "qualification_scope": qualification.get("scope", "production"),
                }
                if retrieval_config is not None:
                    task["retrieval_config"] = retrieval_config
                tasks.append(task)
    return tasks
def build_e1_curriculum_worker_configs(
    config: dict[str, Any], output_root: Path, *, extension: bool = False, primary_size: str = "10M",
) -> list[dict[str, Any]]:
    """Build curriculum pilots from an explicit schedule; never invent stages."""
    qualification = config["qualification"]
    schedule = qualification.get("curriculum")
    if not schedule:
        return []
    if not isinstance(schedule, list):
        raise ValueError("qualification.curriculum must be a list of explicit stage objects")
    normalized = []
    previous_step = 0
    for stage in schedule:
        if not isinstance(stage, dict) or "start_step" not in stage or "retrieval_config" not in stage:
            raise ValueError("Each curriculum stage requires start_step and retrieval_config")
        start_step = int(stage["start_step"])
        if start_step <= previous_step:
            raise ValueError("Curriculum start_step values must be positive and strictly increasing")
        previous_step = start_step
        normalized.append({"start_step": start_step, "retrieval_config": dict(stage["retrieval_config"])})
    if normalized[0]["start_step"] != 1:
        raise ValueError("Curriculum must define its initial stage at start_step=1")
    if normalized[-1]["start_step"] > int(qualification["pilot_steps"]):
        raise ValueError("Final curriculum stage must begin within the qualification pilot budget")
    final_config = normalized[-1]["retrieval_config"]
    for key in ("generator_version", "sequence_lengths", "key_counts", "query_distances", "overwrite_counts"):
        if key in config["retrieval"] and final_config.get(key) != config["retrieval"][key]:
            raise ValueError(f"Final curriculum stage must match main-study retrieval.{key}")
    candidates = [float(qualification["extension_learning_rate"])] if extension else [
        float(rate) for rate in qualification["learning_rate_candidates"]
    ]
    endpoints = [condition for condition in config["conditions"]
                 if condition["initialization"] == "default" and condition["variant"] in {"scalar", "channel-wise"}]
    budget = config["budgets"][primary_size]
    tasks = []
    for family in ("mqar", "overwrite"):
        for condition in endpoints:
            for rate in candidates:
                rate_tag = f"{rate:.0e}".replace("+", "")
                task = {
                    "operation": "train", "experiment_id": config["experiment_id"],
                    "condition_id": condition["id"], "family": family, "scale": primary_size,
                    "variant": condition["variant"], "initialization": "default", "seed": 42,
                    "output_dir": str((output_root / "qualification" / "E1" / "curriculum" / family / primary_size /
                                       condition["id"] / f"lr_{rate_tag}").resolve()),
                    **budget,
                    "max_optimizer_steps": int(qualification["pilot_steps"]),
                    "evaluation_interval": int(qualification["validation_interval"]),
                    "learning_rate": rate, "qualification": True, "qualification_stage": "E1-curriculum",
                    "qualification_scope": qualification.get("scope", "production"),
                    "validation_answer_accuracy": True, "curriculum": normalized,
                }
                tasks.append(task)
    return tasks


def read_validation_history(run_dir: Path) -> list[dict[str, Any]]:
    """Read persisted validation measurements, preferring the immediate history file."""
    history_path = run_dir / "validation_history.json"
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        history = metrics.get("validation_history", [])
    if not isinstance(history, list) or not history:
        raise ValueError(f"No validation history in {run_dir}")
    return history


def assess_e0(
    results: Iterable[dict[str, Any]], threshold: float = 0.9, consecutive: int = 2,
    *, expected_families: Iterable[str] | None = None,
    expected_variants: Iterable[str] = ("scalar", "channel-wise"),
    require_complete: bool = False,
) -> dict[str, Any]:
    """Require each endpoint/task run to clear the answer threshold consecutively.

    The historical helper remains permissive for fixture use. Production E0
    passes the expected family/endpoint set and ``require_complete=True`` so a
    partial qualification cannot authorize the study.
    """
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("E0 answer threshold must be between zero and one")
    if int(consecutive) < 1:
        raise ValueError("E0 consecutive validation count must be positive")
    result_rows = []
    passed = True
    seen: set[tuple[str, str]] = set()
    for result in results:
        identity = {key: result.get(key) for key in ("family", "variant", "condition_id", "seed", "qualification_scope")}
        cell = (str(identity["family"]), str(identity["variant"]))
        if cell in seen:
            raise ValueError(f"Duplicate E0 qualification cell: {cell[0]}/{cell[1]}")
        seen.add(cell)
        history = result.get("validation_history")
        if history is None and result.get("run_dir"):
            history = read_validation_history(Path(result["run_dir"]))
        if not isinstance(history, list):
            raise ValueError(f"E0 result has no validation history: {identity}")
        scores = [entry.get("answer_accuracy") for entry in history]
        if any(score is None for score in scores):
            raise ValueError(f"E0 validation history lacks answer_accuracy: {identity}")
        found = any(all(float(score) >= threshold for score in scores[index:index + consecutive])
                    for index in range(max(0, len(scores) - consecutive + 1)))
        steps = [entry.get("step") for entry in history]
        if all(step is not None for step in steps):
            if any(int(current) <= int(previous) for previous, current in zip(steps, steps[1:])):
                raise ValueError(f"E0 validation steps are not strictly increasing: {identity}")
        result_rows.append({**identity, "answer_accuracy": scores, "validation_steps": steps, "passed": found})
        passed &= found
    expected_cells = {
        (str(family), str(variant))
        for family in (expected_families or ())
        for variant in expected_variants
    }
    missing = sorted(expected_cells - seen)
    unexpected = sorted(seen - expected_cells) if expected_cells else []
    if require_complete and (missing or unexpected):
        passed = False
    return {"status": "passed" if passed and result_rows and not (require_complete and (missing or unexpected)) else "failed", "threshold": threshold,
            "consecutive_validations": consecutive, "runs": result_rows, "completed_at": _now(),
            "scope": "smoke" if any(row.get("qualification_scope") == "smoke" for row in result_rows) else "production",
            "criterion": "answer_accuracy >= threshold for consecutive validations",
            "expected_cells": sorted({f"{family}/{variant}" for family, variant in expected_cells}),
            "missing_cells": [f"{family}/{variant}" for family, variant in missing],
            "unexpected_cells": [f"{family}/{variant}" for family, variant in unexpected]}


def select_e1_recipe(
    results: Iterable[dict[str, Any]],
    expected_families: Iterable[str] = ("tinystories", "mqar", "overwrite"),
) -> dict[str, Any]:
    """Select one LR from within-family validation ranks across both endpoints."""
    rows = list(results)
    if not rows:
        raise ValueError("No E1 pilot results were provided")
    grouped: dict[float, dict[str, list[tuple[str, float]]]] = {}
    details = []
    endpoints_by_family: dict[str, set[str]] = {}
    failed_rows = []
    for row in rows:
        if row.get("status") in {"failed", "error", "incomplete"}:
            failed_rows.append({key: row.get(key) for key in ("family", "variant", "learning_rate", "status", "error")})
            continue
        metrics = row.get("metrics")
        if metrics is None and row.get("run_dir"):
            metrics = json.loads((Path(row["run_dir"]) / "metrics.json").read_text(encoding="utf-8"))
        if not isinstance(metrics, dict):
            raise ValueError("E1 result is missing metrics")
        history = metrics.get("validation_history") or row.get("validation_history")
        if not history and row.get("run_dir"):
            history = read_validation_history(Path(row["run_dir"]))
        if not history:
            raise ValueError("E1 result has no validation history")
        final = history[-1]
        score = final.get("answer_accuracy") if row.get("family") in {"mqar", "overwrite"} else None
        metric_name = "answer_accuracy"
        if score is None:
            score = -float(final["loss"])
            metric_name = "negative_validation_loss"
        rate_value = row.get("learning_rate", row.get("rate", metrics.get("learning_rate")))
        if rate_value is None:
            raise ValueError("E1 result is missing its learning rate")
        rate = float(rate_value)
        family = str(row.get("family"))
        variant = str(row.get("variant"))
        grouped.setdefault(rate, {}).setdefault(family, []).append((variant, float(score)))
        endpoints_by_family.setdefault(family, set()).add(variant)
        details.append({"family": row.get("family"), "variant": row.get("variant"), "learning_rate": rate,
                        "metric": metric_name, "validation_score": float(score), "step": final.get("step")})
    rates = sorted(grouped)
    families = sorted(set(expected_families))
    expected_endpoints = {"scalar", "channel-wise"}
    complete_rates = []
    excluded_rates = []
    for rate in rates:
        missing = [
            f"{family}/{variant}" for family in families for variant in expected_endpoints
            if variant not in {variant for variant, _ in grouped[rate].get(family, [])}
        ]
        if missing:
            excluded_rates.append({"learning_rate": rate, "reason": "incomplete family/endpoint coverage", "missing_cells": missing})
        else:
            complete_rates.append(rate)
    if not complete_rates:
        raise ValueError("No E1 learning rate has complete validation coverage across expected families and endpoints")
    rates = complete_rates
    per_family_scores: dict[str, dict[float, float]] = {}
    per_family_best_ranks: dict[str, dict[float, int]] = {}
    selected_metrics_by_family: dict[str, dict[str, Any]] = {}
    for family in families:
        # Metrics remain in family-specific units; cross-family aggregation uses ranks only.
        means = {rate: sum(score for _, score in grouped[rate][family]) / len(grouped[rate][family]) for rate in rates}
        per_family_scores[family] = means
        ordered = sorted(rates, key=lambda rate: (-means[rate], rate))
        ranks: dict[float, float] = {}
        index = 0
        while index < len(ordered):
            tied = [ordered[index]]
            while index + len(tied) < len(ordered) and abs(means[ordered[index + len(tied)]] - means[tied[0]]) <= 1e-12:
                tied.append(ordered[index + len(tied)])
            average_rank = (index + index + len(tied) - 1) / 2
            for rate in tied:
                ranks[rate] = average_rank
            index += len(tied)
        per_family_best_ranks[family] = ranks
    rank_means = {rate: sum(per_family_best_ranks[family][rate] for family in families) / len(families) for rate in rates}
    selected = sorted(rates, key=lambda rate: (rank_means[rate], rate))[0]
    for family in families:
        selected_metrics_by_family[family] = {
            variant: next(detail for detail in reversed(details)
                          if detail["family"] == family and detail["variant"] == variant
                          and detail["learning_rate"] == selected)
            for variant in expected_endpoints
        }
    return {"status": "selected", "selected_learning_rate": selected,
            "mean_family_validation_rank_by_learning_rate": {str(rate): score for rate, score in rank_means.items()},
            "family_validation_scores_by_learning_rate": {
                family: {str(rate): score for rate, score in scores.items()} for family, scores in per_family_scores.items()
            },
            "pilot_count_by_learning_rate": {str(rate): sum(len(grouped[rate].get(family, [])) for family in families) for rate in rates},
            "selected_metrics_by_family_endpoint": selected_metrics_by_family,
            "excluded_incomplete_learning_rates": excluded_rates, "failed_candidates": failed_rows,
            "selection_metric": "mean within-family endpoint rank; higher answer accuracy or lower validation loss wins within each family",
            "results": details, "completed_at": _now()}


def training_time_estimates(
    results: Iterable[dict[str, Any]], budgets: dict[str, Any], *,
    seed_counts: dict[str, int] | None = None, calibration_timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Estimate per-family/condition cost only from measured same-size pilot timing."""
    pilots = []
    for row in results:
        metrics = row.get("metrics")
        if metrics is None and row.get("run_dir"):
            metrics = json.loads((Path(row["run_dir"]) / "metrics.json").read_text(encoding="utf-8"))
        training = metrics.get("training", {}) if isinstance(metrics, dict) else {}
        steps = int(training.get("fixed_optimizer_steps", row.get("completed_steps", 0)))
        elapsed = float(training.get("wall_time_seconds", row.get("wall_time_seconds", 0.0)))
        if steps <= 0 or elapsed <= 0:
            raise ValueError("Pilot timing requires positive completed steps and wall time")
        seconds_per_step = elapsed / steps
        size = str(row.get("scale", row.get("size", "")))
        family = str(row.get("family", ""))
        condition = str(row.get("condition_id", row.get("variant", "")))
        if size not in budgets:
            raise ValueError(f"Runtime pilot has no measured budget size: {size}")
        pilots.append({"scale": size, "family": family, "condition_id": condition,
                       "variant": row.get("variant"), "initialization": row.get("initialization"),
                       "steps": steps, "wall_time_seconds": elapsed, "seconds_per_step": seconds_per_step,
                       "timing_scope": "training-phase wall time includes optimizer work, periodic in-loop validation, and best-validation checkpoint writes"})
    if not pilots:
        raise ValueError("No measured E1 runtime pilots were provided")
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for pilot in pilots:
        key = (pilot["scale"], pilot["family"], pilot["condition_id"])
        grouped.setdefault(key, []).append(pilot["seconds_per_step"])
    estimates = {}
    totals = {size: 0.0 for size in budgets}
    for (size, family, condition), speeds in sorted(grouped.items()):
        measured = sorted(speeds)[len(speeds) // 2]
        steps = int(budgets[size]["max_optimizer_steps"])
        hours = steps * measured / 3600
        estimates.setdefault(size, {}).setdefault(family, {})[condition] = {
            "measured_seconds_per_step": measured, "optimizer_steps": steps, "training_hours_per_run": hours,
        }
        totals[size] += hours * (seed_counts or {}).get(size, 1)
    per_seed_training = {
        size: sum(condition["training_hours_per_run"] for family_groups in families.values() for condition in family_groups.values())
        for size, families in estimates.items()
    }
    calibration_hours = {size: 0.0 for size in budgets}
    for key, seconds in (calibration_timings or {}).items():
        size = str(key).split("/", 1)[0]
        if size in calibration_hours:
            calibration_hours[size] += float(seconds) / 3600
    actual_seed_counts = seed_counts or {size: 1 for size in budgets}
    scenario_totals = {}
    for size in budgets:
        scenario_totals[size] = {}
        for seed_count in (1, 3, 5):
            scenario_totals[size][str(seed_count)] = {
                "training_plus_periodic_validation_hours": per_seed_training.get(size, 0.0) * seed_count,
                "one_time_calibration_hours": calibration_hours[size] * seed_count,
                "combined_measured_hours": (per_seed_training.get(size, 0.0) + calibration_hours[size]) * seed_count,
            }
    return {
        "status": "measured", "pilot_runs": pilots, "budget_estimates_by_size_family_condition": estimates,
        "training_plus_periodic_validation_hours_per_seed_by_size": per_seed_training,
        "training_plus_periodic_validation_total_hours_by_size_actual_seed_count": totals,
        "calibration_hours_per_seed_by_size": calibration_hours,
        "scenario_totals_by_size_and_seed_count": scenario_totals,
        "seed_counts": actual_seed_counts,
        "calibration_seconds_by_condition": calibration_timings or {},
        "validation_checkpoint_and_evaluation_overhead": "Periodic in-loop validation and best-validation checkpoint writes are included in worker training wall time; final validation, final checkpoint writing, and evaluation are not measured by this estimate.",
        "unmeasured_conditions": [
            {"scale": size, "family": family, "condition_id": condition}
            for size in budgets
            for family in {p["family"] for p in pilots if p["scale"] == size}
            for condition in {p["condition_id"] for p in pilots if p["scale"] == size}
            if (size, family, condition) not in grouped
        ],
    }


def render_training_time_estimate_markdown(timing: dict[str, Any]) -> str:
    """Render the measured per-cell costs and 1/3/5-seed totals for review."""
    calibration_lines = [
        f"- {key}: {value:.3f}" for key, value in sorted(timing.get("calibration_seconds_by_condition", {}).items())
    ] or ["- No calibration timing was recorded."]
    lines = [
        "# Training time estimate", "",
        "Estimates use measured per-condition optimizer-step timing. Training-phase wall time includes optimizer work, periodic validation, and best-validation checkpoint writes; it excludes final validation, final checkpoint writing, and evaluation. Calibration timing is listed separately.", "",
        "## Per-run estimates", "",
        "| Size | Family | Condition | Steps | Hours per run | Measured seconds per step |",
        "|---|---|---|---:|---:|---:|",
    ]
    estimates = timing.get("budget_estimates_by_size_family_condition", {})
    for size, families in sorted(estimates.items()):
        for family, conditions in sorted(families.items()):
            for condition, values in sorted(conditions.items()):
                lines.append(
                    f"| {size} | {family} | {condition} | {values['optimizer_steps']} | "
                    f"{values['training_hours_per_run']:.3f} | {values['measured_seconds_per_step']:.4g} |"
                )
    lines.extend(["", "## Seed scenarios", "",
                  "| Size | Seeds | Training plus periodic validation (hours) | Calibration (hours) | Combined measured hours |",
                  "|---|---:|---:|---:|---:|"])
    for size, scenarios in sorted(timing.get("scenario_totals_by_size_and_seed_count", {}).items()):
        for count, values in sorted(scenarios.items(), key=lambda item: int(item[0])):
            lines.append(
                f"| {size} | {count} | {values['training_plus_periodic_validation_hours']:.2f} | "
                f"{values['one_time_calibration_hours']:.2f} | {values['combined_measured_hours']:.2f} |"
            )
    lines.extend(["", "Calibration by condition (seconds):", "", *calibration_lines, ""])
    return "\n".join(lines)


def corrected_pilot_budget(
    timing: dict[str, Any], *, calibration_hours: float = 0.0,
    validation_hours: float = 0.0, evaluation_hours: float = 0.0,
    isolated_benchmark_hours: float = 0.0, failure_allowance: float = 0.15,
) -> dict[str, Any]:
    """Combine measured pilot components into an authorization record.

    This is a planning artifact only. It refuses missing or negative inputs and
    never authorizes execution; a human must approve the resulting scope in the
    study config.
    """
    if timing.get("status") != "measured":
        raise ValueError("Corrected pilot budget requires measured timing")
    if not 0.0 <= float(failure_allowance) < 1.0:
        raise ValueError("failure allowance must be in [0, 1)")
    components = {
        "training": sum(float(value) for value in timing.get("training_plus_periodic_validation_total_hours_by_size_actual_seed_count", {}).values()),
        "calibration": float(calibration_hours),
        "validation": float(validation_hours),
        "held_out_evaluation": float(evaluation_hours),
        "isolated_inference_benchmark": float(isolated_benchmark_hours),
    }
    if any(value < 0.0 for value in components.values()):
        raise ValueError("pilot budget components must be non-negative")
    measured_total = sum(components.values())
    allowance_hours = measured_total * float(failure_allowance)
    return {
        "status": "measured",
        "authorization_required": True,
        "components_hours": components,
        "measured_total_hours": measured_total,
        "failure_allowance_fraction": float(failure_allowance),
        "failure_allowance_hours": allowance_hours,
        "authorized_total_hours": measured_total + allowance_hours,
        "scope": "one fresh corrected paired seed; no replication or larger scale",
        "decision": "pending_researcher_authorization",
    }


def validate_e1_learnability(recipe: dict[str, Any], minimum_answer_accuracy: float) -> dict[str, Any]:
    """Apply an explicit engineering floor to selected validation retrieval accuracy."""
    selected = recipe.get("selected_metrics_by_family_endpoint", {})
    cells = []
    passed = True
    for family in ("mqar", "overwrite"):
        for endpoint in ("scalar", "channel-wise"):
            row = selected.get(family, {}).get(endpoint)
            accuracy = None if row is None or row.get("metric") != "answer_accuracy" else float(row["validation_score"])
            cell_passes = accuracy is not None and accuracy >= minimum_answer_accuracy
            passed &= cell_passes
            cells.append({"family": family, "endpoint": endpoint, "answer_accuracy": accuracy, "passed": cell_passes})
    return {"status": "passed" if passed else "failed", "minimum_validation_answer_accuracy": float(minimum_answer_accuracy),
            "cells": cells, "criterion_type": "engineering learnability floor; not a literature claim"}


def write_protocol_lock(
    output_root: Path, config_path: Path, *, e0_result: dict[str, Any],
    recipe: dict[str, Any], timing: dict[str, Any], calibration: dict[str, Any],
) -> Path:
    """Write the lock only after learnability, recipe selection, and timing are measured."""
    if e0_result.get("status") != "passed":
        raise ValueError("Cannot lock protocol before E0 learnability passes")
    scope = str(e0_result.get("scope", "production"))
    if scope == "smoke" and float(e0_result.get("threshold", 1.0)) != 0.0:
        raise ValueError("Smoke qualification must use an explicit zero learnability threshold")
    if recipe.get("status") != "selected" or timing.get("status") != "measured":
        raise ValueError("Cannot lock protocol before E1 recipe and runtime measurements pass")
    if recipe.get("learnability_status") != "passed":
        raise ValueError("Cannot lock protocol before the E1 learnability floor passes")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_root = Path(__file__).resolve().parent
    source_hashes = {}
    for source_name in ("recurrence.py", "model.py", "controls.py", "retrieval.py", "study_qualification.py", "study_runner.py"):
        source_path = source_root / source_name
        if source_path.is_file():
            source_hashes[source_name] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    payload = {
        "status": "locked", "scope": scope, "experiment_id": config["experiment_id"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "locked_at": _now(), "e0_result": e0_result, "recipe_selection": recipe,
        "training_time_estimate": timing, "calibration": calibration,
        "protocol_contract": {
            "version": "granularity-study-protocol-v2",
            "qualification": {
                "e0_threshold": e0_result.get("threshold"),
                "e0_consecutive_validations": e0_result.get("consecutive_validations"),
                "e0_expected_cells": e0_result.get("expected_cells", []),
                "e1_selection": recipe.get("selection_metric"),
                "final_difficulty_floor": recipe.get("minimum_final_answer_accuracy"),
            },
            "training_budget": config.get("budgets", {}),
            "data_and_generator": {
                "datasets": config.get("datasets", []),
                "retrieval": config.get("retrieval", {}),
                "splits": {"training": "train", "recipe_selection": "validation", "held_out": "held_out"},
            },
            "evaluation": config.get("evaluation", {}),
            "stopping_and_exclusions": {
                "failed_candidates": "retained and excluded from recipe selection",
                "incomplete_cells": "cannot authorize a protocol lock",
                "capacity": "reported only after complete corrected pilot",
            },
        },
        "configuration_snapshot": config,
        "implementation_sources": source_hashes,
    }
    path = output_root / "protocol_lock.json"
    _write_json(path, payload)
    return path
