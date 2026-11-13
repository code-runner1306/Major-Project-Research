"""Compact, seed-aware analysis helpers for the granularity/init study.

The module accepts task-index rows from the study runner rather than assuming a
particular directory layout. Large per-example outputs remain in their own
files; reports contain paths and seed-level summaries only.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, stdev
import time
from typing import Any, Iterable, Mapping

import torch

from artifacts import atomic_write_json, read_json


def _load_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (str, Path)):
        try:
            return read_json(value)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
    return None


def _finite(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _retrieval_payload(row: Mapping[str, Any]) -> dict[str, Any] | None:
    metrics = _load_payload(row.get("metrics")) or _load_payload(row.get("metrics_path"))
    if metrics is None:
        return None
    result_path = row.get("evaluation_path") or row.get("result_path")
    result = _load_payload(result_path)
    if result is not None:
        return result.get("retrieval", result)
    return metrics.get("retrieval")


def summarize_capacity(
    tasks: Iterable[Mapping[str, Any]], *, accuracy_threshold: float = 0.90,
    fixed_distance: int | None = None, metric: str = "exact_query_accuracy",
) -> dict[str, Any]:
    """Summarize observed max key count meeting threshold, without extrapolation.

    Input rows should identify family/scale/condition/initialization/seed and
    point to one E3 evaluation result. The result's ``breakdowns`` map is the
    ``evaluate_retrieval`` shape. If a fixed distance is requested, rows are
    filtered using realized query distance when available.
    """
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    seed_rows: list[dict[str, Any]] = []
    for row in tasks:
        retrieval = _retrieval_payload(row)
        breakdowns = retrieval.get("breakdowns", {}) if retrieval else {}
        joint = (retrieval or {}).get("key_count_by_query_distance", breakdowns.get("key_count_by_query_distance", breakdowns.get("key_count_query_distance", {})))
        key_counts = breakdowns.get("key_count", {})
        distances = breakdowns.get("query_distance", {})
        observed: list[dict[str, Any]] = []
        cells = joint if fixed_distance is not None else key_counts
        for cell, bucket in cells.items():
            key = cell
            if fixed_distance is not None:
                if not isinstance(bucket, Mapping):
                    continue
                selected = bucket.get(str(fixed_distance))
                if selected is None:
                    continue
                bucket = selected
                key = cell
            accuracy = _finite(bucket.get("accuracy")) if isinstance(bucket, Mapping) else None
            if accuracy is None:
                continue
            try:
                key_count = int(str(key).split("/")[0])
            except ValueError:
                continue
            observed.append({"key_count": key_count, "accuracy": accuracy,
                             "correct": bucket.get("correct"), "total": bucket.get("total")})
        observed.sort(key=lambda item: item["key_count"])
        qualifying = [item["key_count"] for item in observed if item["accuracy"] >= accuracy_threshold]
        identity = (str(row.get("family", "unknown")), str(row.get("scale", "unknown")),
                    str(row.get("condition", row.get("variant", "unknown"))),
                    str(row.get("initialization", "unknown")), _protocol_version(row))
        seed = row.get("seed")
        seed_entry = {"family": identity[0], "scale": identity[1], "condition": identity[2],
                      "initialization": identity[3], "seed": seed, "observed": observed,
                      "capacity_key_count": max(qualifying) if qualifying else None,
                      "threshold_met": bool(qualifying), "capacity_metric": metric,
                      "protocol_version": identity[4], "status": row.get("status", "complete"),
                      "eligible_for_primary": _row_is_complete(row)}
        if not retrieval:
            seed_entry["capacity_unavailable_reason"] = "missing or incomplete retrieval evaluation artifact"
        if fixed_distance is not None:
            seed_entry["fixed_distance_requested"] = fixed_distance
            seed_entry["fixed_distance_present"] = any(
                isinstance(buckets, Mapping) and str(fixed_distance) in buckets for buckets in joint.values()
            ) if joint else False
            if not joint:
                seed_entry["capacity_unavailable_reason"] = "evaluation artifact has only marginal key-count breakdowns; distance-specific capacity cannot be computed"
            elif not seed_entry["fixed_distance_present"]:
                seed_entry["capacity_unavailable_reason"] = "no answers have the requested actual query distance"
        seed_rows.append(seed_entry)
        grouped[identity].append(seed_entry)

    summaries = []
    for identity, rows in sorted(grouped.items()):
        eligible_rows = [r for r in rows if r["eligible_for_primary"]]
        capacities = [r["capacity_key_count"] for r in eligible_rows if r["capacity_key_count"] is not None]
        summaries.append({
            "family": identity[0], "scale": identity[1], "condition": identity[2],
            "initialization": identity[3], "protocol_version": identity[4],
            "seed_count": len(rows), "eligible_seed_count": len(eligible_rows),
            "capacity_by_seed": [{"seed": r["seed"], "key_count": r["capacity_key_count"]} for r in sorted(rows, key=lambda x: str(x["seed"]))],
            "mean_capacity_key_count": mean(capacities) if capacities else None,
            "std_capacity_key_count": stdev(capacities) if len(capacities) > 1 else (0.0 if capacities else None),
            "conditional_mean_capacity_key_count": mean(capacities) if capacities else None,
            "seeds_meeting_threshold": len(capacities), "seeds_without_threshold_point": len(rows) - len(capacities),
            "capacity_estimate_scope": "conditional_on_complete_evaluation_and_threshold_point",
        })
    protocols = sorted({row["protocol_version"] for row in seed_rows})
    return {"analysis": "E3_observed_capacity", "accuracy_threshold": accuracy_threshold,
            "fixed_distance": fixed_distance, "metric": metric, "summaries": summaries,
            "seed_results": seed_rows,
            "protocol_versions": protocols,
            "interpretation": "Capacity is the largest tested key count meeting threshold; no interpolation or extrapolation is performed. Aggregate capacity is conditional on complete evaluation and a threshold point; failed or missing seeds remain visible and are not treated as zero capacity."}


def _field(row: Mapping[str, Any], name: str, default: Any = "unknown") -> Any:
    return row.get(name, default)


def _protocol_version(row: Mapping[str, Any]) -> str:
    """Return the scientific protocol identity carried by an analysis row."""
    value = row.get("protocol_version")
    if value is None and isinstance(row.get("manifest"), Mapping):
        value = row["manifest"].get("protocol_version")
    return str(value or "legacy-v1")


def _row_is_complete(row: Mapping[str, Any]) -> bool:
    """Whether a row is eligible for a primary model-level estimate."""
    status = str(row.get("status", "complete"))
    return status in {"complete", "evaluated"} and row.get("eligible_for_primary") is not False


def summarize_paired_interventions(tasks: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize E4 effects relative to each same-checkpoint unmodified result."""
    grouped: dict[tuple[str, str, str, int, str], dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in tasks:
        family = str(_field(row, "family"))
        scale = str(_field(row, "scale"))
        condition = str(_field(row, "condition", _field(row, "initialization")))
        seed = int(_field(row, "seed", -1))
        intervention = row.get("group_count")
        if intervention is None or seed < 0:
            if intervention is None:
                intervention = -1
            elif seed < 0:
                continue
        result = _load_payload(row.get("result_path", row.get("result")))
        if result is None:
            continue
        grouped[(family, scale, condition, seed, _protocol_version(row))][int(intervention)] = result
    metrics = ("exact_query_accuracy", "latest_value_accuracy", "stale_value_error_rate",
               "unrelated_association_retention")
    deltas: dict[tuple[str, str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for identity, by_group in grouped.items():
        if -1 not in by_group:
            continue
        baseline = by_group[-1]
        for group_count, current in sorted(by_group.items()):
            if group_count == -1:
                continue
            row: dict[str, Any] = {"group_count": group_count}
            for metric in metrics:
                value, base = _finite(current.get(metric)), _finite(baseline.get(metric))
                if value is not None and base is not None:
                    row[metric + "_delta"] = value - base
            deltas[identity].append(row)
    seed_effects = []
    metric_groups: dict[tuple[str, str, str, int, str], list[tuple[int, float]]] = defaultdict(list)
    for identity, entries in sorted(deltas.items()):
        for entry in entries:
            for metric in metrics:
                value = _finite(entry.get(metric + "_delta"))
                if value is not None:
                    metric_groups[(identity[0], identity[1], identity[2], identity[4], entry["group_count"], metric)].append((identity[3], value))
            seed_effects.append({"family": identity[0], "scale": identity[1], "condition": identity[2],
                                 "seed": identity[3], "protocol_version": identity[4], "effects": entries})
    aggregate = []
    for (family, scale, condition, protocol, groups, metric), values in sorted(metric_groups.items()):
        vals = [v for _, v in values]
        aggregate.append({"family": family, "scale": scale, "condition": condition,
                          "protocol_version": protocol,
                          "group_count": groups, "metric": metric,
                          "paired_seed_deltas": [{"seed": s, "delta": v} for s, v in values],
                          "mean_delta": mean(vals), "std_across_seeds": stdev(vals) if len(vals) > 1 else 0.0,
                          "n_seeds": len(vals)})
    return {"analysis": "E4_post_training_gate_sharing", "reference_group_count": -1,
            "reference_label": "unmodified channel-wise gates", "seed_effects": seed_effects,
            "aggregate": aggregate,
            "intervention": {"primary": "arithmetic_mean_of_effective_retention_factors_alpha",
                             "alternative": "geometric_mean_of_effective_retention_factors_exp(mean(log(alpha)))",
                             "model_implementation": "The existing runtime grouping averages alpha (effective retention factors) within each group, then broadcasts it.",
                             "warning": "These are inference interventions on channel-wise trained checkpoints; they do not measure separately trained grouped models or runtime savings.",
                             "causal_scope": "inference-time sensitivity only"}}


def _mean_sd_ci(values: list[float]) -> dict[str, Any]:
    n = len(values)
    avg = mean(values) if n else None
    sd = stdev(values) if n > 1 else (0.0 if n else None)
    # Student-t 95% two-sided critical values, small sample table; use normal
    # approximation beyond 30 df. This describes seed-level uncertainty.
    critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
                11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
                16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}
    half = critical.get(n - 1, 1.96) * sd / math.sqrt(n) if n > 1 and sd is not None else None
    return {"mean": avg, "std_across_seeds": sd, "n_seeds": n,
            "approx_95pct_t_interval": [avg - half, avg + half] if avg is not None and half is not None else None}


def paired_interaction(
    records: Iterable[Mapping[str, Any]], *, metric: str,
    granularity: str = "channel-wise", control: str = "scalar",
    default_init: str = "default", calibrated_init: str = "calibrated",
) -> dict[str, Any]:
    """Compute per-seed difference-in-differences for the primary E2/E6 contrast."""
    cells: dict[tuple[str, str, int, str, str, str], float] = {}
    for row in records:
        value = _finite(row.get(metric))
        if value is None:
            continue
        cells[(str(_field(row, "family")), str(_field(row, "scale")), int(_field(row, "seed", -1)),
               str(_field(row, "granularity", _field(row, "condition"))),
               str(_field(row, "initialization")), _protocol_version(row))] = value
    identities = {(f, s, seed, protocol) for f, s, seed, _, _, protocol in cells}
    effects = []
    by_family: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for family, scale, seed, protocol in sorted(identities):
        try:
            dflt = cells[(family, scale, seed, granularity, default_init, protocol)] - cells[(family, scale, seed, control, default_init, protocol)]
            calib = cells[(family, scale, seed, granularity, calibrated_init, protocol)] - cells[(family, scale, seed, control, calibrated_init, protocol)]
        except KeyError:
            continue
        interaction = calib - dflt
        by_family[(family, scale, protocol)].append(interaction)
        effects.append({"family": family, "scale": scale, "seed": seed, "protocol_version": protocol,
                        "granularity_minus_control_default": dflt,
                        "granularity_minus_control_calibrated": calib,
                        "difference_in_differences": interaction})
    summaries = [{"family": family, "scale": scale, "protocol_version": protocol, **_mean_sd_ci(values)}
                 for (family, scale, protocol), values in sorted(by_family.items())]
    protocols = sorted({identity[3] for identity in identities})
    return {"metric": metric, "contrast": f"({granularity}-{control})[{calibrated_init}] - ({granularity}-{control})[{default_init}]",
            "unit_of_uncertainty": "paired training seed; test examples are not treated as independent trained models",
            "protocol_versions": protocols, "mixed_protocols": len(protocols) > 1,
            "per_seed_effects": effects, "summary": summaries}


def build_e6_report(
    tasks: Iterable[Mapping[str, Any]], *, output_path: str | Path | None = None,
    metrics: tuple[str, ...] = ("retrieval.exact_query_accuracy", "retrieval.latest_value_accuracy",
                                "retrieval.stale_value_error_rate", "final_validation.loss",
                                "training.processed_tokens", "training.fixed_optimizer_steps",
                                "training.wall_time_seconds", "parameter.parameter_count"),
) -> dict[str, Any]:
    """Create compact top-level report referencing raw artifacts by path."""
    rows = list(tasks)
    per_seed = []
    grouped: dict[tuple[str, str, str, str, str, str], list[float]] = defaultdict(list)
    primary_grouped: dict[tuple[str, str, str, str, str, str], list[float]] = defaultdict(list)
    interaction_rows: list[dict[str, Any]] = []
    primary_interaction_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row in rows:
        refs = row.get("artifact_paths", {}) if isinstance(row.get("artifact_paths"), Mapping) else {}
        metrics_data = _load_payload(row.get("metrics")) or _load_payload(row.get("metrics_path")) or _load_payload(refs.get("metrics")) or {}
        identity = {key: row.get(key) for key in ("experiment", "family", "scale", "condition", "granularity", "initialization", "recipe", "seed", "status", "protocol_version")}
        identity["protocol_version"] = _protocol_version(row)
        identity["granularity"] = identity.get("granularity") or row.get("variant")
        artifact_refs = {key: str(row[key]) for key in ("manifest_path", "metrics_path", "checkpoint_path", "evaluation_path") if row.get(key)}
        artifact_refs.update({key: str(value) for key, value in refs.items() if value})
        missing = []
        declared_status = row.get("status")
        if declared_status is not None and declared_status not in {"complete", "evaluated"}:
            missing.append(f"task_status_{declared_status}")
        # A runner task row is an E6 completion claim only if its training and
        # required evaluation artifacts exist and lifecycle reports completion.
        if refs:
            required = ["manifest", "metrics", "final_checkpoint", "lifecycle"]
            scale = str(row.get("scale", ""))
            if row.get("family") in {"mqar", "overwrite"}:
                required.append(f"E3_{scale}")
            if row.get("variant") == "channel-wise":
                required.append(f"E4_{scale}")
            for name in required:
                path = refs.get(name)
                if not path or not Path(path).is_file():
                    missing.append(name)
            lifecycle = _load_payload(refs.get("lifecycle"))
            lifecycle_status = lifecycle.get("status") if lifecycle else None
            identity["status"] = row.get("status") or lifecycle_status or "unknown"
            if lifecycle_status not in {"complete", "evaluated"}:
                missing.append("lifecycle_status_not_complete")
            if missing:
                exclusions.append({"identity": {k: identity.get(k) for k in ("family", "scale", "condition", "seed")},
                                   "missing_or_incomplete": sorted(set(missing))})
        seed_metrics = {}
        for metric in metrics:
            value: Any = metrics_data
            for part in metric.split("."):
                value = value.get(part) if isinstance(value, Mapping) else None
            seed_metrics[metric] = _finite(value)
            if seed_metrics[metric] is not None:
                group = (str(row.get("family", "unknown")), str(row.get("scale", "unknown")),
                         str(row.get("condition", row.get("granularity", "unknown"))),
                         str(row.get("initialization", "unknown")), _protocol_version(row), metric)
                grouped[group].append(seed_metrics[metric])
        eligible_for_primary = not missing and _row_is_complete({**row, **identity})
        identity["eligible_for_primary"] = eligible_for_primary
        if eligible_for_primary:
            for metric, value in seed_metrics.items():
                if value is not None:
                    primary_grouped[(str(row.get("family", "unknown")), str(row.get("scale", "unknown")),
                                     str(row.get("condition", row.get("granularity", "unknown"))),
                                     str(row.get("initialization", "unknown")), _protocol_version(row), metric)].append(value)
        per_seed.append({**identity, "metrics": seed_metrics, "artifacts": artifact_refs})
        interaction_row = {**identity, **seed_metrics}
        interaction_rows.append(interaction_row)
        if eligible_for_primary:
            primary_interaction_rows.append(interaction_row)
    summaries = [{"family": key[0], "scale": key[1], "condition": key[2],
                  "initialization": key[3], "protocol_version": key[4], "metric": key[5], **_mean_sd_ci(vals)}
                 for key, vals in sorted(grouped.items())]
    primary_summaries = [{"family": key[0], "scale": key[1], "condition": key[2],
                          "initialization": key[3], "protocol_version": key[4], "metric": key[5], **_mean_sd_ci(vals)}
                         for key, vals in sorted(primary_grouped.items())]
    primary_interactions = {
        metric: paired_interaction(
            primary_interaction_rows, metric=metric,
            calibrated_init="calibrated-geometric" if any(r.get("initialization") == "calibrated-geometric" for r in primary_interaction_rows) else "calibrated",
        )
        for metric in metrics
    }
    if not rows:
        exclusions.append({"identity": {}, "missing_or_incomplete": ["no task records were supplied"]})
    protocols = sorted({_protocol_version(row) for row in rows})
    cohort_cells: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
    cohort_seeds: dict[tuple[str, str, str], set[Any]] = defaultdict(set)
    for row in rows:
        cohort = (str(row.get("family", "unknown")), str(row.get("scale", "unknown")), _protocol_version(row))
        granularity = str(row.get("granularity", row.get("condition", row.get("variant", "unknown"))))
        initialization = str(row.get("initialization", "unknown"))
        cohort_cells[cohort].add((granularity, initialization))
        cohort_seeds[cohort].add(row.get("seed"))
    cohort_audit = []
    for cohort, seeds in sorted(cohort_seeds.items()):
        expected_cells = cohort_cells[cohort]
        for seed in sorted(seeds, key=str):
            present = {
                (str(row.get("granularity", row.get("condition", row.get("variant", "unknown")))),
                 str(row.get("initialization", "unknown")))
                for row in rows if (str(row.get("family", "unknown")), str(row.get("scale", "unknown")), _protocol_version(row)) == cohort and row.get("seed") == seed and _row_is_complete(row)
            }
            missing_cells = sorted(expected_cells - present)
            cohort_audit.append({"family": cohort[0], "scale": cohort[1], "protocol_version": cohort[2],
                                 "seed": seed, "expected_cells": sorted(expected_cells),
                                 "missing_cells": missing_cells, "paired_complete": not missing_cells})
    incomplete_cohort = any(not row["paired_complete"] for row in cohort_audit)
    complete = bool(rows) and not exclusions and not incomplete_cohort and len(protocols) <= 1
    report = {"report_version": 1, "analysis": "E6", "status": "complete" if complete else "incomplete",
              "primary_comparison":
              "paired granularity × initialization difference-in-differences across training seeds",
              "uncertainty_note": "All aggregate uncertainty is computed over paired training seeds. Example-level counts are descriptive evaluation precision and are not independent model replicates.",
              "task_count": len(rows), "status_counts": _status_counts(per_seed),
              "protocol_versions": protocols, "mixed_protocols": len(protocols) > 1,
              "cohort_audit": cohort_audit,
              "exclusions": exclusions,
              "seed_records": per_seed, "summary": summaries,
              "primary_summary": primary_summaries,
              "primary_interactions": primary_interactions}
    if output_path is not None:
        atomic_write_json(output_path, report)
        report["paper_outputs"] = generate_e6_outputs(rows, Path(output_path).parent, report)
        atomic_write_json(output_path, report)
    return report


def _status_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("status", "unknown"))] += 1
    return dict(sorted(counts.items()))


def generate_e6_outputs(tasks: list[Mapping[str, Any]], root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    """Write a compact CSV and available figures; raw arrays stay in task files."""
    import csv
    root.mkdir(parents=True, exist_ok=True)
    table = root / "e6_seed_metrics.csv"
    columns = ("family", "scale", "condition", "granularity", "initialization", "seed", "status",
               "retrieval.exact_query_accuracy", "retrieval.latest_value_accuracy",
               "retrieval.stale_value_error_rate", "final_validation.loss",
               "training.processed_tokens", "training.fixed_optimizer_steps",
               "training.wall_time_seconds", "parameter.parameter_count")
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in report.get("seed_records", []):
            writer.writerow({**{key: row.get(key, "") for key in columns if "." not in key},
                             **row.get("metrics", {})})

    outputs: dict[str, Any] = {"seed_table": table.name, "figures": []}
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        outputs["figure_status"] = "matplotlib unavailable"
        return outputs

    # Quality by condition: a point per seed, so the training-run sample size is visible.
    quality_rows = [r for r in report.get("seed_records", [])
                    if r.get("metrics", {}).get("final_validation.loss") is not None]
    if quality_rows:
        fig, axis = plt.subplots(figsize=(max(7, len(quality_rows) * .38), 4))
        labels = [f"{r.get('family')}\n{r.get('granularity') or r.get('condition')}\n{r.get('initialization')}\n{r.get('seed')}" for r in quality_rows]
        axis.scatter(range(len(quality_rows)), [r["metrics"]["final_validation.loss"] for r in quality_rows])
        axis.set_xticks(range(len(quality_rows)), labels, rotation=60, ha="right", fontsize=7)
        axis.set_ylabel("Final validation loss")
        axis.set_title("Per-seed quality")
        fig.tight_layout()
        path = root / "e6_quality_by_seed.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs["figures"].append(path.name)

    # Learning curves use only points persisted with an actual x coordinate.
    curve_groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in tasks:
        metrics_data = _load_payload(row.get("metrics")) or _load_payload(row.get("metrics_path")) or {}
        for point in metrics_data.get("validation_history", []):
            if not isinstance(point, Mapping):
                continue
            x = _finite(point.get("processed_tokens"))
            x_label = "processed_tokens"
            if x is None:
                x = _finite(point.get("wall_time_seconds"))
                x_label = "wall_time_seconds"
            if x is None:
                x = _finite(point.get("step"))
                x_label = "optimizer_step"
            y = _finite(point.get("loss"))
            if x is not None and y is not None:
                label = "/".join(str(row.get(k, "?")) for k in ("family", "variant", "initialization", "seed"))
                curve_groups[label].append((x, y))
    if curve_groups:
        fig, axis = plt.subplots(figsize=(7, 4))
        x_label = "Recorded training coordinate"
        for label, points in curve_groups.items():
            points.sort()
            axis.plot([x for x, _ in points], [y for _, y in points], label=label)
        axis.set(xlabel=x_label, ylabel="Validation loss", title="Persisted validation learning curves")
        if len(curve_groups) <= 12:
            axis.legend(fontsize=6)
        fig.tight_layout()
        path = root / "e6_learning_curves.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs["figures"].append(path.name)
    else:
        outputs["learning_curve_status"] = "no validation history with a recorded token, wall-time, or optimizer-step coordinate"

    # Capacity and intervention plots are built from separately referenced E3/E4 artifacts.
    capacity_points: dict[str, list[tuple[int, float]]] = defaultdict(list)
    intervention_points: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in tasks:
        refs = row.get("artifact_paths", {})
        stage_refs: list[tuple[str, str, Any]] = []
        if isinstance(refs, Mapping):
            stage_refs.extend((str(key), "E3" if str(key).startswith("E3_") else "E4", value)
                              for key, value in refs.items() if str(key).startswith(("E3_", "E4_")))
            stage_refs.extend((key, stage, refs.get(key)) for key, stage in (("e3", "E3"), ("capacity", "E3"), ("e4", "E4"), ("intervention", "E4")))
        stage_refs.extend((key, stage, row.get(key + "_path")) for key, stage in (("e3", "E3"), ("capacity", "E3"), ("e4", "E4"), ("intervention", "E4")))
        for key, stage, path_value in stage_refs:
            if not path_value:
                continue
            payload = _load_payload(path_value)
            if not payload:
                continue
            if stage == "E3":
                for item in payload.get("grid_results", []):
                    metrics_data = item.get("metrics", {})
                    acc = _finite(metrics_data.get("exact_query_accuracy"))
                    point = item.get("grid", {})
                    key_count = int(point.get("key_count", 0))
                    if acc is not None:
                        label = (f"{row.get('condition_id', row.get('variant', 'condition'))}/"
                                 f"L{point.get('sequence_length')}/D{point.get('query_distance')}/R{point.get('overwrite_count')}")
                        capacity_points[label].append((key_count, acc))
            else:
                for item in payload.get("paired_deltas", {}).get("mean_alpha", []):
                    delta_values = item.get("deltas_vs_unmodified", {})
                    delta = _finite(delta_values.get("exact_query_accuracy"))
                    ylabel = "Accuracy change vs unmodified"
                    if delta is None:
                        delta = _finite(delta_values.get("validation_perplexity_delta_vs_full_channel"))
                        ylabel = "Validation perplexity change vs unmodified"
                    groups = int(item.get("group_count", 0))
                    if delta is not None:
                        intervention_points[str(row.get("condition_id", row.get("variant", "condition"))) + "|" + ylabel].append((groups, delta))
    if capacity_points:
        fig, axis = plt.subplots(figsize=(6, 4))
        for label, points in capacity_points.items():
            by_key: dict[int, list[float]] = defaultdict(list)
            for key, accuracy in points:
                by_key[key].append(accuracy)
            axis.plot(sorted(by_key), [mean(by_key[key]) for key in sorted(by_key)], marker="o", label=label)
        axis.axhline(.9, color="gray", linestyle="--", linewidth=.8)
        axis.set(xlabel="Tested key count", ylabel="Exact answer accuracy", title="E3 recall by tested key count")
        axis.legend(fontsize=7)
        fig.tight_layout()
        path = root / "e6_e3_capacity.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs["figures"].append(path.name)
    if intervention_points:
        fig, axis = plt.subplots(figsize=(6, 4))
        for label, points in intervention_points.items():
            by_group: dict[int, list[float]] = defaultdict(list)
            for groups, delta in points:
                by_group[groups].append(delta)
            axis.plot(sorted(by_group), [mean(by_group[group]) for group in sorted(by_group)], marker="o", label=label)
        axis.axhline(0, color="black", linewidth=.8)
        axis.set(xlabel="Shared gates per head", ylabel="Metric change vs unmodified", title="E4 post-training intervention")
        axis.legend(fontsize=7)
        fig.tight_layout()
        path = root / "e6_e4_intervention.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs["figures"].append(path.name)
    return outputs


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run one study analysis task from runner JSON")
    parser.add_argument("--task-json", required=True, help="Path to a JSON task record")
    args = parser.parse_args()
    payload = read_json(args.task_json)
    print(json.dumps(run_study_task(payload), indent=2, sort_keys=True))


def _path_in_task(task: Mapping[str, Any], name: str) -> Path:
    artifacts = task.get("artifact_paths", {})
    candidate = artifacts.get(name) if isinstance(artifacts, Mapping) else None
    candidate = candidate or task.get(name + "_path")
    if candidate is None:
        raise ValueError(f"Task is missing artifact path {name!r}")
    return Path(candidate)


def _load_study_model(task: Mapping[str, Any], device: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Load an evaluation-only model from its immutable manifest/checkpoint."""
    import torch
    from config import ModelConfig
    from controls import construct_model
    from recurrence import RecurrenceOptions

    manifest = read_json(_path_in_task(task, "manifest"))
    model_data = manifest["model"]
    model_config = ModelConfig(
        name=str(model_data.get("name", task.get("scale", "model"))),
        d_model=int(model_data["d_model"]), n_layer=int(model_data["n_layer"]),
        n_head=int(model_data["n_head"]), d_ff=tuple(int(x) for x in model_data["d_ff"]),
        attention_layout=str(model_data["attention_layout"]),
    )
    recurrence = model_data.get("recurrence", {})
    options = RecurrenceOptions(
        backend=recurrence.get("requested_backend", "reference"),
        input_dtype=recurrence.get("input_dtype", "autocast"),
        state_accumulator_dtype=recurrence.get("state_accumulator_dtype", "input"),
        chunk_size=int(recurrence.get("chunk_size", 64)),
        qualification_artifact=recurrence.get("qualification_artifact"),
    )
    family = str(task.get("family", manifest["family"]))
    variant = str(task.get("variant", manifest.get("variant", model_data.get("variant", "channel-wise"))))
    retrieval_config = task.get("retrieval_config", {})
    inferred_vocab = retrieval_config.get("vocabulary_size", 8_192) if isinstance(retrieval_config, Mapping) else 8_192
    vocab_size = int(task.get("vocab_size", 50_257 if family == "tinystories" else inferred_vocab))
    # Initialization and group width are part of the study task/manifest metadata.
    initialization = str(task.get("initialization", manifest.get("initialization", "default")))
    gate_groups = int(task.get("gate_groups", model_data.get("gate_groups", 4))) if variant == "grouped-4" else None
    study_init_seed = model_data.get("study_init_seed")
    model, _, _ = construct_model(model_config, variant, vocab_size,
                                   recurrence_options=options, gate_groups=gate_groups,
                                   gate_initialization=initialization,
                                   study_init_seed=int(study_init_seed) if study_init_seed is not None else None)
    checkpoint_path = _path_in_task(task, "final_checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint.get("model", checkpoint))
    model.load_state_dict(state)
    model.to(device).eval()
    return model, manifest, {"family": family, "variant": variant, "vocab_size": vocab_size}


def _retrieval_config(task: Mapping[str, Any], grid: list[Mapping[str, Any]]) -> Any:
    from config import RetrievalConfig
    raw = task.get("retrieval_config", {})
    valid = set(RetrievalConfig.__dataclass_fields__)
    kwargs = {key: value for key, value in raw.items() if key in valid} if isinstance(raw, Mapping) else {}
    # Every E3 grid point creates its own dataset, so these default lists do not
    # determine the realized task difficulty.
    return RetrievalConfig(**kwargs)


def _grid_from_config(task: Mapping[str, Any]) -> list[dict[str, int]]:
    explicit = task.get("evaluation_grid")
    if explicit:
        return [{key: int(item[key]) for key in ("sequence_length", "key_count", "query_distance", "overwrite_count")}
                for item in explicit]
    raw = task.get("retrieval_config", {})
    if not isinstance(raw, Mapping):
        raise ValueError("E3 requires evaluation_grid or retrieval_config")
    lengths, keys = raw.get("sequence_lengths", (128, 256, 512)), raw.get("key_counts", (4, 8, 16, 32))
    distances, rewrites = raw.get("query_distances", (16, 64, 128)), raw.get("overwrite_counts", (1, 2, 3))
    return [{"sequence_length": int(length), "key_count": int(key), "query_distance": int(distance),
             "overwrite_count": int(overwrite)}
            for length in lengths for key in keys for distance in distances for overwrite in rewrites]


def _training_distribution_label(task: Mapping[str, Any], point: Mapping[str, int]) -> dict[str, Any]:
    raw = task.get("retrieval_config")
    if not isinstance(raw, Mapping):
        return {"training_distribution": "unknown", "reason": "training retrieval configuration was not provided"}
    fields = (("sequence_length", "sequence_lengths"), ("key_count", "key_counts"),
              ("query_distance", "query_distances"))
    if str(task.get("family")) == "overwrite":
        fields += (("overwrite_count", "overwrite_counts"),)
    missing = [field for field, config_key in fields if point.get(field) not in raw.get(config_key, ())]
    return {"training_distribution": "extrapolation" if missing else "in_training_distribution",
            "dimensions_outside_training_distribution": missing,
            "source": "locked retrieval_config difficulty lists",
            "distance_semantics": "classification uses configured anchor distance; capacity is separately stratified by actual per-answer distance"}


def _capture_analysis(device: Any, operation: Any) -> tuple[Any, dict[str, Any]]:
    from resource_telemetry import DeviceResourceCapture
    capture = DeviceResourceCapture(device)
    started = time.perf_counter()
    with capture:
        value = operation()
    return value, {"elapsed_seconds": time.perf_counter() - started,
                   "device_resources": capture.result or {"status": "unavailable"}}


@torch.inference_mode()
def _evaluate_grid_point(model: Any, task: Mapping[str, Any], grid_point: Mapping[str, int], device: Any,
                         *, examples: int, seed: int) -> dict[str, Any]:
    from dataclasses import replace
    from config import RetrievalConfig
    from retrieval import RetrievalDataset, evaluate_retrieval
    family = str(task.get("family", "mqar"))
    kind = "mqar" if family == "mqar" else "overwrite"
    base = _retrieval_config(task, [grid_point])
    config = replace(base,
                     sequence_lengths=(int(grid_point["sequence_length"]),),
                     key_counts=(int(grid_point["key_count"]),),
                     query_distances=(int(grid_point["query_distance"]),),
                     overwrite_counts=(int(grid_point["overwrite_count"]),),
                     held_out_examples=examples)
    seed = int(task.get("evaluation_seed", seed))
    dataset = RetrievalDataset(task=kind, examples=examples, config=config,
                               seed=seed, split="held_out", sequence_length=int(grid_point["sequence_length"]))
    count = min(len(dataset), examples)
    indices = tuple(round(i * (len(dataset) - 1) / max(count - 1, 1)) for i in range(count))
    metrics = evaluate_retrieval(model, dataset, device, example_indices=indices)
    if kind == "overwrite":
        latest_total = sum(answer.kind == "latest" for index in indices for answer in dataset.example(index).answers)
        stale = int(metrics.get("stale_value_errors", round((_finite(metrics.get("stale_value_error_rate")) or 0.0) * latest_total)))
        other = int(metrics.get("other_latest_value_errors", round((_finite(metrics.get("other_latest_value_error_rate")) or 0.0) * latest_total)))
        metrics["latest_answer_error_breakdown"] = {
            "stale_obsolete_value_errors": stale,
            "other_incorrect_value_errors": other,
            "latest_answers": latest_total,
            "stale_obsolete_value_error_rate": stale / latest_total if latest_total else None,
            "other_incorrect_value_error_rate": other / latest_total if latest_total else None,
            "definition": "Among latest-value query answers, wrong predictions matching any obsolete write are stale errors; other wrong predictions are non-stale errors.",
        }
    return {"grid": dict(grid_point), "metrics": metrics, "evaluated_examples": count,
            "example_indices": list(indices)}


def run_study_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Execute isolated E3/E4 checkpoint evaluation or E6 compact aggregation.

    E3/E4 output files are separate from ordinary ``metrics.json`` and lifecycle
    state, so incomplete analyses cannot make a training task appear complete.
    """
    stage = str(task.get("stage", "")).upper()
    output_path = Path(task.get("output_path", task.get("stage_output_path", "")))
    if not task.get("output_path", task.get("stage_output_path")):
        raise ValueError("Study analysis task requires output_path")
    if stage == "E6":
        result = build_e6_report(task.get("tasks", ()), output_path=output_path)
        if result.get("status") != "complete":
            raise RuntimeError(f"E6 analysis incomplete; missing/incomplete artifacts are listed in {output_path}")
        return result
    if stage not in {"E3", "E4"}:
        raise ValueError(f"Unsupported study analysis stage {stage!r}")
    import torch
    if stage == "E4" and str(task.get("family")) == "tinystories":
        refs = task.get("artifact_paths", {}) if isinstance(task.get("artifact_paths"), Mapping) else {}
        metrics_data = _load_payload(task.get("metrics")) or _load_payload(task.get("metrics_path")) or _load_payload(refs.get("metrics"))
        posthoc = metrics_data.get("posthoc_gate_grouping") if metrics_data else None
        if not isinstance(posthoc, Mapping) or posthoc.get("status") != "complete":
            raise ValueError("TinyStories E4 requires completed posthoc_gate_grouping in the ordinary metrics artifact")
        lm_deltas = []
        for group, delta in posthoc.get("sensitivity_vs_full_channel", {}).items():
            if not isinstance(delta, Mapping):
                continue
            try:
                group_count = int(group)
            except (TypeError, ValueError):
                continue
            lm_deltas.append({"group_count": group_count, "grid": {},
                              "deltas_vs_unmodified": {
                                  key: value
                                  for key, value in delta.items()
                                  if key.endswith("_delta_vs_full_channel") and _finite(value) is not None
                              }})
        result = {"analysis": "E4_post_training_gate_sharing", "identity": {key: task.get(key) for key in
                  ("experiment_id", "condition_id", "family", "scale", "variant", "initialization", "seed")},
                  "source": "existing validation subset intervention recorded during ordinary evaluation",
                  "primary_convention": "mean_alpha, as documented by existing model runtime grouping",
                  "metrics_by_group": posthoc.get("metrics_by_group_count", {}),
                  "paired_sensitivity_vs_unmodified": posthoc.get("sensitivity_vs_full_channel", {}),
                  "paired_deltas": {"mean_alpha": lm_deltas},
                  "alternative_convention": {"status": "not_available_in_existing_artifact",
                                              "reason": "the earlier validation-only screening did not persist alternative aggregation results"},
                  "interpretation": posthoc.get("interpretation_note")}
        atomic_write_json(output_path, result)
        return result
    device = torch.device(task.get("device") or ("cuda" if task.get("run_mode", "production") == "production" and torch.cuda.is_available() else "cpu"))
    model, manifest, model_identity = _load_study_model(task, device)
    family = model_identity["family"]
    if family not in {"mqar", "overwrite"}:
        raise ValueError(f"{stage} retrieval analysis is not defined for family {family!r}")
    grid = _grid_from_config(task)
    examples = int(task.get("evaluation_examples", 512))
    seed = int(task.get("seed", manifest.get("seeds", {}).get("research_seed", 0)))
    if stage == "E3":
        rows, resource_data = _capture_analysis(device, lambda: [
            _evaluate_grid_point(model, task, point, device, examples=examples, seed=seed) for point in grid
        ])
        for row in rows:
            row.update(_training_distribution_label(task, row["grid"]))
        result = {"analysis": "E3_capacity_curves", "identity": _task_identity(task, manifest),
                  "evaluation_protocol": {"split": "held_out", "examples_per_point": examples,
                                          "grid": grid, "generator_version": rows[0]["metrics"]["generator_manifest"]["generator_version"] if rows else None},
                  "distance_semantics": {"grid_query_distance": "configured anchor distance used to generate the example grid",
                                         "capacity_query_distance": "actual per-answer query position minus latest relevant value position; computed from joint key-count × actual-distance counts"},
                  "grid_results": rows,
                  "evaluation_runtime": resource_data,
                  "interpretation": "Per-point held-out results. Capacity summaries use tested points only; no extrapolation."}
        for row in result["grid_results"]:
            row["anchor_distance"] = row["grid"]["query_distance"]
            row["grid"]["anchor_distance"] = row["grid"]["query_distance"]
        result["status"] = "complete"
        result["capacity_summary"] = _capacity_from_grid_rows(result, threshold=float(task.get("capacity_threshold", .90)))
        if result["capacity_summary"]["status"] != "complete":
            result["status"] = "incomplete"
    else:
        groups = [int(g) for g in task.get("group_counts", (1, 2, 4, 8))]
        group_results: dict[str, Any] = {}
        group_results["unmodified"], unmodified_resources = _capture_analysis(device, lambda: [
            _evaluate_grid_point(model, task, point, device, examples=examples, seed=seed) for point in grid
        ])
        from diagnostics import collect_gate_diagnostics
        first_dataset_config = _retrieval_config(task, [grid[0]])
        from dataclasses import replace
        from retrieval import RetrievalDataset
        diagnostic_config = replace(first_dataset_config,
                                    sequence_lengths=(int(grid[0]["sequence_length"]),),
                                    key_counts=(int(grid[0]["key_count"]),),
                                    query_distances=(int(grid[0]["query_distance"]),),
                                    overwrite_counts=(int(grid[0]["overwrite_count"]),))
        diagnostic_dataset = RetrievalDataset(task="mqar" if family == "mqar" else "overwrite",
                                              examples=1, config=diagnostic_config,
                                              seed=int(task.get("evaluation_seed", seed)), split="held_out",
                                              sequence_length=int(grid[0]["sequence_length"]))
        gate_behavior, diagnostics_resources = _capture_analysis(device, lambda: collect_gate_diagnostics(
            model, [diagnostic_dataset.example(0).tokens.unsqueeze(0)], device))
        for row in group_results["unmodified"]:
            row.update(_training_distribution_label(task, row["grid"]))
        sensitivity_results: dict[str, Any] = {}
        intervention_resources: dict[str, Any] = {}
        for aggregation in ("mean_alpha", "mean_log_decay"):
            sensitivity_results[aggregation] = {}
            def evaluate_interventions() -> dict[str, Any]:
                values: dict[str, Any] = {}
                try:
                    for groups_per_head in groups:
                        model.set_runtime_gate_groups(groups_per_head, aggregation=aggregation)
                        values[str(groups_per_head)] = [
                            _evaluate_grid_point(model, task, point, device, examples=examples, seed=seed)
                            for point in grid
                        ]
                        for row in values[str(groups_per_head)]:
                            row.update(_training_distribution_label(task, row["grid"]))
                finally:
                    model.set_runtime_gate_groups(None)
                return values
            sensitivity_results[aggregation], intervention_resources[aggregation] = _capture_analysis(device, evaluate_interventions)
        result = {"analysis": "E4_post_training_gate_sharing", "identity": _task_identity(task, manifest),
                  "evaluation_protocol": {"split": "held_out", "examples_per_point": examples, "grid": grid},
                  "group_count_convention": "A group count is the number of shared gates per head; unmodified retains every channel gate.",
                  "gate_behavior_diagnostic": {"layers": gate_behavior.get("layers", {}),
                                               "channel_symmetry_scale": "square_root_of_mean_within_head_alpha_variance",
                                               "half_life_note": "Decay-only diagnostic; delta writes also affect memory retention."},
                  "evaluation_runtime": {"unmodified": unmodified_resources,
                                         "gate_diagnostics": diagnostics_resources,
                                         "interventions": intervention_resources},
                  "intervention": {"primary": "arithmetic_mean_of_effective_retention_factors_alpha",
                                   "alternative": "geometric_mean_of_effective_retention_factors_exp(mean(log(alpha)))",
                                   "existing_convention": "The original post-hoc grouping implementation averages alpha values within each group, then broadcasts.",
                                   "alpha_note": "alpha is the effective retention factor in the recurrence; log(alpha) aggregation is included as a bounded sensitivity check.",
                                   "causal_scope": "inference-time sensitivity only; no separately trained grouped models or runtime savings"},
                  "metrics_by_group": sensitivity_results["mean_alpha"],
                  "sensitivity_by_alternative_convention": sensitivity_results["mean_log_decay"],
                  "paired_deltas": {"mean_alpha": _group_result_deltas({**group_results, **sensitivity_results["mean_alpha"]}),
                                    "mean_log_decay": _group_result_deltas({**group_results, **sensitivity_results["mean_log_decay"]})},
                  "interpretation": "Same checkpoint, same held-out examples; inference sensitivity only, not grouped-model training quality or runtime savings."}
    atomic_write_json(output_path, result)
    if stage == "E3" and result.get("status") != "complete":
        raise RuntimeError(f"E3 actual-distance capacity breakdown unavailable; details persisted to {output_path}")
    return result


def _task_identity(task: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {"experiment_id": task.get("experiment_id"), "condition_id": task.get("condition_id"),
            "family": task.get("family", manifest.get("family")), "scale": task.get("scale", manifest.get("scale")),
            "variant": task.get("variant", manifest.get("variant")),
            "initialization": task.get("initialization", manifest.get("initialization")),
            "seed": task.get("seed", manifest.get("seeds", {}).get("research_seed"))}


def _capacity_from_grid_rows(result: Mapping[str, Any], *, threshold: float) -> dict[str, Any]:
    # Capacity is based on joint per-answer *actual* distance cells. The grid's
    # requested distance may differ across individual answers in a multi-query
    # example, so never use a pooled per-grid accuracy as a fixed-distance point.
    cells: dict[tuple[int, int, int, int], list[int]] = defaultdict(lambda: [0, 0])
    unavailable_reasons: list[str] = []
    for row in result.get("grid_results", []):
        point = row["grid"]
        metrics = row["metrics"]
        breakdowns = metrics.get("breakdowns", {})
        joint = metrics.get("key_count_by_query_distance")
        if joint is None and isinstance(breakdowns, Mapping):
            joint = breakdowns.get("key_count_by_query_distance")
        if not isinstance(joint, Mapping):
            unavailable_reasons.append("missing key_count_by_query_distance in retrieval metrics")
            continue
        for key_count, distance_buckets in joint.items():
            if not isinstance(distance_buckets, Mapping):
                continue
            for actual_distance, bucket in distance_buckets.items():
                if not isinstance(bucket, Mapping):
                    continue
                correct, total = bucket.get("correct"), bucket.get("total")
                if isinstance(correct, int) and isinstance(total, int) and total > 0:
                    key = (int(point["sequence_length"]), int(point["query_distance"]), int(actual_distance),
                           int(point["overwrite_count"]), int(key_count))
                    cells[key][0] += correct
                    cells[key][1] += total
    conditions: dict[tuple[int, int, int, int], list[tuple[int, float, int, int]]] = defaultdict(list)
    for (length, anchor_distance, actual_distance, rewrites, key_count), (correct, total) in cells.items():
        conditions[(length, anchor_distance, actual_distance, rewrites)].append((key_count, correct / total, correct, total))
    rows = []
    for (length, anchor_distance, actual_distance, rewrites), values in sorted(conditions.items()):
        tested = sorted(values, key=lambda item: item[0])
        passed = [key for key, accuracy, _, _ in tested if accuracy >= threshold]
        rows.append({"sequence_length": length, "anchor_distance": anchor_distance,
                     "actual_query_distance": actual_distance, "overwrite_count": rewrites,
                     "largest_tested_key_count_meeting_threshold": max(passed) if passed else None,
                     "threshold_met": bool(passed), "tested_points": [{"key_count": k, "accuracy": a, "correct": c, "total": n} for k, a, c, n in tested]})
    return {"threshold": threshold, "metric": "exact_query_accuracy", "by_fixed_difficulty": rows,
            "unmet_threshold_conditions": sum(not row["threshold_met"] for row in rows),
            "status": "complete" if not unavailable_reasons else ("unavailable" if not cells else "partial"),
            "unavailable_reasons": sorted(set(unavailable_reasons)),
            "distance_definition": "actual query token position minus most recent relevant value token position, as recorded per answer",
            "note": "No interpolation or extrapolation beyond tested key counts; key-count accuracy is pooled only within exact sequence length, configured anchor distance, actual per-answer distance, and overwrite-count cells."}


def _group_result_deltas(groups: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    baseline = groups.get("unmodified", [])
    output = []
    metrics = ("exact_query_accuracy", "latest_value_accuracy", "stale_value_error_rate", "unrelated_association_retention")
    for group, rows in groups.items():
        if group == "unmodified":
            continue
        for index, row in enumerate(rows):
            base = baseline[index]["metrics"] if index < len(baseline) else {}
            deltas = {metric: _finite(row["metrics"].get(metric)) - _finite(base.get(metric))
                      for metric in metrics if _finite(row["metrics"].get(metric)) is not None and _finite(base.get(metric)) is not None}
            output.append({"group_count": int(group), "grid": row["grid"], "deltas_vs_unmodified": deltas})
    return output


if __name__ == "__main__":
    main()
