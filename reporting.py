"""Descriptive, paired reporting rebuilt from validated per-seed artifacts."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

from artifacts import SeedManifest, atomic_write_json, read_json, validate_completion


def _metric_value(metrics: dict[str, Any], metric: str) -> float | None:
    path = metric.split(".")
    value: Any = metrics
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def descriptive_summary(records: Iterable[dict[str, Any]], metric_paths: tuple[str, ...]) -> dict[str, Any]:
    rows = list(records)
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["scale"], row["variant"], str(row.get("protocol_version", "legacy-v1")))].append(row)
    summary: dict[str, Any] = {}
    for (family, scale, variant, protocol_version), values in sorted(grouped.items()):
        entry: dict[str, Any] = {"family": family, "scale": scale, "variant": variant,
                                 "protocol_version": protocol_version, "seeds": []}
        for record in sorted(values, key=lambda item: item["seed"]):
            entry["seeds"].append({"seed": record["seed"], "metrics": {metric: _metric_value(record["metrics"], metric) for metric in metric_paths}})
        entry["descriptive"] = {}
        for metric in metric_paths:
            observed = [item["metrics"][metric] for item in entry["seeds"] if item["metrics"][metric] is not None]
            entry["descriptive"][metric] = {
                "mean": mean(observed) if observed else None,
                "std": stdev(observed) if len(observed) > 1 else 0.0 if observed else None,
                "n": len(observed),
            }
        summary[f"{family}/{scale}/{variant}/{protocol_version}"] = entry
    # Explicit pairwise differences, not p-values/significance stars.
    for family_scale in {(record["family"], record["scale"], str(record.get("protocol_version", "legacy-v1"))) for record in rows}:
        family, scale, protocol_version = family_scale
        scalar = {item["seed"]: item for item in grouped.get((family, scale, "scalar", protocol_version), [])}
        for comparison in ("channel-wise", "scalar-matched"):
            other = {item["seed"]: item for item in grouped.get((family, scale, comparison, protocol_version), [])}
            pairs = []
            for seed in sorted(scalar.keys() & other.keys()):
                differences = {}
                for metric in metric_paths:
                    left, right = _metric_value(scalar[seed]["metrics"], metric), _metric_value(other[seed]["metrics"], metric)
                    differences[metric] = right - left if left is not None and right is not None else None
                pairs.append({"seed": seed, "difference": differences})
            if pairs:
                summary[f"{family}/{scale}/{comparison}_minus_scalar"] = {
                    "family": family, "scale": scale, "protocol_version": protocol_version,
                    "comparison": f"{comparison}_minus_scalar", "paired_seed_differences": pairs,
                }
    return summary


def rebuild_validated_records(
    root: str | Path, *, expected_run_mode: str | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for manifest_path in Path(root).rglob("manifest.json"):
        run_dir = manifest_path.parent
        try:
            raw = read_json(manifest_path)
            if expected_run_mode is not None and raw.get("run_mode") != expected_run_mode:
                exclusions.append({"path": str(run_dir), "reason": f"{raw.get('run_mode', 'legacy')} artifact excluded from {expected_run_mode} report"})
                continue
            expected = SeedManifest(
                family=raw["family"], run_mode=raw["run_mode"], scale=raw["scale"], variant=raw["variant"],
                model=raw["model"], seeds=raw["seeds"], budget=raw["budget"], data_identity=raw["data_identity"],
                evaluation_protocol=raw["evaluation_protocol"], expected_artifacts=tuple(raw["expected_artifacts"]),
                environment=raw["environment"], source=raw["source"],
                protocol_version=raw.get("protocol_version", "legacy-v1"),
                schema_version=raw["schema_version"],
            )
            valid, reason = validate_completion(run_dir, expected)
            if not valid:
                exclusions.append({"path": str(run_dir), "reason": reason})
                continue
            metrics = read_json(run_dir / "metrics.json")
            records.append({
                "family": raw["family"], "scale": raw["scale"], "variant": raw["variant"],
                "seed": raw["seeds"]["research_seed"], "metrics": metrics, "path": str(run_dir),
                "protocol_version": raw.get("protocol_version", "legacy-v1"),
            })
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            exclusions.append({"path": str(run_dir), "reason": f"corrupt or legacy artifact: {exc}"})
    return records, exclusions


def _collect_backend_identity(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate backend/provenance identity across validated records (additive)."""

    requested: set[str] = set()
    resolved: set[str] = set()
    semantic_versions: set[str] = set()
    chunk_sizes: set[str] = set()
    input_dtypes: set[str] = set()
    accumulator_dtypes: set[str] = set()
    qualification_hashes: set[str] = set()
    load_failure_reasons: list[dict[str, str]] = []
    for record in records:
        metrics = record.get("metrics", {})
        recurrence = metrics.get("recurrence", {})
        static_recurrence = metrics.get("static_cost", {}).get("recurrence", {}) if isinstance(metrics.get("static_cost"), dict) else {}
        for source in (recurrence, static_recurrence):
            if not isinstance(source, dict):
                continue
            if source.get("requested_backend") is not None:
                requested.add(str(source["requested_backend"]))
            if source.get("resolved_backend") is not None:
                resolved.add(str(source["resolved_backend"]))
            for key, bucket in (
                ("semantic_version", semantic_versions), ("chunk_size", chunk_sizes),
                ("input_dtype", input_dtypes), ("state_accumulator_dtype", accumulator_dtypes),
                ("qualification_hash", qualification_hashes),
            ):
                if source.get(key) is not None:
                    bucket.add(str(source[key]))
            if source.get("load_failure_reason"):
                load_failure_reasons.append({"path": record.get("path", ""), "reason": str(source["load_failure_reason"])})
    return {
        "requested_backends": sorted(requested),
        "resolved_backends": sorted(resolved),
        "semantic_versions": sorted(semantic_versions),
        "chunk_sizes": sorted(chunk_sizes),
        "input_dtypes": sorted(input_dtypes),
        "state_accumulator_dtypes": sorted(accumulator_dtypes),
        "qualification_hashes": sorted(qualification_hashes),
        "load_failure_reasons": load_failure_reasons,
    }


def _scan_profiling_artifacts(root: Path) -> dict[str, Any]:
    """Summarize pilot/bench artifacts found under root without requiring them."""

    pilots: list[dict[str, Any]] = []
    for path in sorted(root.rglob("pilot_result.json")):
        try:
            payload = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            pilots.append({"path": str(path), "status": "unreadable", "reason": str(exc)})
            continue
        runs = payload.get("runs", []) if isinstance(payload, dict) else []
        entry: dict[str, Any] = {"path": str(path), "status": "ok", "runs": len(runs) if isinstance(runs, list) else 0}
        recurrence = payload.get("recurrence", {}) if isinstance(payload, dict) else {}
        if isinstance(recurrence, dict):
            entry["resolved_backend"] = recurrence.get("resolved_backend")
            entry["qualification_hash"] = recurrence.get("qualification_hash")
        pilots.append(entry)
    benchmarks: list[dict[str, Any]] = []
    for name in ("recurrence_benchmark.json", "recurrence_microbenchmark.json"):
        for path in sorted(root.rglob(name)):
            try:
                payload = read_json(path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                benchmarks.append({"path": str(path), "status": "unreadable", "reason": str(exc)})
                continue
            cases = payload.get("cases", []) if isinstance(payload, dict) else []
            benchmarks.append({
                "path": str(path), "status": "ok",
                "cases": len(cases) if isinstance(cases, list) else 0,
                "speedups": payload.get("speedups", []) if isinstance(payload, dict) else [],
            })
    return {
        "pilots": pilots,
        "recurrence_benchmarks": benchmarks,
        "note": "Absent pilot/bench artifacts are reported as empty lists, never fabricated.",
    }


def _collect_performance_comparisons(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Speed/memory/energy comparisons rebuilt from validated metrics (additive)."""

    comparisons: dict[str, Any] = {}
    for record in records:
        key = f"{record['family']}/{record['scale']}/{record['variant']}"
        metrics = record.get("metrics", {})
        benchmarks = metrics.get("benchmarks", {}) if isinstance(metrics.get("benchmarks"), dict) else {}
        tokens_per_second: dict[str, float | None] = {}
        for context, value in benchmarks.items():
            if isinstance(value, dict) and isinstance(value.get("prefill"), dict):
                tokens = value["prefill"].get("tokens_per_second")
                tokens_per_second[str(context)] = float(tokens) if isinstance(tokens, (int, float)) and math.isfinite(tokens) else None
        energy = metrics.get("energy", {})
        training_energy = metrics.get("training_energy", {})
        resource_usage: dict[str, Any] = {}
        for phase in ("training_resources", "evaluation_resources"):
            capture = metrics.get(phase)
            if isinstance(capture, dict):
                resource_usage[phase.removesuffix("_resources")] = {
                    "duration_wall_seconds": capture.get("duration_wall_seconds"),
                    "activity_sample_interval_seconds": capture.get("activity_sample_interval_seconds"),
                    "activity_sample_count": capture.get("activity_sample_count"),
                    "runtime": capture.get("runtime", {}),
                    "activity_summary": capture.get("activity_summary", {}),
                    "torch_memory": capture.get("torch_memory", {}),
                    "other_gpu_activity": capture.get("other_gpu_activity", {}),
                    "monitor_errors": capture.get("monitor_errors", []),
                }
        entry = comparisons.setdefault(key, {"family": record["family"], "scale": record["scale"], "variant": record["variant"], "seeds": []})
        entry["seeds"].append({
            "seed": record["seed"], "path": record.get("path", ""),
            "tokens_per_second": tokens_per_second,
            "training_joules_per_million_tokens": (
                training_energy.get("joules_per_million_training_tokens")
                if isinstance(training_energy, dict) else None
            ),
            "energy_supported": bool(energy.get("training", {}).get("supported", False)) if isinstance(energy, dict) else False,
            "resource_usage": resource_usage,
        })
    return comparisons


def _collect_availability_notes(root: Path, records: list[dict[str, Any]], exclusions: list[dict[str, str]]) -> dict[str, Any]:
    """Unavailable/unqualified reasons from records, profiling artifacts, and exclusions."""

    unavailable: list[dict[str, str]] = []
    unqualified: list[dict[str, str]] = []
    for record in records:
        metrics = record.get("metrics", {})
        recurrence = metrics.get("recurrence", {})
        if not isinstance(recurrence, dict):
            continue
        if recurrence.get("load_failure_reason"):
            unavailable.append({"path": record.get("path", ""), "reason": str(recurrence["load_failure_reason"])})
        if recurrence.get("requested_backend") == "cuda-fused" and not recurrence.get("qualification_hash"):
            unqualified.append({"path": record.get("path", ""), "reason": "cuda-fused record has no qualification hash"})
    profiling = _scan_profiling_artifacts(root)
    for pilot in profiling["pilots"]:
        if pilot.get("status") == "unreadable":
            unavailable.append({"path": pilot["path"], "reason": pilot.get("reason", "unreadable pilot artifact")})
    for bench in profiling["recurrence_benchmarks"]:
        if bench.get("status") == "unreadable":
            unavailable.append({"path": bench["path"], "reason": bench.get("reason", "unreadable benchmark artifact")})
    for item in exclusions:
        unavailable.append({"path": item["path"], "reason": item["reason"]})
    return {"unavailable": unavailable, "unqualified": unqualified}


def write_first_submission_reports(root: str | Path, *, expected_run_mode: str | None = None) -> dict[str, Any]:
    """Write raw seed tables and descriptive paired results; never significance stars."""

    root = Path(root)
    records, exclusions = rebuild_validated_records(root, expected_run_mode=expected_run_mode)
    metrics = (
        "final_validation.perplexity", "retrieval.exact_query_accuracy",
        "retrieval.latest_value_accuracy", "retrieval.stale_value_error_rate",
        "retrieval.unrelated_association_retention",
    )
    backend_identity = _collect_backend_identity(records)
    report = {
        "report_version": 1,
        "interpretation": "descriptive paired seed summaries only; no null-hypothesis significance tests or significance stars",
        "validated_records": records,
        "exclusions": exclusions,
        "summary": descriptive_summary(records, metrics),
        # Additive backend/profiling coverage for 7.4; existing keys are unchanged.
        "backend_identity": backend_identity,
        "qualification_evidence": {
            "hashes": backend_identity["qualification_hashes"],
            "profiling_artifacts": _scan_profiling_artifacts(root),
        },
        "performance_comparisons": _collect_performance_comparisons(records),
        "availability_notes": _collect_availability_notes(root, records, exclusions),
    }
    report["plots"] = generate_research_plots(root, records)
    atomic_write_json(root / "aggregate_report.json", report)
    lines = [
        "# First-submission experiment summary",
        "",
        "Results are raw seed values, means, standard deviations, and paired differences. They do not use significance stars.",
        "",
        "| Family | Scale | Variant/control | Seed values |",
        "| --- | --- | --- | --- |",
    ]
    for entry in report["summary"].values():
        if "seeds" not in entry:
            continue
        raw = "; ".join(f"seed {item['seed']}: {item['metrics']}" for item in entry["seeds"])
        lines.append(f"| {entry['family']} | {entry['scale']} | {entry['variant']} | {raw} |")
    if exclusions:
        lines.extend(["", "## Excluded artifacts"])
        lines.extend(f"- `{item['path']}`: {item['reason']}" for item in exclusions)
    lines.extend([
        "", "## Backend and qualification coverage",
        f"- Requested backends: {report['backend_identity']['requested_backends'] or 'none recorded'}",
        f"- Resolved backends: {report['backend_identity']['resolved_backends'] or 'none recorded'}",
        f"- Qualification hashes: {report['backend_identity']['qualification_hashes'] or 'none recorded'}",
        f"- Unavailable reasons: {len(report['availability_notes']['unavailable'])}; "
        f"unqualified reasons: {len(report['availability_notes']['unqualified'])}",
    ])
    (root / "first_submission_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def generate_research_plots(root: str | Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Create descriptive figures only; never annotate p-values or stars."""

    root = Path(root)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {"generated": [], "unavailable": "matplotlib is not installed"}
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    unavailable: list[str] = []

    # Quality and convergence.
    quality = [(f"{row['family']}\n{row['scale']}\n{row['variant']}", _metric_value(row["metrics"], "final_validation.perplexity")) for row in records]
    quality = [(label, value) for label, value in quality if value is not None]
    if quality:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(range(len(quality)), [value for _, value in quality])
        axes[0].set_xticks(range(len(quality)), [label for label, _ in quality], rotation=45, ha="right")
        axes[0].set_ylabel("Final validation perplexity")
        axes[0].set_title("Fixed-budget quality")
        for row in records:
            losses = row["metrics"].get("training", {}).get("train_losses", [])
            if losses:
                axes[1].plot(range(1, len(losses) + 1), losses, label=f"{row['family']}/{row['variant']}")
        axes[1].set_xlabel("Optimizer step")
        axes[1].set_ylabel("Training loss")
        axes[1].set_title("Convergence by fixed optimizer step")
        if axes[1].lines:
            axes[1].legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(plot_dir / "quality_and_convergence.png", dpi=150)
        plt.close(figure)
        generated.append("plots/quality_and_convergence.png")
    else:
        unavailable.append("quality_and_convergence: no completed quality metrics")

    # Prefill/decode throughput and cache memory scaling.
    efficiency_rows: list[tuple[str, int, float, float, int, float]] = []
    for row in records:
        for context, value in row["metrics"].get("benchmarks", {}).items():
            if not isinstance(value, dict) or "metadata" not in value:
                continue
            try:
                efficiency_rows.append((
                    row["variant"], int(value["metadata"]["prompt_length"]),
                    float(value["prefill"]["tokens_per_second"]), float(value["cached_decode"]["tokens_per_second"]),
                    int(value["prefill"]["cache_bytes"]), float(value["prefill"]["peak_memory_mb"]),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    if efficiency_rows:
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        for variant in sorted({item[0] for item in efficiency_rows}):
            values = sorted((item for item in efficiency_rows if item[0] == variant), key=lambda item: item[1])
            axes[0].plot([item[1] for item in values], [item[2] for item in values], marker="o", label=f"{variant} prefill")
            axes[0].plot([item[1] for item in values], [item[3] for item in values], marker="s", linestyle="--", label=f"{variant} cached decode")
            axes[1].plot([item[1] for item in values], [item[4] for item in values], marker="o", label=f"{variant} cache")
            axes[1].plot([item[1] for item in values], [item[5] for item in values], marker="s", linestyle="--", label=f"{variant} peak memory")
        axes[0].set(xlabel="Prompt length", ylabel="Tokens / second", title="Prefill vs cached decode")
        axes[1].set(xlabel="Prompt length", ylabel="Bytes / MB (separate units)", title="Cache and memory scaling")
        axes[0].legend(fontsize=7)
        axes[1].legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(plot_dir / "efficiency_and_cache_scaling.png", dpi=150)
        plt.close(figure)
        generated.append("plots/efficiency_and_cache_scaling.png")
    else:
        unavailable.append("efficiency_and_cache_scaling: no cache benchmarks")

    # Retrieval difficulty and overwrite-specific measures.
    retrieval_rows: list[tuple[str, str, float]] = []
    for row in records:
        retrieval = row["metrics"].get("retrieval", {})
        for length, item in retrieval.get("breakdowns", {}).get("sequence_length", {}).items():
            if isinstance(item, dict) and item.get("accuracy") is not None:
                retrieval_rows.append((row["variant"], length, float(item["accuracy"])))
    if retrieval_rows:
        figure, axis = plt.subplots(figsize=(6, 4))
        for variant in sorted({item[0] for item in retrieval_rows}):
            values = sorted((item for item in retrieval_rows if item[0] == variant), key=lambda item: int(item[1]))
            axis.plot([int(item[1]) for item in values], [item[2] for item in values], marker="o", label=variant)
        axis.set(xlabel="Sequence length", ylabel="Exact query accuracy", title="MQAR/overwrite retrieval difficulty")
        axis.legend()
        figure.tight_layout()
        figure.savefig(plot_dir / "retrieval_difficulty.png", dpi=150)
        plt.close(figure)
        generated.append("plots/retrieval_difficulty.png")
    else:
        unavailable.append("retrieval_difficulty: no completed retrieval metrics")

    # Raw paired quality differences, deliberately without significance marks.
    pairs: list[tuple[str, float]] = []
    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        grouped[(row["family"], row["scale"], row["seed"])][row["variant"]] = row
    for key, variants in grouped.items():
        scalar = _metric_value(variants.get("scalar", {}).get("metrics", {}), "final_validation.perplexity")
        for variant in ("channel-wise", "scalar-matched"):
            value = _metric_value(variants.get(variant, {}).get("metrics", {}), "final_validation.perplexity")
            if scalar is not None and value is not None:
                pairs.append((f"{key[0]}\n{variant}-scalar\nseed {key[2]}", value - scalar))
    if pairs:
        figure, axis = plt.subplots(figsize=(max(6, len(pairs) * 1.1), 4))
        axis.axhline(0, color="black", linewidth=0.8)
        axis.bar(range(len(pairs)), [value for _, value in pairs])
        axis.set_xticks(range(len(pairs)), [label for label, _ in pairs], rotation=45, ha="right")
        axis.set_ylabel("Final perplexity difference")
        axis.set_title("Raw paired seed differences (no significance testing)")
        figure.tight_layout()
        figure.savefig(plot_dir / "paired_seed_differences.png", dpi=150)
        plt.close(figure)
        generated.append("plots/paired_seed_differences.png")

    # Layer-wise mean decay gates.
    gate_rows: list[tuple[str, int, float]] = []
    for row in records:
        for layer, values in row["metrics"].get("gate_diagnostics", {}).get("layers", {}).items():
            if isinstance(values, dict) and values.get("mean") is not None:
                gate_rows.append((row["variant"], int(layer), float(values["mean"])))
    if gate_rows:
        figure, axis = plt.subplots(figsize=(6, 4))
        for variant in sorted({item[0] for item in gate_rows}):
            values = sorted((item for item in gate_rows if item[0] == variant), key=lambda item: item[1])
            axis.plot([item[1] for item in values], [item[2] for item in values], marker="o", label=variant)
        axis.set(xlabel="Layer", ylabel="Mean decay gate", title="Gate behavior")
        axis.legend()
        figure.tight_layout()
        figure.savefig(plot_dir / "gate_behavior.png", dpi=150)
        plt.close(figure)
        generated.append("plots/gate_behavior.png")

    energy_rows = []
    for row in records:
        value = _metric_value(row["metrics"], "energy.training.joules_per_million_training_tokens")
        if value is not None:
            energy_rows.append((f"{row['family']}/{row['variant']}", value))
    if energy_rows:
        figure, axis = plt.subplots(figsize=(max(6, len(energy_rows)), 4))
        axis.bar(range(len(energy_rows)), [value for _, value in energy_rows])
        axis.set_xticks(range(len(energy_rows)), [label for label, _ in energy_rows], rotation=45, ha="right")
        axis.set(ylabel="Joules / million training tokens", title="Energy efficiency (same-device only)")
        figure.tight_layout()
        figure.savefig(plot_dir / "energy_efficiency.png", dpi=150)
        plt.close(figure)
        generated.append("plots/energy_efficiency.png")
    else:
        unavailable.append("energy_efficiency: unsupported or no same-device energy readings")
    return {"generated": generated, "unavailable": unavailable}
