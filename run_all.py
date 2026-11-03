"""Entry point for the hardened experiment grid."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from artifacts import atomic_write_json
from config import production_profile, smoke_profile
from orchestration import run_cpu_smoke, run_selected
from pilot import PilotSettings, run_v100_pilot


def _parse_csv_values(value: str, option: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(","))
    if not values or any(not item for item in values):
        raise ValueError(f"{option} expects a comma-separated list without empty values")
    if len(set(values)) != len(values):
        raise ValueError(f"{option} values must be unique")
    return values


def _parse_seed_counts(value: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in value.split(","):
        try:
            scale, raw_count = item.split(":", maxsplit=1)
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError("--estimate-seeds format is SCALE:COUNT,..., for example 10M:5,30M:5") from exc
        if scale not in {"10M", "30M", "55M", "100M"} or count < 1 or scale in counts:
            raise ValueError(f"Invalid or duplicate estimated seed count: {item!r}")
        counts[scale] = count
    return counts


def build_profile(
    mode: str, output: str | None, matched: bool, *, recurrence_backend: str | None,
    state_accumulator_dtype: str, qualification_artifact: str | None,
):
    if mode == "smoke":
        root = Path(output) if output else Path("output") / f"smoke_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        profile = smoke_profile(root)
        if recurrence_backend not in (None, "reference"):
            raise ValueError("Smoke mode supports only the portable reference recurrence backend")
        return profile
    profile = production_profile(Path(output) if output else "output")
    if recurrence_backend is None:
        raise ValueError("Production requires --recurrence-backend {reference,cuda-fused}; use --pilot for measurement only")
    return replace(
        profile,
        enable_parameter_matched_scalar_control=matched or profile.enable_parameter_matched_scalar_control,
        recurrence_backend=recurrence_backend,
        state_accumulator_dtype=state_accumulator_dtype,
        recurrence_qualification_artifact=Path(qualification_artifact) if qualification_artifact else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scalar/channel-wise Gated DeltaNet experiments")
    parser.add_argument("--mode", choices=("production", "smoke"), default="production")
    parser.add_argument("--output", help="isolated output root; smoke defaults to a unique directory")
    parser.add_argument("--family", choices=("all", "tinystories", "mqar", "overwrite"), default="all")
    parser.add_argument("--models", help="comma-separated model scales to run, e.g. 10M,30M")
    parser.add_argument("--seeds", help="comma-separated seeds to run, e.g. 42,43")
    parser.add_argument(
        "--variant", "--variants", "--varient", dest="variant",
        help="comma-separated model variants to run, e.g. scalar,channelwise",
    )
    parser.add_argument(
        "--max-optimizer-steps", type=int,
        help="fixed production training budget; required when the profile has no step budget",
    )
    parser.add_argument("--matched-control", action="store_true", help="add exactly parameter-matched expanded-FFN scalar control")
    parser.add_argument("--full-prefix-benchmark", action="store_true", help="include the slow no-cache generation benchmark")
    parser.add_argument("--no-cache-benchmarks", action="store_true")
    parser.add_argument("--no-gate-diagnostics", action="store_true")
    parser.add_argument("--no-posthoc-gate-grouping", action="store_true", help="skip post-training gate-sharing diagnostic evaluations")
    parser.add_argument("--no-energy", action="store_true")
    parser.add_argument("--recurrence-backend", choices=("reference", "cuda-fused"))
    parser.add_argument("--state-accumulator-dtype", choices=("input", "float32"), default="input")
    parser.add_argument("--qualification-artifact")
    execution_group = parser.add_mutually_exclusive_group()
    execution_group.add_argument(
        "--train-only", action="store_true",
        help="train selected runs to the trained lifecycle state without evaluation",
    )
    execution_group.add_argument(
        "--evaluate-only", action="store_true",
        help="evaluate compatible trained checkpoints without starting training",
    )
    parser.add_argument("--skip-report", action="store_true", help="skip aggregate report generation for isolated worker jobs")
    parser.add_argument(
        "--concurrent-training-runs", type=int, default=1,
        help="record the launcher concurrency limit in run provenance (default: 1)",
    )
    parser.add_argument("--pilot", action="store_true", help="measure real V100 training work without unlocking production training")
    parser.add_argument("--pilot-scales", default="55M,100M", help="comma-separated production scales for --pilot")
    parser.add_argument("--pilot-warmup", type=int, default=25)
    parser.add_argument("--pilot-steps", type=int, default=200)
    parser.add_argument(
        "--pilot-data-fed", dest="pilot_data_fed", action="store_true", default=True,
        help="sample input fetch/generation for TinyStories, MQAR, and overwrite (default)",
    )
    parser.add_argument(
        "--pilot-no-data-fed", dest="pilot_data_fed", action="store_false",
        help="skip dataset-fed timing and use resident-compute proxies",
    )
    parser.add_argument("--pilot-trace", action="store_true", help="capture one Chrome profiler trace outside measured pilot steps")
    parser.add_argument("--pilot-parallel-scales", default="10M,30M", help="low-tier scales for bounded concurrency qualification")
    parser.add_argument("--pilot-parallel-workers", default="2,3", help="comma-separated concurrent worker counts to try")
    parser.add_argument(
        "--pilot-require-parallel-workers", type=int,
        help="qualify this exact count for every selected scale/dataset before parallel training",
    )
    parser.add_argument("--pilot-no-parallel-check", action="store_true", help="skip parallel capacity/throughput qualification")
    parser.add_argument(
        "--estimate-seeds", default="10M:5,30M:5,55M:1,100M:1",
        help="seed-count plan included in the pilot time estimate, e.g. 10M:5,30M:5,55M:1,100M:1",
    )
    args = parser.parse_args()
    if args.concurrent_training_runs < 1:
        parser.error("--concurrent-training-runs must be at least 1")
    if args.pilot_require_parallel_workers is not None:
        if args.pilot_require_parallel_workers < 2:
            parser.error("--pilot-require-parallel-workers must be at least 2")
        if not args.pilot:
            parser.error("--pilot-require-parallel-workers requires --pilot")
    if args.pilot and args.mode != "production":
        parser.error("--pilot requires --mode production and the college CUDA machine")
    if args.pilot and args.recurrence_backend is None:
        args.recurrence_backend = "reference"
    try:
        profile = build_profile(
            args.mode, args.output, args.matched_control,
            recurrence_backend=args.recurrence_backend,
            state_accumulator_dtype=args.state_accumulator_dtype,
            qualification_artifact=args.qualification_artifact,
        )
    except ValueError as exc:
        parser.error(str(exc))
    profile_overrides = {}
    try:
        if args.models is not None:
            models = _parse_csv_values(args.models, "--models")
            unknown_models = tuple(model for model in models if model not in profile.models)
            if unknown_models:
                raise ValueError(
                    f"Unknown model scale(s) {', '.join(unknown_models)}; "
                    f"choose from {', '.join(profile.models)}"
                )
            profile_overrides["scales"] = models
        if args.seeds is not None:
            seed_strings = _parse_csv_values(args.seeds, "--seeds")
            try:
                profile_overrides["seeds"] = tuple(int(seed) for seed in seed_strings)
            except ValueError as exc:
                raise ValueError("--seeds values must be integers") from exc
        if args.variant is not None:
            variants = _parse_csv_values(args.variant, "--variant")
            unknown_variants = tuple(variant for variant in variants if variant not in profile.variants)
            if unknown_variants:
                raise ValueError(
                    f"Unknown variant(s) {', '.join(unknown_variants)}; "
                    f"choose from {', '.join(profile.variants)}"
                )
            profile_overrides["selected_variants"] = variants
        if args.max_optimizer_steps is not None:
            if args.max_optimizer_steps <= 0:
                raise ValueError("--max-optimizer-steps must be a positive integer")
            profile_overrides["budget"] = replace(
                profile_overrides.get("budget", profile.budget),
                max_optimizer_steps=args.max_optimizer_steps,
            )
        profile_overrides["concurrent_training_runs"] = args.concurrent_training_runs
    except ValueError as exc:
        parser.error(str(exc))
    profile = replace(
        profile,
        **profile_overrides,
        enable_full_prefix_benchmark=args.full_prefix_benchmark,
        enable_cache_benchmarks=profile.enable_cache_benchmarks and not args.no_cache_benchmarks,
        enable_gate_diagnostics=profile.enable_gate_diagnostics and not args.no_gate_diagnostics,
        enable_posthoc_gate_grouping=profile.enable_posthoc_gate_grouping and not args.no_posthoc_gate_grouping,
        enable_energy_collection=profile.enable_energy_collection and not args.no_energy,
    )
    if args.pilot:
        scales = tuple(scale.strip() for scale in args.pilot_scales.split(",") if scale.strip())
        try:
            estimate_seed_counts = _parse_seed_counts(args.estimate_seeds)
            parallel_scales = (
                tuple(scale.strip() for scale in args.pilot_parallel_scales.split(",") if scale.strip())
                if not args.pilot_no_parallel_check else ()
            )
            parallel_workers = tuple(int(worker.strip()) for worker in args.pilot_parallel_workers.split(",") if worker.strip())
            if args.pilot_require_parallel_workers is not None:
                parallel_workers = (args.pilot_require_parallel_workers,)
                if not set(scales).issubset(parallel_scales):
                    raise ValueError("Parallel worker qualification must include every selected pilot scale")
            result = run_v100_pilot(profile, PilotSettings(
                scales=scales, variants=profile.variants,
                warmup_steps=args.pilot_warmup, measured_steps=args.pilot_steps,
                data_fed=args.pilot_data_fed, collect_trace=args.pilot_trace,
                planned_optimizer_steps=profile.budget.max_optimizer_steps or 1_000,
                seed_counts=estimate_seed_counts,
                parallel_scales=tuple(scale for scale in parallel_scales if scale in scales),
                parallel_worker_counts=parallel_workers,
            ))
            if args.pilot_require_parallel_workers is not None:
                required_count = str(args.pilot_require_parallel_workers)
                statuses = result["parallel_worker_statuses"]
                failures = [
                    f"{scale}/{family}: {statuses.get(scale, {}).get(family, {}).get(required_count, 'missing')}"
                    for scale in scales
                    for family in ("tinystories", "mqar", "overwrite")
                    if statuses.get(scale, {}).get(family, {}).get(required_count)
                    not in {"complete", "skipped_vram_capacity"}
                ]
                if failures:
                    raise RuntimeError("Parallel worker qualification failed: " + "; ".join(failures))
                atomic_write_json(profile.output_dir / "parallel_worker_qualification.json", {
                    "requested_workers": args.pilot_require_parallel_workers,
                    "scale_dataset_statuses": {
                        scale: {
                            family: statuses.get(scale, {}).get(family, {}).get(required_count, "missing")
                            for family in ("tinystories", "mqar", "overwrite")
                        }
                        for scale in scales
                    },
                    "pilot_output_dir": result["output_dir"],
                    "capacity_skips_fall_back_to_sequential": True,
                })
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    elif args.mode == "smoke":
        result = run_cpu_smoke(profile)
    else:
        families = None if args.family == "all" else (args.family,)
        result = run_selected(
            profile, families,
            stop_after_trained=args.train_only,
            evaluate_only=args.evaluate_only,
            write_report=not args.skip_report,
        )
    print(f"Completed: {profile.output_dir}")
    print(result if args.pilot else result.get("smoke", {"status": "complete"}))


if __name__ == "__main__":
    main()
