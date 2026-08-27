"""Entry point for the hardened experiment grid."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from config import production_profile, smoke_profile
from orchestration import run_cpu_smoke, run_selected
from pilot import PilotSettings, run_v100_pilot


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
    parser.add_argument("--matched-control", action="store_true", help="add exactly parameter-matched expanded-FFN scalar control")
    parser.add_argument("--no-cache-benchmarks", action="store_true")
    parser.add_argument("--no-gate-diagnostics", action="store_true")
    parser.add_argument("--no-energy", action="store_true")
    parser.add_argument("--recurrence-backend", choices=("reference", "cuda-fused"))
    parser.add_argument("--state-accumulator-dtype", choices=("input", "float32"), default="input")
    parser.add_argument("--qualification-artifact")
    parser.add_argument("--pilot", action="store_true", help="measure real V100 training work without unlocking production training")
    parser.add_argument("--pilot-scales", default="55M,100M", help="comma-separated production scales for --pilot")
    parser.add_argument("--pilot-warmup", type=int, default=25)
    parser.add_argument("--pilot-steps", type=int, default=200)
    parser.add_argument("--pilot-data-fed", action="store_true", help="also sample TinyStories input fetch outside timed GPU windows")
    parser.add_argument("--pilot-trace", action="store_true", help="capture one Chrome profiler trace outside measured pilot steps")
    args = parser.parse_args()
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
    profile = replace(
        profile,
        enable_cache_benchmarks=profile.enable_cache_benchmarks and not args.no_cache_benchmarks,
        enable_gate_diagnostics=profile.enable_gate_diagnostics and not args.no_gate_diagnostics,
        enable_energy_collection=profile.enable_energy_collection and not args.no_energy,
    )
    if args.pilot:
        scales = tuple(scale.strip() for scale in args.pilot_scales.split(",") if scale.strip())
        try:
            result = run_v100_pilot(profile, PilotSettings(
                scales=scales, warmup_steps=args.pilot_warmup, measured_steps=args.pilot_steps,
                data_fed=args.pilot_data_fed, collect_trace=args.pilot_trace,
            ))
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    elif args.mode == "smoke":
        result = run_cpu_smoke(profile)
    else:
        families = None if args.family == "all" else (args.family,)
        result = run_selected(profile, families)
    print(f"Completed: {profile.output_dir}")
    print(result.get("smoke", {"status": "complete"}))


if __name__ == "__main__":
    main()
