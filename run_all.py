"""Entry point for the hardened experiment grid."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from config import production_profile, smoke_profile
from orchestration import run_cpu_smoke, run_selected


def build_profile(mode: str, output: str | None, matched: bool):
    if mode == "smoke":
        root = Path(output) if output else Path("output") / f"smoke_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        return smoke_profile(root)
    profile = production_profile(Path(output) if output else "output")
    return replace(profile, enable_parameter_matched_scalar_control=matched or profile.enable_parameter_matched_scalar_control)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scalar/channel-wise Gated DeltaNet experiments")
    parser.add_argument("--mode", choices=("production", "smoke"), default="production")
    parser.add_argument("--output", help="isolated output root; smoke defaults to a unique directory")
    parser.add_argument("--family", choices=("all", "tinystories", "mqar", "overwrite"), default="all")
    parser.add_argument("--matched-control", action="store_true", help="add exactly parameter-matched expanded-FFN scalar control")
    parser.add_argument("--no-cache-benchmarks", action="store_true")
    parser.add_argument("--no-gate-diagnostics", action="store_true")
    parser.add_argument("--no-energy", action="store_true")
    args = parser.parse_args()
    profile = build_profile(args.mode, args.output, args.matched_control)
    profile = replace(
        profile,
        enable_cache_benchmarks=profile.enable_cache_benchmarks and not args.no_cache_benchmarks,
        enable_gate_diagnostics=profile.enable_gate_diagnostics and not args.no_gate_diagnostics,
        enable_energy_collection=profile.enable_energy_collection and not args.no_energy,
    )
    if args.mode == "smoke":
        result = run_cpu_smoke(profile)
    else:
        families = None if args.family == "all" else (args.family,)
        result = run_selected(profile, families)
    print(f"Completed: {profile.output_dir}")
    print(result.get("smoke", {"status": "complete"}))


if __name__ == "__main__":
    main()
