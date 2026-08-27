"""Dedicated MQAR command entry point."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from config import production_profile, smoke_profile
from orchestration import run_family


def main() -> None:
    parser = argparse.ArgumentParser(description="Run separately trained deterministic MQAR")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--matched-control", action="store_true")
    parser.add_argument("--recurrence-backend", choices=("reference", "cuda-fused"))
    parser.add_argument("--state-accumulator-dtype", choices=("input", "float32"), default="input")
    parser.add_argument("--qualification-artifact")
    args = parser.parse_args()
    if not args.smoke and args.recurrence_backend is None:
        parser.error("Production requires --recurrence-backend {reference,cuda-fused}")
    profile = smoke_profile(args.output or "output/mqar_smoke") if args.smoke else production_profile(args.output or "output")
    profile = replace(
        profile, enable_parameter_matched_scalar_control=args.matched_control or profile.enable_parameter_matched_scalar_control,
        recurrence_backend=args.recurrence_backend or "reference",
        state_accumulator_dtype=args.state_accumulator_dtype,
        recurrence_qualification_artifact=Path(args.qualification_artifact) if args.qualification_artifact else None,
    )
    print(run_family(profile, "mqar"))


if __name__ == "__main__":
    main()
