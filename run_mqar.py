"""Dedicated MQAR command entry point."""

from __future__ import annotations

import argparse
from dataclasses import replace

from config import production_profile, smoke_profile
from orchestration import run_family


def main() -> None:
    parser = argparse.ArgumentParser(description="Run separately trained deterministic MQAR")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--matched-control", action="store_true")
    args = parser.parse_args()
    profile = smoke_profile(args.output or "output/mqar_smoke") if args.smoke else production_profile(args.output or "output")
    profile = replace(profile, enable_parameter_matched_scalar_control=args.matched_control or profile.enable_parameter_matched_scalar_control)
    print(run_family(profile, "mqar"))


if __name__ == "__main__":
    main()
