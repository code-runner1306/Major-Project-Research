"""Single offline CPU smoke command for all first-submission experiment paths."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from config import smoke_profile
from orchestration import run_cpu_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="unique smoke output directory")
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path("output") / f"smoke_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    print(run_cpu_smoke(smoke_profile(output)))


if __name__ == "__main__":
    main()
