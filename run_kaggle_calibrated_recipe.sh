#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
# Arguments are forwarded directly; do not put secrets on the command line.
exec timeout --signal=INT --kill-after=30s 41400s "${PYTHON_BIN:-python3}" run_recipe_study.py "$@"
