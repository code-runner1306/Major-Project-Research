#!/usr/bin/env bash
# Two selected research models, one process per T4. See KAGGLE_SESSIONS.md.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ "${1:-}" == "--smoke-test" ]]; then
  shift
  exec bash "$ROOT/run_kaggle_v4_smoke.sh" "$@"
fi
exec "${PYTHON_BIN:-python3}" "$ROOT/kaggle_session.py" "$@"
