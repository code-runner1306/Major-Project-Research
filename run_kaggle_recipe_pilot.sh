#!/usr/bin/env bash
# Numerical/performance qualification only. Does not start the full study.
set -Eeuo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
SESSION_REMAINING_SECONDS="${SESSION_REMAINING_SECONDS:-6000}"
[[ "$SESSION_REMAINING_SECONDS" =~ ^[0-9]+$ ]] && (( SESSION_REMAINING_SECONDS > 900 && SESSION_REMAINING_SECONDS <= 41400 )) || { echo 'Invalid remaining notebook time' >&2; exit 2; }
PILOT_TIMEOUT_SECONDS=$(( SESSION_REMAINING_SECONDS < 6000 ? SESSION_REMAINING_SECONDS : 6000 ))
if [[ "${PILOT_WATCHDOG_ACTIVE:-0}" != 1 ]]; then
  exec timeout --signal=INT --kill-after=30s "${PILOT_TIMEOUT_SECONDS}s" env PILOT_WATCHDOG_ACTIVE=1 bash "$0"
fi
PILOT_WORK_SECONDS=$(( PILOT_TIMEOUT_SECONDS - 600 < 5400 ? PILOT_TIMEOUT_SECONDS - 600 : 5400 ))
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/recipe_pilot_$(date -u +%Y%m%dT%H%M%SZ)}"
command=("$PYTHON_BIN" benchmark_kaggle_recipe.py --output "$OUTPUT_ROOT"
  --steps "${PILOT_STEPS:-50}" --warmup "${PILOT_WARMUP:-5}"
  --batch "${MICROBATCH:-32}" --accumulation "${GRAD_ACCUMULATION:-16}" --chunk-size "${CHUNK_SIZE:-16}"
  --max-seconds "$PILOT_WORK_SECONDS")
[[ "${CPU_SMOKE:-0}" == 1 ]] && command+=(--cpu-smoke)
# Separate watchdog covers numerical checks and driver stalls, too.
RC=0
"${command[@]}" || RC=$?
if [[ "${GIT_PUSH:-0}" == 1 && -d "$OUTPUT_ROOT" ]]; then
  "$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY' || RC=1
import pathlib,sys
from kaggle_session import export_evidence,publish,remote_url
root=pathlib.Path(sys.argv[1])
branch='diagnostic/recipe-pilot/'+root.name
publish(export_evidence(root),remote_url('origin'),branch)
print('Published results branch:',branch,flush=True)
PY
fi
exit "$RC"
