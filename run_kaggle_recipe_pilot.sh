#!/usr/bin/env bash
# Numerical/performance qualification only. Does not start the full study.
set -Eeuo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/recipe_pilot_$(date -u +%Y%m%dT%H%M%SZ)}"
command=("$PYTHON_BIN" benchmark_kaggle_recipe.py --output "$OUTPUT_ROOT"
  --steps "${PILOT_STEPS:-50}" --warmup "${PILOT_WARMUP:-5}"
  --batch "${MICROBATCH:-32}" --chunk-size "${CHUNK_SIZE:-16}"
  --max-seconds 5400)
[[ "${CPU_SMOKE:-0}" == 1 ]] && command+=(--cpu-smoke)
# Separate watchdog covers numerical checks and driver stalls, too.
RC=0
timeout --signal=INT --kill-after=30s 6000s "${command[@]}" || RC=$?
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
