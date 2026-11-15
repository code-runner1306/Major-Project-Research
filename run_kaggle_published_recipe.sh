#!/usr/bin/env bash
# Real-task timing first; development requires matching timing evidence.
set -Eeuo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
SESSION_REMAINING_SECONDS="${SESSION_REMAINING_SECONDS:-41400}"
[[ "$SESSION_REMAINING_SECONDS" =~ ^[0-9]+$ ]] && (( SESSION_REMAINING_SECONDS > 900 && SESSION_REMAINING_SECONDS <= 41400 )) || { echo 'Invalid remaining session budget' >&2; exit 2; }
if [[ "${RECIPE_WATCHDOG_ACTIVE:-0}" != 1 ]]; then
  exec timeout --signal=INT --kill-after=30s "${SESSION_REMAINING_SECONDS}s" env RECIPE_WATCHDOG_ACTIVE=1 bash "$0"
fi
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-39600}"
[[ "$TIME_BUDGET_SECONDS" =~ ^[0-9]+$ ]] && (( TIME_BUDGET_SECONDS > 900 && TIME_BUDGET_SECONDS <= 39600 && TIME_BUDGET_SECONDS + 300 <= SESSION_REMAINING_SECONDS )) || { echo 'Workload budget must fit remaining notebook time with publication reserve' >&2; exit 2; }
[[ "${FAMILY:-}" == mqar || "${FAMILY:-}" == overwrite ]] || { echo 'Set FAMILY=mqar or FAMILY=overwrite' >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${MODE:-timing}"
if [[ "$MODE" == development && -z "${EPOCHS:-}" ]]; then
  echo 'Choose EPOCHS explicitly after reviewing both families timing results' >&2
  exit 2
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/published_${FAMILY}_${MODE}_$(date -u +%Y%m%dT%H%M%SZ)}"
command=("$PYTHON_BIN" run_published_recipe.py --family "$FAMILY" --mode "$MODE"
  --output "$OUTPUT_ROOT" --backend "${BACKEND:-reference}" --steps "${TIMING_STEPS:-50}"
  --epochs "${EPOCHS:-64}" --learning-rate "${LEARNING_RATE:-0.0021544346900318843}"
  --microbatch "${MICROBATCH:-32}" --effective-batch "${EFFECTIVE_BATCH:-512}"
  --max-seconds "$TIME_BUDGET_SECONDS"
  --validation-examples "${VALIDATION_EXAMPLES:-512}" --seed "${SEED:-42}")
[[ -n "${QUALIFICATION_ROOT:-}" ]] && command+=(--qualification "$QUALIFICATION_ROOT")
[[ -n "${TIMING_EVIDENCE_ROOT:-}" ]] && command+=(--timing-evidence "$TIMING_EVIDENCE_ROOT")
[[ "${CPU_SMOKE:-0}" == 1 ]] && command+=(--cpu-smoke)
RC=0
"${command[@]}" || RC=$?
if [[ "${GIT_PUSH:-0}" == 1 && -d "$OUTPUT_ROOT" ]]; then
  "$PYTHON_BIN" - "$OUTPUT_ROOT" "$FAMILY" "$MODE" <<'PY' || RC=1
import pathlib,sys
from kaggle_session import export_evidence,publish,remote_url
root=pathlib.Path(sys.argv[1])
branch=f'diagnostic/published-recipe/{sys.argv[2]}/{sys.argv[3]}/{root.name}'
publish(export_evidence(root),remote_url('origin'),branch)
print('Published results branch:',branch,flush=True)
PY
fi
exit "$RC"
