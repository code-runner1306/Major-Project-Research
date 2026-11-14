#!/usr/bin/env bash
# Diagnostic only: varied-data learnability on both V4 tasks. TRAIN_EXAMPLES=128 reproduces the fixed-data setup.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STEPS="${STEPS:-200}"
EVAL_EVERY="${EVAL_EVERY:-50}"
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-10000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working/varied_learnability_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$STEPS" =~ ^[0-9]+$ && "$EVAL_EVERY" =~ ^[0-9]+$ && "$TRAIN_EXAMPLES" =~ ^[0-9]+$ ]] && ((STEPS >= 10 && EVAL_EVERY >= 1 && TRAIN_EXAMPLES >= 8)) || { echo 'Invalid STEPS, EVAL_EVERY or TRAIN_EXAMPLES' >&2; exit 2; }
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Output already exists: $OUTPUT_ROOT" >&2; exit 2; }
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.device_count() >= 2, "Two CUDA GPUs required"; print("GPUs:", [torch.cuda.get_device_name(i) for i in range(2)])'
mkdir -p "$OUTPUT_ROOT"
printf 'Output: %s\n' "$OUTPUT_ROOT"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" fixed_learnability.py --family mqar --steps "$STEPS" --eval-every "$EVAL_EVERY" --train-examples "$TRAIN_EXAMPLES" --output "$OUTPUT_ROOT/mqar" > "$OUTPUT_ROOT/mqar.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" fixed_learnability.py --family overwrite --steps "$STEPS" --eval-every "$EVAL_EVERY" --train-examples "$TRAIN_EXAMPLES" --output "$OUTPUT_ROOT/overwrite" > "$OUTPUT_ROOT/overwrite.log" 2>&1 &
P1=$!
trap 'kill "$P0" "$P1" 2>/dev/null || true' INT TERM
RC=0
wait "$P0" || { echo 'MQAR diagnostic failed; see mqar.log' >&2; RC=1; }
wait "$P1" || { echo 'Overwrite diagnostic failed; see overwrite.log' >&2; RC=1; }
"$PYTHON_BIN" - "$OUTPUT_ROOT" "$RC" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
results = {}
for family in ('mqar', 'overwrite'):
    path = root / family / 'diagnostic_summary.json'
    results[family] = json.loads(path.read_text()) if path.is_file() else {'status': 'failed_or_incomplete'}
(root / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2), flush=True)
PY
exit "$RC"
