#!/usr/bin/env bash
# Family-specific 10M exploratory run. Requires Bash, Python, PyTorch CUDA, and this repository.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash run_48h_exploratory_study.sh --family mqar|overwrite [options]

Options:
  --python PATH         Python executable (default: python3)
  --output-root PATH    Output directory (default is family-specific under output/)
  --study-config PATH   Study config (default: study_configs/granularity_init_study.json)
  --dry-run             Print stages and planned tasks without training
  --evaluate-only       Evaluate existing trained checkpoints only
  -h, --help            Show this help
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FAMILY=""
PYTHON_BIN="${PYTHON:-python3}"
OUTPUT_ROOT=""
STUDY_CONFIG="study_configs/granularity_init_study.json"
DRY_RUN=0
EVALUATE_ONLY=0

while (($#)); do
  case "$1" in
    --family)
      (($# >= 2)) || die "--family requires mqar or overwrite"
      FAMILY="$2"; shift 2 ;;
    --python)
      (($# >= 2)) || die "--python requires an executable path"
      PYTHON_BIN="$2"; shift 2 ;;
    --output-root)
      (($# >= 2)) || die "--output-root requires a path"
      OUTPUT_ROOT="$2"; shift 2 ;;
    --study-config)
      (($# >= 2)) || die "--study-config requires a path"
      STUDY_CONFIG="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --evaluate-only) EVALUATE_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1 (use --help)" ;;
  esac
done

[[ -n "$FAMILY" ]] || die "--family is required"
case "$FAMILY" in
  mqar|overwrite) ;;
  *) die "--family must be exactly 'mqar' or 'overwrite'" ;;
esac

cd -- "$REPO_DIR"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  die "Python executable not found or not executable: $PYTHON_BIN"
fi
if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="output/granularity_48h_exploratory_lr3e-3_${FAMILY}"
fi
if [[ "$OUTPUT_ROOT" = /* ]]; then
  ROOT="$OUTPUT_ROOT"
else
  ROOT="$REPO_DIR/$OUTPUT_ROOT"
fi
if [[ "$STUDY_CONFIG" = /* ]]; then
  STUDY_CONFIG_PATH="$STUDY_CONFIG"
else
  STUDY_CONFIG_PATH="$REPO_DIR/$STUDY_CONFIG"
fi
[[ -f "$STUDY_CONFIG_PATH" ]] || die "Study config not found: $STUDY_CONFIG_PATH"

CONDITIONS=(
  'C1-scalar-default|scalar|default'
  'C3-channel-default|channel-wise|default'
  'C4-scalar-calibrated|scalar|calibrated-geometric'
  'C6-channel-calibrated|channel-wise|calibrated-geometric'
)
SCALE="10M"
SEED=42
STEP_BUDGET=1000
LEARNING_RATE="0.003"
EXPERIMENT_ID="granularity-init-48h-exploratory-lr3e-3-v1"

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() {
  local line="[$(timestamp)] $*"
  printf '%s\n' "$line"
  if ((DRY_RUN == 0)); then printf '%s\n' "$line" >> "$MASTER_LOG"; fi
}

task_paths() {
  local condition_id="$1" operation="$2"
  TASK_TAG="${FAMILY}_${SCALE}_${condition_id}_seed_${SEED}_${operation}"
  TASK_CONFIG="$ROOT/task_configs/${TASK_TAG}.json"
  TASK_LOG="$ROOT/logs/${TASK_TAG}.log"
  RUN_DIR="$ROOT/runs/$FAMILY/$SCALE/$condition_id/seed_$SEED"
}

write_task_config() {
  local condition_id="$1" variant="$2" initialization="$3" operation="$4"
  task_paths "$condition_id" "$operation"
  mkdir -p -- "$ROOT/task_configs"
  TASK_CONFIG="$TASK_CONFIG" STUDY_CONFIG_PATH="$STUDY_CONFIG_PATH" FAMILY="$FAMILY" \
    CONDITION_ID="$condition_id" VARIANT="$variant" INITIALIZATION="$initialization" \
    RUN_DIR="$RUN_DIR" SCALE="$SCALE" SEED="$SEED" STEP_BUDGET="$STEP_BUDGET" \
    LEARNING_RATE="$LEARNING_RATE" OPERATION="$operation" \
    "$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path

study = json.loads(Path(os.environ["STUDY_CONFIG_PATH"]).read_text(encoding="utf-8-sig"))
budget = study["budgets"][os.environ["SCALE"]]
retrieval = dict(study["retrieval"])
retrieval["validation_examples"] = 512
retrieval["held_out_examples"] = 512
task = {
    "operation": os.environ["OPERATION"],
    "experiment_id": "granularity-init-48h-exploratory-lr3e-3-v1",
    "condition_id": os.environ["CONDITION_ID"],
    "family": os.environ["FAMILY"],
    "scale": os.environ["SCALE"],
    "variant": os.environ["VARIANT"],
    "initialization": os.environ["INITIALIZATION"],
    "seed": int(os.environ["SEED"]),
    "output_dir": str(Path(os.environ["RUN_DIR"]).resolve()),
    "max_optimizer_steps": int(os.environ["STEP_BUDGET"]),
    "batch_size": int(budget["batch_size"]),
    "grad_accum_steps": int(budget["grad_accum_steps"]),
    "context_length": int(budget["context_length"]),
    "validation_steps": 100,
    "evaluation_interval": 100,
    "learning_rate": float(os.environ["LEARNING_RATE"]),
    "calibration_min_half_life": int(study["calibration"]["min_half_life"]),
    "calibration_max_half_life": int(study["calibration"]["max_half_life"]),
    "calibration_examples": int(study["calibration"]["examples"]),
    "retrieval_config": retrieval,
    "run_mode": "production",
    "recurrence_backend": "reference",
    "state_accumulator_dtype": "input",
    "recurrence_chunk_size": 64,
    "validation_answer_accuracy": True,
    "concurrent_training_runs": 1,
}
Path(os.environ["TASK_CONFIG"]).write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
PY
}

task_complete() {
  local task_config="$1" operation="$2" answer
  answer="$("$PYTHON_BIN" - "$task_config" "$operation" <<'PY'
import json, sys
from study_runner import task_complete
with open(sys.argv[1], encoding="utf-8-sig") as stream:
    task = json.load(stream)
print("COMPLETE" if task_complete(task, sys.argv[2]) else "INCOMPLETE")
PY
  )" || return 2
  answer="$(printf '%s\n' "$answer" | tail -n 1 | tr -d '\r')"
  [[ "$answer" == "COMPLETE" ]]
}

run_task() {
  local condition_id="$1" variant="$2" initialization="$3" operation="$4" rc
  write_task_config "$condition_id" "$variant" "$initialization" "$operation"
  if ((DRY_RUN)); then
    log "DRY RUN $operation $FAMILY/$SCALE/$condition_id/seed$SEED; config=$TASK_CONFIG"
    return 0
  fi
  mkdir -p -- "$ROOT/logs"
  log "START $operation $FAMILY/$SCALE/$condition_id/seed$SEED; log=$TASK_LOG"
  set +e
  "$PYTHON_BIN" run_all.py --study-task-config "$TASK_CONFIG" 2>&1 | tee -a "$TASK_LOG" | tee -a "$MASTER_LOG"
  rc=${PIPESTATUS[0]}
  set -e
  if ((rc != 0)); then
    log "FAILED rc=$rc task=$TASK_TAG"
    die "Study task failed: $TASK_TAG (exit $rc). See $TASK_LOG"
  fi
  if ! task_complete "$TASK_CONFIG" "$operation"; then
    die "Task returned success but its checkpoint/evaluation did not pass task_complete validation: $TASK_TAG"
  fi
  log "COMPLETED $TASK_TAG"
}

printf 'Exploratory study: 4 10M runs; family=%s; seed=%s; steps=%s; LR=%s; workers=1\n' \
  "$FAMILY" "$SEED" "$STEP_BUDGET" "$LEARNING_RATE"
printf 'Output root: %s\n' "$ROOT"
printf 'Default root is family-specific; provide --output-root for writable storage (for example /kaggle/working/...).\n'
printf 'Old-timing baseline: about 9 h 20 min training only for this family. It excludes validation every 100 steps, calibration, and evaluation; finishing within 48 hours is not guaranteed.\n'
printf 'Validation and held-out evaluation use 512 examples each; training remains at 50,000 examples.\n'
printf 'LR 3e-3 is a fixed literature-informed choice and is not tuned on these runs.\n'

if ((DRY_RUN)); then
  printf 'PLAN train all conditions for %s, then evaluate all conditions for %s\n' "$FAMILY" "$FAMILY"
  for condition in "${CONDITIONS[@]}"; do
    IFS='|' read -r condition_id variant initialization <<< "$condition"
    task_paths "$condition_id" train
    printf '  TRAIN %s %s %s seed=%s steps=%s config=%s\n' "$FAMILY" "$variant" "$initialization" "$SEED" "$STEP_BUDGET" "$TASK_CONFIG"
  done
  for condition in "${CONDITIONS[@]}"; do
    IFS='|' read -r condition_id variant initialization <<< "$condition"
    task_paths "$condition_id" evaluate
    printf '  EVALUATE %s %s %s seed=%s config=%s\n' "$FAMILY" "$variant" "$initialization" "$SEED" "$TASK_CONFIG"
  done
  exit 0
fi

mkdir -p -- "$ROOT/logs" "$ROOT/task_configs"
MASTER_LOG="$ROOT/logs/run_48h_$(date -u '+%Y%m%d_%H%M%S').log"
: > "$MASTER_LOG"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
log "Python/PyTorch/CUDA preflight"
preflight_output="$("$PYTHON_BIN" -c 'import sys, torch, torch.nn; print("Python:", sys.executable); print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 3)' 2>&1)" \
  || die "Python/PyTorch CUDA preflight failed; no model run was started: $preflight_output"
log "$preflight_output"

if ((EVALUATE_ONLY == 0)); then
  log "Training all four conditions for family=$FAMILY"
  for condition in "${CONDITIONS[@]}"; do
    IFS='|' read -r condition_id variant initialization <<< "$condition"
    write_task_config "$condition_id" "$variant" "$initialization" train
    if task_complete "$TASK_CONFIG" train; then
      log "SKIP validated trained run: $RUN_DIR"
      continue
    fi
    if [[ -d "$RUN_DIR" ]] && find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
      die "Refusing to overwrite or guess the state of partial/incompatible artifacts: $RUN_DIR. Use --evaluate-only after evaluation failure, or choose a fresh --output-root."
    fi
    run_task "$condition_id" "$variant" "$initialization" train
  done
fi

log "Training barrier passed; evaluating all four conditions for family=$FAMILY"
for condition in "${CONDITIONS[@]}"; do
  IFS='|' read -r condition_id variant initialization <<< "$condition"
  write_task_config "$condition_id" "$variant" "$initialization" evaluate
  if task_complete "$TASK_CONFIG" evaluate; then
    log "SKIP validated evaluation: $RUN_DIR"
    continue
  fi
  if ((EVALUATE_ONLY == 0)) && ! task_complete "$TASK_CONFIG" train; then
    die "Evaluation requires a validated trained checkpoint: $RUN_DIR"
  fi
  run_task "$condition_id" "$variant" "$initialization" evaluate
done

SUMMARY_PATH="$ROOT/run_summary.json" FAMILY="$FAMILY" ROOT="$ROOT" \
  "$PYTHON_BIN" - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path
summary = {
    "experiment_id": "granularity-init-48h-exploratory-lr3e-3-v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "family": os.environ["FAMILY"], "run_count": 4, "scale": "10M",
    "conditions": ["C1-scalar-default", "C3-channel-default", "C4-scalar-calibrated", "C6-channel-calibrated"],
    "seed": 42, "max_optimizer_steps": 1000, "learning_rate": 0.003,
    "status": "complete",
    "learning_rate_selection": "Fixed literature-informed rate; not tuned on these exploratory runs.",
    "interpretation": "One-family, one-seed, 1,000-step exploratory comparison only; not the full planned study.",
}
Path(os.environ["SUMMARY_PATH"]).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
PY
log "All requested training and evaluation tasks completed. Summary: $SUMMARY_PATH"
