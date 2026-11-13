#!/usr/bin/env bash
# Kaggle GPU pilot for the calibrated granularity comparison.
# In a Kaggle notebook, enable a GPU, then run:
#   !git clone --branch maink https://github.com/code-runner1306/Major-Project-Research.git
#   %cd /kaggle/working/Major-Project-Research
#   !bash run_kaggle_12h_replication.sh
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
HOURS="${HOURS:-12}"
PILOT_STEPS="${PILOT_STEPS:-20}"
TRAIN_STEPS="${TRAIN_STEPS:-650}"
SAFETY_MINUTES="${SAFETY_MINUTES:-20}"
SEEDS_TEXT="${SEEDS:-47 48 49}"
read -r -a SEEDS_ARRAY <<< "$SEEDS_TEXT"

STUDY_CONFIG="$ROOT/study_configs/granularity_corrected_pilot.json"
if [[ ! -f "$ROOT/run_all.py" || ! -f "$STUDY_CONFIG" ]]; then
  echo "ERROR: Run this script from the Major-Project-Research checkout." >&2
  exit 2
fi
if [[ ! "$HOURS" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ ! "$PILOT_STEPS" =~ ^[0-9]+$ ]] || [[ ! "$TRAIN_STEPS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: HOURS, PILOT_STEPS, and TRAIN_STEPS must be positive numbers." >&2
  exit 2
fi
if (( PILOT_STEPS < 10 || TRAIN_STEPS < 100 )); then
  echo "ERROR: PILOT_STEPS must be at least 10 and TRAIN_STEPS at least 100." >&2
  exit 2
fi
if (( ${#SEEDS_ARRAY[@]} < 1 )); then
  echo "ERROR: SEEDS must contain at least one seed." >&2
  exit 2
fi

GPU_INFO="$($PYTHON_BIN - <<'PY'
import json, sys
try:
    import torch
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"PyTorch import failed: {exc}"}))
    raise SystemExit(0)
if not torch.cuda.is_available():
    print(json.dumps({"ok": False, "error": "PyTorch cannot see a CUDA GPU. Enable Kaggle GPU acceleration and restart the session."}))
else:
    free_b, total_b = torch.cuda.mem_get_info()
    print(json.dumps({
        "ok": free_b >= 6 * 1024**3,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "free_gib": round(free_b / 1024**3, 2),
        "total_gib": round(total_b / 1024**3, 2),
        "error": None if free_b >= 6 * 1024**3 else "At least 6 GiB free GPU memory is required for the two-process timing/training pair."
    }))
PY
)"
# Corrected research compute needs recorded human approval in the committed config
# (same gate as study_runner). Fails before any GPU time is spent.
if ! AUTH_OUTPUT="$("$PYTHON_BIN" - "$STUDY_CONFIG" 2>&1 <<'PY'
import json, sys
from study_runner import require_compute_authorization
require_compute_authorization(json.load(open(sys.argv[1], encoding="utf-8-sig")), stage="kaggle-12h-replication")
PY
)"; then
  echo "ERROR: Compute authorization check failed for $STUDY_CONFIG: $AUTH_OUTPUT" >&2
  exit 2
fi
echo "Kaggle GPU: $GPU_INFO"
GPU_OK="$(printf '%s' "$GPU_INFO" | "$PYTHON_BIN" -c 'import json,sys; print("yes" if json.load(sys.stdin).get("ok") else "no")')"
if [[ "$GPU_OK" != "yes" ]]; then
  echo "ERROR: $GPU_INFO" >&2
  exit 2
fi

if [[ -d /kaggle/working ]]; then
  OUTPUT_BASE="${OUTPUT_BASE:-/kaggle/working}"
else
  OUTPUT_BASE="${OUTPUT_BASE:-$ROOT/output}"
fi
STAMP="$(date -u '+%Y%m%d_%H%M%S')"
OUT="$OUTPUT_BASE/kaggle_12h_mqar_replication_$STAMP"
mkdir -p "$OUT/tasks" "$OUT/logs" "$OUT/runs"
MASTER_LOG="$OUT/run.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

MAX_SECONDS="$("$PYTHON_BIN" -c 'import sys; print(int(float(sys.argv[1])*3600))' "$HOURS")"
RESERVE_SECONDS=$((SAFETY_MINUTES * 60))
WORKERS=2
EXPERIMENT_ID="kaggle-12h-corrected-calibrated-granularity-mqar-v1"
CONDITIONS=(
  "C4-scalar-calibrated|scalar|calibrated-geometric"
  "C6-channel-calibrated|channel-wise|calibrated-geometric"
)

write_task_config() {
  local filepath="$1" phase="$2" condition_id="$3" variant="$4" initialization="$5"
  local seed="$6" steps="$7" samples="$8" run_dir="$9" experiment_id="${10}"
  TASK_PATH="$filepath" STUDY_CONFIG_PATH="$STUDY_CONFIG" TASK_PHASE="$phase" \
    TASK_CONDITION="$condition_id" TASK_VARIANT="$variant" TASK_INITIALIZATION="$initialization" \
    TASK_SEED="$seed" TASK_STEPS="$steps" TASK_SAMPLES="$samples" TASK_RUN_DIR="$run_dir" \
    TASK_EXPERIMENT="$experiment_id" TASK_WORKERS="$WORKERS" "$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path

study = json.loads(Path(os.environ["STUDY_CONFIG_PATH"]).read_text(encoding="utf-8-sig"))
budget = study["budgets"]["10M"]
steps = int(os.environ["TASK_STEPS"])
samples = int(os.environ["TASK_SAMPLES"])
interval = 100 if steps >= 100 else max(5, steps // 2)
validation_steps = 100 if steps >= 100 else 4
retrieval = dict(study["retrieval"])
retrieval["validation_examples"] = samples
retrieval["held_out_examples"] = samples
task = {
    "operation": os.environ["TASK_PHASE"],
    "experiment_id": os.environ["TASK_EXPERIMENT"],
    "condition_id": os.environ["TASK_CONDITION"],
    "family": "mqar",
    "scale": "10M",
    "variant": os.environ["TASK_VARIANT"],
    "initialization": os.environ["TASK_INITIALIZATION"],
    "seed": int(os.environ["TASK_SEED"]),
    "output_dir": str(Path(os.environ["TASK_RUN_DIR"]).resolve()),
    "max_optimizer_steps": steps,
    "batch_size": int(budget["batch_size"]),
    "grad_accum_steps": int(budget["grad_accum_steps"]),
    "context_length": int(budget["context_length"]),
    "validation_steps": validation_steps,
    "evaluation_interval": interval,
    "learning_rate": 0.003,
    "calibration_min_half_life": int(study["calibration"]["min_half_life"]),
    "calibration_max_half_life": int(study["calibration"]["max_half_life"]),
    "calibration_examples": int(study["calibration"]["examples"]),
    "retrieval_config": retrieval,
    "run_mode": "production",
    "recurrence_backend": "reference",
    "state_accumulator_dtype": study.get("runtime", {}).get("state_accumulator_dtype", "input"),
    "recurrence_chunk_size": 64,
    "validation_answer_accuracy": True,
    "concurrent_training_runs": int(os.environ["TASK_WORKERS"]),
    "protocol_version": study.get("runtime", {}).get("protocol_version", "corrected-v1"),
}
path = Path(os.environ["TASK_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
PY
}

run_pair_phase() {
  local stage="$1" seed="$2" steps="$3" samples="$4" label="$5"
  local pids=() status_files=() configs=() tags=()
  local timeout_seconds=$(( MAX_SECONDS - ( $(date +%s) - pilot_start ) - RESERVE_SECONDS ))
  if (( timeout_seconds < 1 )); then
    echo "Time guard: no execution time remains before the safety reserve."
    return 1
  fi
  for entry in "${CONDITIONS[@]}"; do
    IFS='|' read -r condition_id variant initialization <<< "$entry"
    local tag="${label}_${condition_id}_seed${seed}_${stage}"
    local cfg="$OUT/tasks/${tag}.json"
    local run_dir="$OUT/runs/${label}/${condition_id}/seed_${seed}"
    local log="$OUT/logs/${tag}.log"
    local status="$OUT/tasks/${tag}.status"
    write_task_config "$cfg" "$stage" "$condition_id" "$variant" "$initialization" "$seed" "$steps" "$samples" "$run_dir" "$EXPERIMENT_ID"
    configs+=("$cfg"); status_files+=("$status"); tags+=("$tag")
    (
      if timeout --signal=INT --kill-after=30s "${timeout_seconds}s" \
          "$PYTHON_BIN" run_all.py --study-task-config "$cfg" 2>&1 | tee "$log"; then
        echo 0 > "$status"
      else
        rc=$?
        echo "$rc" > "$status"
      fi
    ) &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  for i in "${!status_files[@]}"; do
    local rc=1
    [[ -f "${status_files[$i]}" ]] && rc="$(<"${status_files[$i]}")"
    if [[ "$rc" != 0 ]]; then
      echo "FAILED: ${tags[$i]} (exit $rc). Log: ${OUT}/logs/${tags[$i]}.log"
      failed=1
    fi
  done
  return "$failed"
}

echo "Output: $OUT"
echo "Plan: C4 scalar-calibrated vs C6 channel-calibrated; MQAR; seeds: $SEEDS_TEXT"
echo "Timing first with $PILOT_STEPS steps and 64 examples; target: $TRAIN_STEPS steps, 512 examples, $HOURS hours maximum."
echo "Two model processes share the GPU during each paired phase."

# The timing pair uses the same model variants and initialization as the study.
PILOT_SEED=991
pilot_start="$(date +%s)"
phase_start="$(date +%s)"
if ! run_pair_phase train "$PILOT_SEED" "$PILOT_STEPS" 64 timing; then
  echo "ERROR: Timing training failed. Full training was not started; inspect $OUT/logs." >&2
  exit 3
fi
PILOT_TRAIN_SECONDS=$(( $(date +%s) - phase_start ))
phase_start="$(date +%s)"
if ! run_pair_phase evaluate "$PILOT_SEED" "$PILOT_STEPS" 64 timing; then
  echo "ERROR: Timing evaluation failed. Full training was not started; inspect $OUT/logs." >&2
  exit 3
fi
PILOT_EVAL_SECONDS=$(( $(date +%s) - phase_start ))
PILOT_SECONDS=$(( $(date +%s) - pilot_start ))

# Scale the observed pair timings conservatively. Evaluation is projected from 64 to 512 examples,
# with a 20-minute minimum per paired seed based on the earlier Exp1 timings.
PLAN_JSON="$(PILOT_TRAIN_SECONDS="$PILOT_TRAIN_SECONDS" PILOT_EVAL_SECONDS="$PILOT_EVAL_SECONDS" \
  PILOT_SECONDS="$PILOT_SECONDS" MAX_SECONDS="$MAX_SECONDS" RESERVE_SECONDS="$RESERVE_SECONDS" \
  PILOT_STEPS="$PILOT_STEPS" TRAIN_STEPS="$TRAIN_STEPS" REQUESTED_SEEDS="${#SEEDS_ARRAY[@]}" \
  "$PYTHON_BIN" - <<'PY'
import json, math, os
train = int(os.environ["PILOT_TRAIN_SECONDS"])
ev = int(os.environ["PILOT_EVAL_SECONDS"])
pilot = int(os.environ["PILOT_SECONDS"])
budget = int(os.environ["MAX_SECONDS"])
reserve = int(os.environ["RESERVE_SECONDS"])
scale = int(os.environ["TRAIN_STEPS"]) / int(os.environ["PILOT_STEPS"])
pair_train = train * scale
pair_eval = max(20 * 60, ev * 8)
pair_safe = math.ceil((pair_train + pair_eval) * 1.25)
available = max(0, budget - pilot - reserve)
requested = int(os.environ["REQUESTED_SEEDS"])
seeds_fit = min(requested, available // pair_safe if pair_safe else 0)
print(json.dumps({
  "pilot_training_seconds": train,
  "pilot_evaluation_seconds": ev,
  "pilot_total_seconds": pilot,
  "estimated_pair_training_seconds": round(pair_train),
  "estimated_pair_evaluation_seconds": pair_eval,
  "estimated_pair_seconds_with_25pct_buffer": pair_safe,
  "budget_seconds": budget,
  "reserve_seconds": reserve,
  "requested_seed_pairs": requested,
  "estimated_seed_pairs_that_fit": seeds_fit,
}))
PY
)"
printf '%s\n' "$PLAN_JSON" | tee "$OUT/timing_estimate.json"
SEEDS_TO_RUN="$(printf '%s' "$PLAN_JSON" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["estimated_seed_pairs_that_fit"])')"
if (( SEEDS_TO_RUN < 1 )); then
  echo "No complete 650-step paired seed is projected to fit inside the time budget after timing. Full training was not started."
  echo "Reduce TRAIN_STEPS or increase HOURS, then rerun. Timing outputs are in $OUT."
  exit 4
fi
if (( SEEDS_TO_RUN < ${#SEEDS_ARRAY[@]} )); then
  echo "WARNING: Estimate says only $SEEDS_TO_RUN of ${#SEEDS_ARRAY[@]} paired seeds fit. Running those first seeds and labeling the result exploratory."
fi

completed=0
for ((i=0; i<SEEDS_TO_RUN; i++)); do
  seed="${SEEDS_ARRAY[$i]}"
  elapsed=$(( $(date +%s) - pilot_start ))
  remaining=$(( MAX_SECONDS - elapsed ))
  pair_safe="$(printf '%s' "$PLAN_JSON" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["estimated_pair_seconds_with_25pct_buffer"])')"
  if (( remaining < pair_safe + RESERVE_SECONDS )); then
    echo "Time guard: skipping seed $seed; not enough estimated time for a complete train/evaluate pair plus reserve."
    break
  fi
  echo "=== Seed $seed: training C4 and C6 ==="
  if ! run_pair_phase train "$seed" "$TRAIN_STEPS" 512 production; then
    echo "ERROR: training failed for seed $seed; stopping before evaluation of incomplete checkpoints." >&2
    exit 5
  fi
  echo "=== Seed $seed: evaluating C4 and C6 ==="
  if ! run_pair_phase evaluate "$seed" "$TRAIN_STEPS" 512 production; then
    echo "ERROR: evaluation failed for seed $seed; inspect $OUT/logs." >&2
    exit 6
  fi
  completed=$((completed + 1))
  echo "Completed paired seed $seed ($completed/$SEEDS_TO_RUN)."
done

cat > "$OUT/README.txt" <<EOF
Kaggle 12-hour MQAR granularity replication
Comparison: scalar-calibrated C4 vs channel-wise-calibrated C6
Scale: 10M; optimizer steps: $TRAIN_STEPS; learning rate: 0.003
Requested seeds: $SEEDS_TEXT
Completed paired seeds: $completed
Timing estimate: $OUT/timing_estimate.json
Run artifacts and metrics: $OUT/runs/
Per-task logs: $OUT/logs/

This is an exploratory replication. A positive result is preliminary evidence, not proof of a publishable claim.
EOF
echo "Finished: $completed paired seed(s). Results: $OUT"
