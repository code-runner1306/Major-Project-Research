#!/usr/bin/env bash
# Family-specific 10M exploratory run. Requires Bash, Python, PyTorch CUDA, and this repository.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash run_48h_exploratory_study.sh --family mqar|overwrite|both [options]

Options:
  --python PATH         Python executable (default: python3)
  --output-root PATH    Production output directory (family-specific default)
  --study-config PATH   Study config (default: study_configs/granularity_corrected_pilot.json)
  --workers N           Production concurrency; default 1 (parallel requires a matching report)
  --qualify-workers N   Run isolated training+evaluation pilot sweep from 1 through N, then exit
  --qualification-steps N  Pilot optimizer steps per task (default: 20)
  --qualification-root PATH  Pilot artifacts (default: unique family-specific output folder)
  --qualification-report PATH  Report required to authorize production --workers > 1
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
STUDY_CONFIG="study_configs/granularity_corrected_pilot.json"
WORKERS=1
QUALIFY_WORKERS=""
QUALIFICATION_STEPS=20
QUALIFICATION_ROOT=""
QUALIFICATION_REPORT=""
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
    --workers)
      (($# >= 2)) || die "--workers requires a positive integer"
      WORKERS="$2"; shift 2 ;;
    --qualify-workers)
      (($# >= 2)) || die "--qualify-workers requires a positive maximum worker count"
      QUALIFY_WORKERS="$2"; shift 2 ;;
    --qualification-steps)
      (($# >= 2)) || die "--qualification-steps requires an integer >= 10"
      QUALIFICATION_STEPS="$2"; shift 2 ;;
    --qualification-root)
      (($# >= 2)) || die "--qualification-root requires a path"
      QUALIFICATION_ROOT="$2"; shift 2 ;;
    --qualification-report)
      (($# >= 2)) || die "--qualification-report requires a path"
      QUALIFICATION_REPORT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --evaluate-only) EVALUATE_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1 (use --help)" ;;
  esac
done

[[ -n "$FAMILY" ]] || die "--family is required"
case "$FAMILY" in
  mqar) FAMILIES=(mqar) ;;
  overwrite) FAMILIES=(overwrite) ;;
  both) FAMILIES=(mqar overwrite) ;;
  *) die "--family must be exactly 'mqar', 'overwrite', or 'both'" ;;
esac

[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || die "--workers must be a positive integer"
[[ "$QUALIFICATION_STEPS" =~ ^[0-9]+$ ]] && ((QUALIFICATION_STEPS >= 10)) || die "--qualification-steps must be an integer >= 10"
if [[ -n "$QUALIFY_WORKERS" ]]; then
  [[ "$QUALIFY_WORKERS" =~ ^[1-9][0-9]*$ ]] || die "--qualify-workers must be a positive integer"
  ((EVALUATE_ONLY == 0)) || die "--qualify-workers cannot be combined with --evaluate-only"
fi
MAX_RUN_WORKERS=$((4 * ${#FAMILIES[@]}))
if [[ -n "$QUALIFY_WORKERS" ]] && ((QUALIFY_WORKERS > MAX_RUN_WORKERS)); then
  die "--qualify-workers cannot exceed $MAX_RUN_WORKERS for this family selection (four production conditions per family)"
fi
((WORKERS <= MAX_RUN_WORKERS)) || die "--workers cannot exceed $MAX_RUN_WORKERS tasks in this family selection"
if [[ -n "$QUALIFY_WORKERS" ]] && [[ -n "$QUALIFICATION_REPORT" ]]; then
  die "--qualification-report is used for production, not while running --qualify-workers"
fi
if [[ -z "$QUALIFY_WORKERS" ]] && ((WORKERS > 1)) && [[ -z "$QUALIFICATION_REPORT" ]] && ((DRY_RUN == 0)); then
  die "Parallel production requires --qualification-report from a successful --qualify-workers sweep"
fi

cd -- "$REPO_DIR"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  die "Python executable not found or not executable: $PYTHON_BIN"
fi
if [[ "$STUDY_CONFIG" = /* ]]; then
  STUDY_CONFIG_PATH="$STUDY_CONFIG"
else
  STUDY_CONFIG_PATH="$REPO_DIR/$STUDY_CONFIG"
fi
[[ -f "$STUDY_CONFIG_PATH" ]] || die "Study config not found: $STUDY_CONFIG_PATH"
# The corrected protocol (mqar-overwrite-v4, float32 state accumulation,
# authorization gate) is selected by the config, not by this script.
PROTOCOL_VERSION="$("$PYTHON_BIN" - "$STUDY_CONFIG_PATH" <<'PY'
import json, sys
from pathlib import Path
study = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
print(study.get("runtime", {}).get("protocol_version", ""))
PY
)" || die "Could not read study config: $STUDY_CONFIG_PATH"
if [[ "$PROTOCOL_VERSION" == "corrected-v1" ]]; then
  CORRECTED=1; ROOT_PREFIX="granularity_48h_corrected_lr3e-3"
else
  CORRECTED=0; ROOT_PREFIX="granularity_48h_exploratory_lr3e-3"
fi
DEFAULT_OUTPUT_ROOT="output/${ROOT_PREFIX}_${FAMILY}"
if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$DEFAULT_OUTPUT_ROOT"
fi
if [[ -n "$QUALIFY_WORKERS" ]]; then
  ((WORKERS == 1)) || die "Use --qualify-workers as the target count; do not also set --workers"
  [[ -z "$OUTPUT_ROOT" || "$OUTPUT_ROOT" == "$DEFAULT_OUTPUT_ROOT" ]] || die "--output-root is for production; use --qualification-root for pilot artifacts"
  if [[ -z "$QUALIFICATION_ROOT" ]]; then
    QUALIFICATION_ROOT="output/granularity_48h_worker_qualification_${FAMILY}_$(date -u '+%Y%m%d_%H%M%S')_$$"
  fi
  if [[ "$QUALIFICATION_ROOT" = /* ]]; then QUAL_ROOT="$QUALIFICATION_ROOT"; else QUAL_ROOT="$REPO_DIR/$QUALIFICATION_ROOT"; fi
  [[ ! -e "$QUAL_ROOT" || -z "$(find "$QUAL_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]] || die "Qualification output root must be empty or new: $QUAL_ROOT"
  ROOT="$QUAL_ROOT"
else
  [[ -z "$QUALIFICATION_ROOT" ]] || die "--qualification-root is only used with --qualify-workers"
  if [[ "$OUTPUT_ROOT" = /* ]]; then ROOT="$OUTPUT_ROOT"; else ROOT="$REPO_DIR/$OUTPUT_ROOT"; fi
fi

CONDITIONS=(
  'C1-scalar-default|scalar|default'
  'C3-channel-default|channel-wise|default'
  'C4-scalar-calibrated|scalar|calibrated-geometric'
  'C6-channel-calibrated|channel-wise|calibrated-geometric'
)
SCALE="10M"
PRODUCTION_SEED=42
STEP_BUDGET=1000
LEARNING_RATE="0.003"
if ((CORRECTED)); then EXPERIMENT_ID="granularity-corrected-48h-exploratory-lr3e-3-v1"; else EXPERIMENT_ID="granularity-init-48h-exploratory-lr3e-3-v1"; fi
QUALIFICATION_RUN_ID="granularity-init-worker-pilot-${FAMILY}-$(date -u '+%Y%m%d_%H%M%S')-$$"
ACTIVE_PIDS=()
cleanup_parallel() {
  local pid
  for pid in "${ACTIVE_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${ACTIVE_PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  ACTIVE_PIDS=()
}
trap cleanup_parallel INT TERM

timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() {
  local line="[$(timestamp)] $*"
  printf '%s\n' "$line"
  if ((DRY_RUN == 0)); then printf '%s\n' "$line" >> "$MASTER_LOG"; fi
}

task_paths() {
  local root="$1" family="$2" condition_id="$3" seed="$4" operation="$5" prefix="${6:-}"
  TASK_TAG="${prefix}${family}_${SCALE}_${condition_id}_seed_${seed}_${operation}"
  TASK_CONFIG="$root/task_configs/${TASK_TAG}.json"
  TASK_LOG="$root/logs/${TASK_TAG}.log"
  RUN_DIR="$root/runs/$family/$SCALE/$condition_id/seed_$seed"
}

write_task_config() {
  local root="$1" family="$2" condition_id="$3" variant="$4" initialization="$5"
  local seed="$6" steps="$7" operation="$8" concurrency="$9" validation_samples="${10}"
  local experiment_id="${11}" qualification="${12:-false}" prefix="${13:-}"
  task_paths "$root" "$family" "$condition_id" "$seed" "$operation" "$prefix"
  mkdir -p -- "$root/task_configs"
  TASK_CONFIG="$TASK_CONFIG" STUDY_CONFIG_PATH="$STUDY_CONFIG_PATH" FAMILY="$family" \
    CONDITION_ID="$condition_id" VARIANT="$variant" INITIALIZATION="$initialization" \
    RUN_DIR="$RUN_DIR" SCALE="$SCALE" SEED="$seed" STEP_BUDGET="$steps" \
    LEARNING_RATE="$LEARNING_RATE" OPERATION="$operation" CONCURRENCY="$concurrency" \
    VALIDATION_SAMPLES="$validation_samples" EXPERIMENT_ID="$experiment_id" \
    QUALIFICATION="$qualification" "$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path

study = json.loads(Path(os.environ["STUDY_CONFIG_PATH"]).read_text(encoding="utf-8-sig"))
budget = study["budgets"][os.environ["SCALE"]]
retrieval = dict(study["retrieval"])
retrieval["validation_examples"] = int(os.environ["VALIDATION_SAMPLES"])
retrieval["held_out_examples"] = int(os.environ["VALIDATION_SAMPLES"])
validation_interval = 100 if int(os.environ["STEP_BUDGET"]) >= 100 else max(5, int(os.environ["STEP_BUDGET"]) // 2)
validation_steps = 100 if int(os.environ["STEP_BUDGET"]) >= 100 else 4
task = {
    "operation": os.environ["OPERATION"],
    "experiment_id": os.environ["EXPERIMENT_ID"],
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
    "validation_steps": validation_steps,
    "evaluation_interval": validation_interval,
    "learning_rate": float(os.environ["LEARNING_RATE"]),
    "calibration_min_half_life": int(study["calibration"]["min_half_life"]),
    "calibration_max_half_life": int(study["calibration"]["max_half_life"]),
    "calibration_examples": int(study["calibration"]["examples"]),
    "retrieval_config": retrieval,
    "run_mode": "production",
    "recurrence_backend": "reference",
    "state_accumulator_dtype": study.get("runtime", {}).get("state_accumulator_dtype", "input"),
    "recurrence_chunk_size": 64,
    "validation_answer_accuracy": True,
    "concurrent_training_runs": int(os.environ["CONCURRENCY"]),
}
if study.get("runtime", {}).get("protocol_version") == "corrected-v1":
    task["protocol_version"] = "corrected-v1"
if os.environ["QUALIFICATION"] == "true":
    task.update({"qualification": True, "qualification_stage": "worker-capacity-pilot",
                 "qualification_scope": "runtime-capacity-only"})
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
  local task_config="$1" task_log="$2" task_tag="$3" operation="$4" rc
  mkdir -p -- "$(dirname -- "$task_log")"
  printf '[%s] START %s\n' "$(timestamp)" "$task_tag"
  set +e
  "$PYTHON_BIN" run_all.py --study-task-config "$task_config" 2>&1 | tee -a "$task_log"
  rc=${PIPESTATUS[0]}
  set -e
  if ((rc != 0)); then printf '[%s] FAILED rc=%s %s\n' "$(timestamp)" "$rc" "$task_tag"; return "$rc"; fi
  if ! task_complete "$task_config" "$operation"; then
    printf '[%s] INVALID artifact after successful command %s\n' "$(timestamp)" "$task_tag"
    return 1
  fi
  printf '[%s] COMPLETED %s\n' "$(timestamp)" "$task_tag"
}

run_bounded_phase() {
  local phase="$1" max_workers="$2" offset j end rc failures=0
  shift 2
  local -a configs=("$@") pids=() labels=()
  for ((offset=0; offset<${#configs[@]}; offset+=max_workers)); do
    pids=(); labels=()
    end=$((offset + max_workers)); ((end > ${#configs[@]})) && end=${#configs[@]}
    for ((j=offset; j<end; j++)); do
      local config="${configs[$j]}"
      local label="$(basename -- "$config" .json)"
      local task_log="$ROOT/logs/${label}.log"
      mkdir -p -- "$ROOT/logs"
      printf '[%s] QUEUED %s\n' "$(timestamp)" "$phase $label" | tee -a "$MASTER_LOG"
      run_task "$config" "$task_log" "$phase $label" "$phase" &
      pids+=("$!"); labels+=("$label")
    done
    ACTIVE_PIDS=("${pids[@]}")
    for ((j=0; j<${#pids[@]}; j++)); do
      if wait "${pids[$j]}"; then rc=0; else rc=$?; fi
      if ((rc != 0)); then
        printf '[%s] FAILED %s rc=%s\n' "$(timestamp)" "${labels[$j]}" "$rc" | tee -a "$MASTER_LOG"
        failures=1
      else
        printf '[%s] FINISHED %s\n' "$(timestamp)" "${labels[$j]}" | tee -a "$MASTER_LOG"
      fi
    done
    ACTIVE_PIDS=()
    ((failures == 0)) || return 1
  done
}

summarize_phase() {
  local phase="$1" workers="$2" command_ok="$3" report_path="$4" makespan="$5" baseline_mean="$6"
  shift 6
  local -a configs=("$@")
  SUMMARY_PATH="$report_path" PHASE="$phase" QUALIFIED_WORKERS="$workers" COMMAND_OK="$command_ok" \
    PHASE_MAKESPAN="$makespan" BASELINE_TASK_SECONDS="$baseline_mean" \
    "$PYTHON_BIN" - "${configs[@]}" <<'PY'
import json, os, sys
from pathlib import Path
from study_runner import task_complete

phase = os.environ["PHASE"]
workers = int(os.environ["QUALIFIED_WORKERS"])
command_ok = os.environ["COMMAND_OK"] == "true"
resource_key = "training_resources" if phase == "train" else "evaluation_resources"
tasks, hardware, task_wall_times = [], None, []
total_tokens = total_examples = 0
safe = command_ok and bool(sys.argv[1:])
for config_file in sys.argv[1:]:
    task = json.loads(Path(config_file).read_text(encoding="utf-8-sig"))
    run_dir = Path(task["output_dir"])
    complete = task_complete(task, phase)
    try:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        metrics = {}
    capture = metrics.get(resource_key) or {}
    runtime = capture.get("runtime") or {}
    hardware = hardware or {key: runtime.get(key) for key in
                            ("gpu_uuid", "nvml_gpu_name", "torch_gpu_name", "driver_version",
                             "torch_version", "torch_cuda_runtime")}
    samples = [s for s in capture.get("activity_samples", []) if s.get("available") is True]
    memory = [s for s in samples if isinstance(s.get("memory_used_bytes"), (int, float))
              and isinstance(s.get("memory_total_bytes"), (int, float)) and s["memory_total_bytes"] > 0]
    vram = [100.0 * s["memory_used_bytes"] / s["memory_total_bytes"] for s in memory]
    other = [int(s["other_compute_process_count"]) for s in samples
             if isinstance(s.get("other_compute_process_count"), int)]
    max_vram = max(vram) if vram else None
    telemetry_ok = bool(runtime.get("cuda_available") and runtime.get("nvml_available")
                        and memory and max_vram is not None and max_vram < 95.0)
    unexpected_process = max(other, default=0) > max(0, workers - 1)
    task_safe = complete and telemetry_ok and not unexpected_process
    safe = safe and task_safe
    summary = capture.get("activity_summary") or {}
    if phase == "train":
        task_wall = (metrics.get("training") or {}).get("wall_time_seconds")
        token_count = int((metrics.get("training") or {}).get("processed_tokens") or 0)
        example_count = 0
        total_tokens += token_count
    else:
        task_wall = capture.get("duration_wall_seconds")
        token_count = 0
        example_count = int(((metrics.get("retrieval") or {}).get("evaluated_examples")) or 0)
        total_examples += example_count
    if isinstance(task_wall, (int, float)) and task_wall > 0:
        task_wall_times.append(float(task_wall))
    tasks.append({
        "family": task["family"], "condition_id": task["condition_id"], "seed": task["seed"],
        "run_dir": str(run_dir), "task_complete": complete,
        "nvml_samples": len(samples), "valid_vram_samples": len(memory),
        "max_whole_gpu_vram_percent": max_vram,
        "max_gpu_utilization_percent": (summary.get("gpu_utilization_percent") or {}).get("max"),
        "process_cpu_percent": summary.get("process_cpu_percent"),
        "system_cpu_percent": summary.get("system_cpu_percent"),
        "process_rss_bytes": summary.get("process_rss_bytes"),
        "system_memory_used_bytes": summary.get("system_memory_used_bytes"),
        "cpu_frequency_current_mhz": summary.get("cpu_frequency_current_mhz"),
        "other_compute_process_count_max": max(other) if other else None,
        "torch_memory": capture.get("torch_memory"), "telemetry_safe": telemetry_ok and not unexpected_process,
        "task_wall_seconds": task_wall, "processed_tokens": token_count,
        "evaluated_examples": example_count,
        "tokens_per_second": token_count / task_wall if phase == "train" and task_wall else None,
        "examples_per_second": example_count / task_wall if phase == "evaluate" and task_wall else None,
    })
baseline_mean = float(os.environ["BASELINE_TASK_SECONDS"]) if os.environ["BASELINE_TASK_SECONDS"] else None
expected_serial = baseline_mean * len(tasks) if baseline_mean is not None else None
makespan = float(os.environ["PHASE_MAKESPAN"])
result = {
    "phase": phase, "requested_workers": workers, "task_count": len(tasks),
    "command_success": command_ok, "qualified": safe, "telemetry_safe": safe,
    "memory_safety_limit_percent": 95.0, "hardware": hardware, "tasks": tasks,
    "phase_makespan_seconds": makespan,
    "sum_per_task_wall_seconds": sum(task_wall_times),
    "mean_per_task_wall_seconds": sum(task_wall_times) / len(task_wall_times) if task_wall_times else None,
    "total_processed_tokens": total_tokens if phase == "train" else None,
    "aggregate_tokens_per_second": total_tokens / makespan if phase == "train" and makespan else None,
    "total_evaluated_examples": total_examples if phase == "evaluate" else None,
    "aggregate_examples_per_second": total_examples / makespan if phase == "evaluate" and makespan else None,
    "estimated_serial_wall_seconds_from_one_worker_baseline": expected_serial,
    "relative_speedup_vs_one_worker_baseline": expected_serial / makespan if expected_serial and makespan else None,
    "gpu_telemetry_limitation": "NVML utilization and memory are whole-device readings repeated in each worker capture; do not sum them. NVML process visibility can be restricted.",
    "cpu_telemetry_limitation": "CPU use and sampled frequency are observations, not a definitive throttling diagnosis.",
}
path = Path(os.environ["SUMMARY_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print("QUALIFIED" if safe else "NOT_QUALIFIED")
sys.exit(0 if safe else 1)
PY
}

write_worker_report() {
  local report_path="$1" requested="$2" attempted_csv="$3" train_limit="$4" eval_limit="$5"
  local joint_limit="$6" config_sha="$7" revision="$8" gpu_json="$9"
  REPORT_PATH="$report_path" FAMILY_SELECTION="$FAMILY" FAMILIES_CSV="$(IFS=,; echo "${FAMILIES[*]}")" \
    REQUESTED_WORKERS="$requested" ATTEMPTED_COUNTS="$attempted_csv" TRAIN_LIMIT="$train_limit" \
    EVAL_LIMIT="$eval_limit" JOINT_LIMIT="$joint_limit" CONFIG_SHA="$config_sha" \
    GIT_REVISION="$revision" GPU_JSON="$gpu_json" QUALIFICATION_STEPS="$QUALIFICATION_STEPS" \
    "$PYTHON_BIN" - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path
root = Path(os.environ["REPORT_PATH"]).parent
attempted = [int(x) for x in os.environ["ATTEMPTED_COUNTS"].split(",") if x]
trials = []
for count in attempted:
    base = root / f"workers_{count}"
    train_path, eval_path = base / "training.json", base / "evaluation.json"
    training = json.loads(train_path.read_text(encoding="utf-8")) if train_path.exists() else {"qualified": False}
    evaluation = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {"qualified": False, "status": "not_run"}
    trials.append({"workers": count, "training": training, "evaluation": evaluation,
                   "jointly_qualified": bool(training.get("qualified") and evaluation.get("qualified"))})
result = {
    "schema_version": 1,
    "experiment_id": "granularity-init-48h-worker-qualification-v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "family_selection": os.environ["FAMILY_SELECTION"],
    "families": os.environ["FAMILIES_CSV"].split(","),
    "requested_target_workers": int(os.environ["REQUESTED_WORKERS"]),
    "attempted_worker_counts_in_order": attempted,
    "qualified_training_workers": int(os.environ["TRAIN_LIMIT"]),
    "qualified_evaluation_workers": int(os.environ["EVAL_LIMIT"]),
    "qualified_workers": int(os.environ["JOINT_LIMIT"]),
    "status": "qualified" if int(os.environ["JOINT_LIMIT"]) >= 1 else "not_qualified",
    "pilot_protocol": {
        "scale": "10M", "condition": "C3-channel-default",
        "optimizer_steps": int(os.environ["QUALIFICATION_STEPS"]), "learning_rate": 0.003,
        "seed_policy": "unique seed for each worker-count/family/worker slot",
        "validation_examples": 64, "held_out_examples": 64,
        "operation": "separate-process production-mode training and run_all evaluation phases",
    },
    "memory_safety_limit_percent": 95.0,
    "hardware": json.loads(os.environ["GPU_JSON"]),
    "study_config_sha256": os.environ["CONFIG_SHA"],
    "git_revision": os.environ["GIT_REVISION"],
    "trials": trials,
    "limitations": [
        "GPU utilization and VRAM are whole-device NVML measurements repeated per worker; do not sum them.",
        "CPU usage/frequency and per-process RSS are logged, but do not prove or disprove CPU throttling.",
        "NVML process visibility depends on driver/container permissions.",
        "A short pilot screens runtime capacity; it cannot guarantee long-run stability or behavior under different external GPU load/software versions.",
        "Each process has its own model, optimizer, data loader, seed/config, and output. Production conditions intentionally use paired seeds/data across variants for controlled comparison; no model state or computation graph is shared.",
    ],
}
path = Path(os.environ["REPORT_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
PY
}

validate_worker_report() {
  local report_path="$1" config_sha="$2" revision="$3" gpu_uuid="$4"
  REPORT_PATH="$report_path" FAMILY_SELECTION="$FAMILY" FAMILIES_CSV="$(IFS=,; echo "${FAMILIES[*]}")" \
    REQUESTED_WORKERS="$WORKERS" CONFIG_SHA="$config_sha" GIT_REVISION="$revision" GPU_UUID="$gpu_uuid" \
    "$PYTHON_BIN" - <<'PY'
import json, os, sys
from pathlib import Path
report = json.loads(Path(os.environ["REPORT_PATH"]).read_text(encoding="utf-8"))
families = os.environ["FAMILIES_CSV"].split(",")
requested = int(os.environ["REQUESTED_WORKERS"])
checks = {
    "qualification status": report.get("status") == "qualified",
    "same selected families": report.get("families") == families,
    "training worker limit": int(report.get("qualified_training_workers", 0)) >= requested,
    "evaluation worker limit": int(report.get("qualified_evaluation_workers", 0)) >= requested,
    "joint worker limit": int(report.get("qualified_workers", 0)) >= requested,
    "same study config": report.get("study_config_sha256") == os.environ["CONFIG_SHA"],
    "same code revision": report.get("git_revision") == os.environ["GIT_REVISION"],
    "same GPU UUID": report.get("hardware", {}).get("gpu_uuid") == os.environ["GPU_UUID"] and bool(os.environ["GPU_UUID"]),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    print("Parallel qualification does not authorize this run: " + ", ".join(failed), file=sys.stderr)
    sys.exit(1)
print("Parallel qualification accepted for", requested, "workers")
PY
}

if [[ -n "$QUALIFY_WORKERS" ]]; then
  printf 'Worker qualification: families=%s; target=%s; fallback order=%s' "$FAMILY" "$QUALIFY_WORKERS" "$QUALIFY_WORKERS"
  for ((n=QUALIFY_WORKERS-1; n>=1; n--)); do printf ',%s' "$n"; done
  printf '\nPilot root: %s\n' "$ROOT"
else
  printf 'Exploratory production: %s 10M runs; families=%s; seed=%s; steps=%s; LR=%s; workers=%s\n' \
    "$((4 * ${#FAMILIES[@]}))" "$FAMILY" "$PRODUCTION_SEED" "$STEP_BUDGET" "$LEARNING_RATE" "$WORKERS"
  printf 'Output root: %s\n' "$ROOT"
fi
printf 'Old-timing baseline: about 9 h 20 min training only per family. It excludes validation every 100 steps, calibration, and evaluation; finishing within 48 hours is not guaranteed.\n'
printf 'Production evaluation uses 512 held-out examples; worker qualification uses 64 validation and 64 held-out examples. Training remains at 50,000 examples.\n'
printf 'LR 3e-3 is fixed and literature-informed; it is not tuned on these runs.\n'

if ((DRY_RUN)); then
  if [[ -n "$QUALIFY_WORKERS" ]]; then
    for ((n=QUALIFY_WORKERS; n>=1; n--)); do
      printf '  QUALIFY %s total workers: up to %s separate tasks concurrently, then evaluate the same tasks concurrently\n' "$n" "$n"
    done
    printf 'DryRun starts no processes and writes no files.\n'
  else
    printf 'PLAN: train all selected conditions across all selected families before any evaluation (bounded to %s total workers).\n' "$WORKERS"
    for condition in "${CONDITIONS[@]}"; do
      IFS='|' read -r condition_id variant initialization <<< "$condition"
      for family in "${FAMILIES[@]}"; do
        printf '  TRAIN %s %s %s seed=%s steps=%s\n' "$family" "$variant" "$initialization" "$PRODUCTION_SEED" "$STEP_BUDGET"
      done
    done
    for condition in "${CONDITIONS[@]}"; do
      IFS='|' read -r condition_id variant initialization <<< "$condition"
      for family in "${FAMILIES[@]}"; do
        printf '  EVALUATE %s %s %s seed=%s\n' "$family" "$variant" "$initialization" "$PRODUCTION_SEED"
      done
    done
    if ((WORKERS > 1)); then printf 'Parallel run requires a matching qualification report.\n'; fi
  fi
  exit 0
fi

if ((CORRECTED)) && [[ -z "$QUALIFY_WORKERS" ]]; then
  # Corrected research compute needs recorded human approval (same gate as study_runner).
  auth_output="$("$PYTHON_BIN" - "$STUDY_CONFIG_PATH" 2>&1 <<'PY'
import json, sys
from study_runner import require_compute_authorization
require_compute_authorization(json.load(open(sys.argv[1], encoding="utf-8-sig")), stage="48h-production")
PY
)" || die "Compute authorization check failed for $STUDY_CONFIG: $auth_output"
fi
mkdir -p -- "$ROOT/logs" "$ROOT/task_configs"
MASTER_LOG="$ROOT/logs/run_48h_$(date -u '+%Y%m%d_%H%M%S').log"
: > "$MASTER_LOG"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
log "Python/PyTorch/CUDA preflight"
preflight_output="$("$PYTHON_BIN" -c 'import sys, torch, torch.nn; print("Python:", sys.executable); print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 3)' 2>&1)" \
  || die "Python/PyTorch CUDA preflight failed; no model run was started: $preflight_output"
log "$preflight_output"
CONFIG_SHA="$(sha256sum "$STUDY_CONFIG_PATH" | awk '{print $1}')"
GIT_REVISION="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || printf unknown)"
GPU_JSON="$("$PYTHON_BIN" - <<'PY'
import json, torch
from resource_telemetry import DeviceResourceCapture
capture = DeviceResourceCapture(torch.device("cuda"))
metadata = capture.runtime_metadata()
keys = ("gpu_uuid", "nvml_gpu_name", "torch_gpu_name", "driver_version", "torch_version", "torch_cuda_runtime")
print(json.dumps({key: metadata.get(key) for key in keys}))
PY
)"

if [[ -n "$QUALIFY_WORKERS" ]]; then
  attempted=() train_limit=0 eval_limit=0 joint_limit=0
  qualification_report="$ROOT/qualification_report.json"
  for ((count=QUALIFY_WORKERS; count>=1; count--)); do
    attempted+=("$count")
    trial_root="$ROOT/workers_$count"
    mkdir -p -- "$trial_root"
    train_configs=() eval_configs=()
    # Interleave families so a two-worker both-family pilot actually runs one task from each.
    idx=0
    for condition in "${CONDITIONS[@]}"; do
      IFS='|' read -r condition_id variant initialization <<< "$condition"
      for family in "${FAMILIES[@]}"; do
        ((idx < count)) || continue
        seed=$((81000 + count * 10 + idx))
        run_dir="$trial_root/runs/$family/$condition_id/seed_$seed"
        write_task_config "$trial_root" "$family" "$condition_id" "$variant" "$initialization" "$seed" "$QUALIFICATION_STEPS" train "$count" 64 "$QUALIFICATION_RUN_ID" true "pilot_"
        train_configs+=("$TASK_CONFIG")
        write_task_config "$trial_root" "$family" "$condition_id" "$variant" "$initialization" "$seed" "$QUALIFICATION_STEPS" evaluate "$count" 64 "$QUALIFICATION_RUN_ID" true "pilot_"
        eval_configs+=("$TASK_CONFIG")
        ((idx+=1))
      done
    done
    log "Qualification trial workers=$count: training ${#train_configs[@]} independent tasks"
    started=$SECONDS
    train_ok=1
    run_bounded_phase train "$count" "${train_configs[@]}" || train_ok=0
    train_elapsed=$((SECONDS - started))
    train_baseline=""
    if [[ -n "${ONE_WORKER_TRAIN_SECONDS:-}" ]]; then train_baseline="$ONE_WORKER_TRAIN_SECONDS"; fi
    summarize_phase train "$count" "$([[ $train_ok == 1 ]] && echo true || echo false)" "$trial_root/training.json" "$train_elapsed" "$train_baseline" "${train_configs[@]}" || train_ok=0
    if ((train_ok)); then
      train_limit=$count
      log "Qualification trial workers=$count: training passed; checking evaluation"
      started=$SECONDS
      eval_ok=1
      run_bounded_phase evaluate "$count" "${eval_configs[@]}" || eval_ok=0
      eval_elapsed=$((SECONDS - started))
      summarize_phase evaluate "$count" "$([[ $eval_ok == 1 ]] && echo true || echo false)" "$trial_root/evaluation.json" "$eval_elapsed" "" "${eval_configs[@]}" || eval_ok=0
      if ((eval_ok)); then eval_limit=$count; joint_limit=$count; fi
    else
      log "Qualification trial workers=$count: training failed or telemetry was insufficient; lowering target"
    fi
    if ((joint_limit > 0)); then break; fi
  done
  attempted_csv="$(IFS=,; echo "${attempted[*]}")"
  write_worker_report "$qualification_report" "$QUALIFY_WORKERS" "$attempted_csv" "$train_limit" "$eval_limit" "$joint_limit" "$CONFIG_SHA" "$GIT_REVISION" "$GPU_JSON"
  log "Worker qualification report: $qualification_report"
  log "Qualified training workers=$train_limit; evaluation workers=$eval_limit; joint maximum=$joint_limit"
  printf 'Next production command (same families, GPU, code revision and study config):\n'
  printf '  bash %q --family %q --workers %s --qualification-report %q\n' "$0" "$FAMILY" "$joint_limit" "$qualification_report"
  ((joint_limit > 0)) || die "No worker count passed both training and evaluation qualification. See $qualification_report"
  exit 0
fi

if ((WORKERS > 1)); then
  validate_worker_report "$QUALIFICATION_REPORT" "$CONFIG_SHA" "$GIT_REVISION" "$(printf '%s' "$GPU_JSON" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin).get("gpu_uuid") or "")')"
fi

train_configs=() eval_configs=()
idx=0
for condition in "${CONDITIONS[@]}"; do
  IFS='|' read -r condition_id variant initialization <<< "$condition"
  for family in "${FAMILIES[@]}"; do
    write_task_config "$ROOT" "$family" "$condition_id" "$variant" "$initialization" "$PRODUCTION_SEED" "$STEP_BUDGET" train "$WORKERS" 512 "$EXPERIMENT_ID"
    train_configs+=("$TASK_CONFIG")
    write_task_config "$ROOT" "$family" "$condition_id" "$variant" "$initialization" "$PRODUCTION_SEED" "$STEP_BUDGET" evaluate "$WORKERS" 512 "$EXPERIMENT_ID"
    eval_configs+=("$TASK_CONFIG")
  done
done

if ((EVALUATE_ONLY == 0)); then
  pending_train_configs=()
  for config in "${train_configs[@]}"; do
    readarray -t task_info < <("$PYTHON_BIN" - "$config" <<'PY'
import json, sys
t=json.load(open(sys.argv[1], encoding="utf-8-sig"))
print(t["output_dir"]); print(t["family"]); print(t["condition_id"])
PY
)
    run_dir="${task_info[0]}"
    if task_complete "$config" train; then log "SKIP validated trained run: $run_dir"; continue; fi
    if [[ -d "$run_dir" ]] && find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
      die "Refusing to overwrite partial/incompatible artifacts: $run_dir. Use a fresh --output-root."
    fi
    pending_train_configs+=("$config")
  done
  if ((${#pending_train_configs[@]})); then
    log "Training all pending selected tasks, bounded to $WORKERS total workers"
    run_bounded_phase train "$WORKERS" "${pending_train_configs[@]}" || die "One or more training tasks failed; evaluation has not started. See per-task logs under $ROOT/logs."
  else
    log "All selected training tasks are already complete"
  fi
fi

log "Training barrier passed; evaluating all selected tasks, bounded to $WORKERS total workers"
pending_eval_configs=()
for config in "${eval_configs[@]}"; do
  if task_complete "$config" evaluate; then log "SKIP validated evaluation: $config"; continue; fi
  if ! task_complete "$config" train; then die "Evaluation requires a validated trained checkpoint. Run training first: $config"; fi
  pending_eval_configs+=("$config")
done
if ((${#pending_eval_configs[@]})); then
  run_bounded_phase evaluate "$WORKERS" "${pending_eval_configs[@]}" || die "One or more evaluation tasks failed. See per-task logs under $ROOT/logs."
else
  log "All selected evaluations are already complete"
fi

SUMMARY_PATH="$ROOT/run_summary.json" EXPERIMENT_ID="$EXPERIMENT_ID" FAMILY="$FAMILY" FAMILIES_CSV="$(IFS=,; echo "${FAMILIES[*]}")" \
  "$PYTHON_BIN" - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path
families=os.environ["FAMILIES_CSV"].split(",")
summary = {
    "experiment_id": os.environ["EXPERIMENT_ID"],
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "family_selection": os.environ["FAMILY"], "families": families,
    "run_count": 4*len(families), "scale": "10M",
    "conditions": ["C1-scalar-default", "C3-channel-default", "C4-scalar-calibrated", "C6-channel-calibrated"],
    "seed": 42, "max_optimizer_steps": 1000, "learning_rate": 0.003,
    "status": "complete", "training_barrier_before_evaluation": True,
    "learning_rate_selection": "Fixed literature-informed rate; not tuned on these exploratory runs.",
    "interpretation": "One seed and 1,000 steps per condition; exploratory comparison only, not the full planned study.",
}
Path(os.environ["SUMMARY_PATH"]).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
PY
log "All requested training and evaluation tasks completed. Summary: $SUMMARY_PATH"
