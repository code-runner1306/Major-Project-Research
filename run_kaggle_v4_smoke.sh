#!/usr/bin/env bash
# Time one corrected V4 MQAR run and one corrected V4 overwrite run on Kaggle.
# Compares one-at-a-time and two-GPU training, then projects 1,000-step timing.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: bash run_kaggle_v4_smoke.sh [options]

Runs 50 steps per V4 task in solo and two-GPU parallel modes. Compares worker
throughput and CPU/GPU utilization, then reports 1,000-step and seed estimates.
The CPU threshold is an operational saturation signal, not proof of hardware throttling.

Options:
  --python PATH       Python executable (default: python3)
  --study-config PATH Corrected config (default: study_configs/granularity_corrected_pilot.json)
  --output-root PATH  Output directory (default: unique folder under /kaggle/working)
  --steps N           Smoke optimizer steps (default: 50; minimum: 10)
  --dry-run           Validate and print the plan without GPU/auth checks or training
  -h, --help          Show this help
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
STUDY_CONFIG="${STUDY_CONFIG:-study_configs/granularity_corrected_pilot.json}"
OUTPUT_ROOT=""
SMOKE_STEPS=50
DRY_RUN=0
while (($#)); do
  case "$1" in
    --python) (($# >= 2)) || die "--python requires a path"; PYTHON_BIN="$2"; shift 2 ;;
    --study-config) (($# >= 2)) || die "--study-config requires a path"; STUDY_CONFIG="$2"; shift 2 ;;
    --output-root) (($# >= 2)) || die "--output-root requires a path"; OUTPUT_ROOT="$2"; shift 2 ;;
    --steps) (($# >= 2)) || die "--steps requires an integer"; SMOKE_STEPS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1 (use --help)" ;;
  esac
done
[[ "$SMOKE_STEPS" =~ ^[0-9]+$ ]] && ((SMOKE_STEPS >= 10)) || die "--steps must be an integer of at least 10"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || [[ -x "$PYTHON_BIN" ]] || die "Python executable not found: $PYTHON_BIN"
if [[ "$STUDY_CONFIG" != /* ]]; then STUDY_CONFIG="$ROOT/$STUDY_CONFIG"; fi
[[ -f "$STUDY_CONFIG" ]] || die "Study config not found: $STUDY_CONFIG"

CONFIG_INFO="$("$PYTHON_BIN" - "$STUDY_CONFIG" <<'PY'
import json, sys
from pathlib import Path
study = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
runtime = study.get("runtime", {})
retrieval = study.get("retrieval", {})
if runtime.get("protocol_version") != "corrected-v1":
    raise SystemExit("config must declare runtime.protocol_version=corrected-v1")
if retrieval.get("generator_version") != "mqar-overwrite-v4":
    raise SystemExit("config must select retrieval.generator_version=mqar-overwrite-v4")
budget = study.get("budgets", {}).get("10M")
if not budget:
    raise SystemExit("config must define budgets.10M")
print(json.dumps({
    "generator_version": retrieval["generator_version"],
    "learning_rate": budget["learning_rate"],
    "batch_size": budget["batch_size"],
    "grad_accum_steps": budget["grad_accum_steps"],
    "context_length": budget["context_length"],
    "state_accumulator_dtype": runtime.get("state_accumulator_dtype", "input"),
    "authorization_status": study.get("compute_authorization", {}).get("status", "missing"),
}, sort_keys=True))
PY
)" || die "Config validation failed: $STUDY_CONFIG"
LEARNING_RATE="$("$PYTHON_BIN" - "$STUDY_CONFIG" <<'PY'
import json, sys
from pathlib import Path
study = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
print(study["budgets"]["10M"]["learning_rate"])
PY
)" || die "Could not read the configured learning rate"

if [[ -z "$OUTPUT_ROOT" ]]; then
  if [[ -d /kaggle/working ]]; then OUTPUT_ROOT=/kaggle/working; else OUTPUT_ROOT="$ROOT/output"; fi
  OUTPUT_ROOT="$OUTPUT_ROOT/kaggle_v4_timing_smoke_$(date -u '+%Y%m%d_%H%M%S')"
fi
if ((DRY_RUN)); then
  printf 'DRY RUN\nConfig: %s\nConfig settings: %s\nFamilies: mqar, overwrite\nCondition: C4-scalar-calibrated\nLearning rate: %s\nSteps per family: %s\nTarget projection: 1000 steps\nOutput: %s\n' \
    "$STUDY_CONFIG" "$CONFIG_INFO" "$LEARNING_RATE" "$SMOKE_STEPS" "$OUTPUT_ROOT"
  exit 0
fi

# The user explicitly authorized this bounded two-task timing smoke. The
# production multi-seed authorization in the config remains untouched.

GPU_INFO="$("$PYTHON_BIN" - <<'PY'
import json, torch
devices = []
for index in range(torch.cuda.device_count()):
    with torch.cuda.device(index):
        free_b, total_b = torch.cuda.mem_get_info()
    devices.append({
        "index": index,
        "name": torch.cuda.get_device_name(index),
        "free_gib": round(free_b / 1024**3, 2),
        "total_gib": round(total_b / 1024**3, 2),
        "enough_free_memory": free_b >= 6 * 1024**3,
    })
ok = len(devices) >= 2 and all(device["enough_free_memory"] for device in devices[:2])
print(json.dumps({
    "ok": ok,
    "devices": devices,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "error": None if ok else "Two visible GPUs with at least 6 GiB free each are required for this parallel smoke.",
}))
PY
)" || die "Could not inspect the Kaggle GPU"
GPU_OK="$(printf '%s' "$GPU_INFO" | "$PYTHON_BIN" -c 'import json,sys; print("yes" if json.load(sys.stdin).get("ok") else "no")')"
[[ "$GPU_OK" == yes ]] || die "$GPU_INFO"

mkdir -p "$OUTPUT_ROOT/tasks" "$OUTPUT_ROOT/runs" "$OUTPUT_ROOT/logs"
OUT="$(cd -- "$OUTPUT_ROOT" && pwd)"
printf '%s\n' "$GPU_INFO" > "$OUT/hardware.json"
printf 'Config: %s\nConfig settings: %s\nGPU: %s\nOutput: %s\n' \
  "$STUDY_CONFIG" "$CONFIG_INFO" "$GPU_INFO" "$OUT"

for FAMILY in mqar overwrite; do
  TASK_CONFIG="$OUT/tasks/${FAMILY}_C4_scalar_calibrated_train.json"
  RUN_DIR="$OUT/runs/$FAMILY/C4-scalar-calibrated/seed_991"
  TASK_CONFIG_PATH="$TASK_CONFIG" RUN_DIR="$RUN_DIR" STUDY_CONFIG_PATH="$STUDY_CONFIG" \
    TASK_FAMILY="$FAMILY" TASK_STEPS="$SMOKE_STEPS" "$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path

study = json.loads(Path(os.environ["STUDY_CONFIG_PATH"]).read_text(encoding="utf-8-sig"))
budget = study["budgets"]["10M"]
retrieval = dict(study["retrieval"])
task = {
    "operation": "train",
    "experiment_id": "kaggle-v4-training-time-smoke-v1",
    "condition_id": "C4-scalar-calibrated",
    "family": os.environ["TASK_FAMILY"],
    "scale": "10M",
    "variant": "scalar",
    "initialization": "calibrated-geometric",
    "seed": 991,
    "output_dir": str(Path(os.environ["RUN_DIR"]).resolve()),
    "max_optimizer_steps": int(os.environ["TASK_STEPS"]),
    "batch_size": int(budget["batch_size"]),
    "grad_accum_steps": int(budget["grad_accum_steps"]),
    "context_length": int(budget["context_length"]),
    "validation_steps": int(budget.get("validation_steps", 100)),
    "evaluation_interval": int(budget.get("evaluation_interval", 100)),
    "learning_rate": float(budget["learning_rate"]),
    "calibration_min_half_life": int(study["calibration"]["min_half_life"]),
    "calibration_max_half_life": int(study["calibration"]["max_half_life"]),
    "calibration_examples": int(study["calibration"]["examples"]),
    "retrieval_config": retrieval,
    "run_mode": "production",
    "recurrence_backend": study.get("runtime", {}).get("recurrence_backend", "reference"),
    "state_accumulator_dtype": study.get("runtime", {}).get("state_accumulator_dtype", "input"),
    "recurrence_chunk_size": 64,
    "validation_answer_accuracy": True,
    "concurrent_training_runs": 2,
    "protocol_version": study["runtime"]["protocol_version"],
}
path = Path(os.environ["TASK_CONFIG_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
PY

done

printf '\nRunning solo baselines, then 2-worker parallel smoke (%s steps per task).\n' "$SMOKE_STEPS"
"$PYTHON_BIN" - "$OUT" "$SMOKE_STEPS" "$PYTHON_BIN" <<'PY'
import copy, json, os, pathlib, statistics, subprocess, sys, threading, time
root, steps, python = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
families = ("mqar", "overwrite")
for family in families:
    base = json.loads((root / "tasks" / f"{family}_C4_scalar_calibrated_train.json").read_text())
    for phase in ("solo", "parallel"):
        cfg = copy.deepcopy(base)
        cfg["output_dir"] = str(root / "runs" / phase / family / "C4-scalar-calibrated" / "seed_991")
        path = root / "tasks" / f"{phase}_{family}.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")

def sample_system(stop, samples):
    try:
        import psutil
        psutil.cpu_percent(None)
        while not stop.wait(1):
            samples["cpu"].append(psutil.cpu_percent(None))
            try:
                raw = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL)
                vals = [int(line.strip()) for line in raw.splitlines() if line.strip().isdigit()]
                if vals: samples["gpu"].append(vals)
            except Exception: pass
    except Exception as exc: samples["sampler_error"] = repr(exc)

def phase_run(phase, assignments):
    stop, samples = threading.Event(), {"cpu": [], "gpu": []}
    sampler = threading.Thread(target=sample_system, args=(stop, samples), daemon=True)
    procs, starts, logs = {}, {}, {}
    sampler.start(); phase_start = time.monotonic()
    for family, gpu in assignments.items():
        log = (root / "logs" / f"{phase}_{family}.log").open("w")
        logs[family] = log; starts[family] = time.monotonic()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        procs[family] = subprocess.Popen([python, "run_all.py", "--study-task-config", str(root / "tasks" / f"{phase}_{family}.json")], stdout=log, stderr=subprocess.STDOUT, env=env)
    while any(p.poll() is None for p in procs.values()): time.sleep(0.5)
    elapsed = time.monotonic() - phase_start
    durations = {f: time.monotonic() - starts[f] for f in procs}
    codes = {f: p.returncode for f, p in procs.items()}
    for log in logs.values(): log.close()
    stop.set(); sampler.join(timeout=3)
    cpu = samples["cpu"]
    gpu_samples = samples["gpu"]
    return {"makespan_seconds": elapsed, "family_seconds": durations, "exit_codes": codes,
            "avg_cpu_percent": statistics.mean(cpu) if cpu else None,
            "peak_cpu_percent": max(cpu) if cpu else None, "cpu_samples": len(cpu),
            "avg_gpu_percent_by_visible_index": [round(statistics.mean(row[i] for row in gpu_samples), 1) for i in range(min(2, min(map(len, gpu_samples))))] if gpu_samples else [],
            "gpu_samples": len(gpu_samples), "sampler_error": samples.get("sampler_error")}

solo = {}
for family in families:
    print(f"Solo baseline: {family}", flush=True)
    solo[family] = phase_run(f"solo_{family}", {family: 0})
    if solo[family]["exit_codes"][family] != 0:
        raise SystemExit(f"Solo run failed ({family}); see {root}/logs/solo_{family}_{family}.log")
print("Parallel trial: MQAR on GPU 0, overwrite on GPU 1", flush=True)
parallel = phase_run("parallel", {"mqar": 0, "overwrite": 1})
if any(code != 0 for code in parallel["exit_codes"].values()):
    raise SystemExit(f"Parallel run failed; inspect {root}/logs/parallel_*.log")

report = {"steps_per_task_per_mode": steps, "solo_baseline": solo, "two_worker_parallel": parallel}
solo_sum = sum(solo[f]["family_seconds"][f] for f in families)
pair_makespan = parallel["makespan_seconds"]
cpu_avg = parallel["avg_cpu_percent"]
cpu_clear = cpu_avg is not None and cpu_avg < 85
speedup = solo_sum / pair_makespan if pair_makespan else 0
individual_slowdowns = {f: parallel["family_seconds"][f] / solo[f]["family_seconds"][f] for f in families}
parallel_ok = cpu_clear and speedup >= 1.25 and max(individual_slowdowns.values()) <= 1.5
report["comparison"] = {"two_worker_speedup_vs_solo_pair": speedup,
    "per_family_slowdown_ratio": individual_slowdowns,
    "cpu_saturation_threshold_percent": 85, "cpu_below_threshold": cpu_clear,
    "max_workers_supported_by_this_smoke": 2 if parallel_ok else 1,
    "decision": "2 workers look viable for this workload" if parallel_ok else "2 workers show CPU/throughput contention; use 1 pending further diagnosis",
    "interpretation": "Operational screen only. CPU utilization does not expose cloud-host CPU frequency throttling; it detects saturation and run slowdown, not definitive throttling."}
report["timing_estimates"] = {"per_family_1000_step_seconds_solo": {f: solo[f]["family_seconds"][f] * 1000 / steps for f in families},
    "one_seed_serial_hours": solo_sum * 1000 / steps * 4 / 3600,
    "one_seed_two_worker_hours": pair_makespan * 1000 / steps * 4 / 3600,
    "assumption": "Corrected config has 4 conditions per dataset; C4 smoke timing is linearly projected to 1000 steps. Seed estimate multiplies by 4 conditions and omits final held-out evaluation."}
(root / "timing_estimate.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
PY
printf '\nParallel timing smoke complete. Estimates: %s\n' "$OUT/timing_estimate.json"
printf 'The smoke check confirms whether both GPU-specific runs completed. Estimates are planning figures, not research results.\n'
