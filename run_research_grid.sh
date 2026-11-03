#!/usr/bin/env bash
# Bounded-parallel train-then-evaluate launcher for TinyStories, MQAR, and overwrite.
# Configure with PARALLEL_WORKERS, MODELS, SEEDS_<SCALE>, MAX_OPTIMIZER_STEPS, OUTPUT_DIR, and
# RECURRENCE_BACKEND environment variables; see --help below.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODELS="${MODELS:-10M 30M 55M 100M}"
OUTPUT_DIR="${OUTPUT_DIR:-output/full_research}"
MAX_OPTIMIZER_STEPS="${MAX_OPTIMIZER_STEPS:-1000}"
RECURRENCE_BACKEND="${RECURRENCE_BACKEND:-reference}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-1}"
RUN_PILOT=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: bash ./run_research_grid.sh [--workers N] [--pilot] [--dry-run]

The launcher trains each selected family in its own stage, then resumes that
family for sequential evaluation and completion before moving to the next family:
TinyStories -> MQAR -> overwrite.

Configuration is through environment variables:
  MODELS="10M 30M 55M 100M"     Scales to include (space separated)
  SEEDS_10M="42,43,44,45,46"   Seeds for each scale are configured separately
  SEEDS_30M="42,43,44,45,46"
  SEEDS_55M="42"
  SEEDS_100M="42"
  MAX_OPTIMIZER_STEPS=1000       Fixed training budget for every run
  --workers N or PARALLEL_WORKERS=N  Maximum concurrent training processes; default is 1
  OUTPUT_DIR=output/full_research
  RECURRENCE_BACKEND=reference   Or cuda-fused with a valid qualification artifact
  PYTHON_BIN=python3
  PILOT_WARMUP=1 PILOT_STEPS=2  Short V100 pilot settings used with --pilot
  PILOT_PARALLEL_SCALES=10M,30M PILOT_PARALLEL_WORKERS=2,3

--pilot first runs a V100 throughput pilot for the selected scales and both
gate variants, including data-fed TinyStories, MQAR, and overwrite samples. It
also runs a bounded 1/2/3-worker concurrency check for the configured low-tier
scales (default 10M and 30M), skipping worker counts that fail the VRAM safety
gate, then runs the all-family CPU smoke suite. These are short profiling
samples, not production training. The larger workload means pilot duration can
exceed 15–20 minutes; the estimate itself is written in seconds after sampling.

--dry-run prints the commands without executing them. The script defaults to
the requested budget proposal: five seeds at 10M and 30M, one seed at 55M and
100M. Override any SEEDS_<SCALE> value to change those counts.

PARALLEL_WORKERS greater than 1 automatically runs a matching data-fed pilot
qualification first. A requested count that fails the VRAM gate falls back to
one training process for that scale. Training processes are isolated; evaluation
stays sequential so inference timing measurements are not made under contention.
USAGE
}

while (($#)); do
  case "$1" in
    --workers|--parallel-workers)
      (($# >= 2)) || { echo "$1 requires a positive integer." >&2; exit 2; }
      PARALLEL_WORKERS="$2"
      shift
      ;;
    --workers=*) PARALLEL_WORKERS="${1#*=}" ;;
    --parallel-workers=*) PARALLEL_WORKERS="${1#*=}" ;;
    --pilot) RUN_PILOT=1 ;;
    --no-pilot) RUN_PILOT=0 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ ! "$PARALLEL_WORKERS" =~ ^[1-9][0-9]{0,2}$ ]] || ((PARALLEL_WORKERS > 128)); then
  echo "PARALLEL_WORKERS must be an integer from 1 to 128." >&2
  exit 2
fi
if ((PARALLEL_WORKERS > 1 && DRY_RUN == 0 && RUN_PILOT == 0)); then
  RUN_PILOT=1
  echo "Parallel training requested; running the required capacity qualification pilot first."
fi

read -r -a MODEL_LIST <<< "$MODELS"
if ((${#MODEL_LIST[@]} == 0)); then
  echo "MODELS must contain at least one scale." >&2
  exit 2
fi

declare -A SEEDS_BY_MODEL=(
  [10M]="${SEEDS_10M:-42,43,44,45,46}"
  [30M]="${SEEDS_30M:-42,43,44,45,46}"
  [55M]="${SEEDS_55M:-42}"
  [100M]="${SEEDS_100M:-42}"
)

for model in "${MODEL_LIST[@]}"; do
  case "$model" in
    10M|30M|55M|100M) ;;
    *) echo "Unsupported model scale '$model'. Choose 10M, 30M, 55M, or 100M." >&2; exit 2 ;;
  esac
  if [[ -z "${SEEDS_BY_MODEL[$model]}" ]]; then
    echo "No seeds configured for $model." >&2
    exit 2
  fi
done

run() {
  printf '\n[%s] +' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf ' %q' "$@"
  printf '\n'
  if ((DRY_RUN == 0)); then
    "$@"
  fi
}

log_progress() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

if ((DRY_RUN == 0)); then
  mkdir -p "$OUTPUT_DIR"
  RUN_LOG="$OUTPUT_DIR/research_grid_$(date -u +%Y%m%dT%H%M%SZ).log"
  exec > >(tee -a "$RUN_LOG") 2>&1
  log_progress "Run log: $RUN_LOG"
fi

if ((RUN_PILOT)); then
  PILOT_SCALES="$(IFS=,; printf '%s' "${MODEL_LIST[*]}")"
  ESTIMATE_SEEDS=""
  for model in "${MODEL_LIST[@]}"; do
    IFS=, read -r -a model_seeds <<< "${SEEDS_BY_MODEL[$model]}"
    seed_count="${#model_seeds[@]}"
    if [[ -n "$ESTIMATE_SEEDS" ]]; then ESTIMATE_SEEDS+=,; fi
    ESTIMATE_SEEDS+="$model:$seed_count"
  done
  PILOT_WARMUP="${PILOT_WARMUP:-1}"
  PILOT_STEPS="${PILOT_STEPS:-2}"
  PILOT_PARALLEL_SCALES="${PILOT_PARALLEL_SCALES:-10M,30M}"
  PILOT_PARALLEL_WORKERS="${PILOT_PARALLEL_WORKERS:-2,3}"
  parallel_qualification_args=()
  if ((PARALLEL_WORKERS > 1)); then
    PILOT_PARALLEL_SCALES="$PILOT_SCALES"
    PILOT_PARALLEL_WORKERS="$PARALLEL_WORKERS"
    parallel_qualification_args=(--pilot-require-parallel-workers "$PARALLEL_WORKERS")
  fi
  log_progress "Starting pilot: scales=$PILOT_SCALES warmup=$PILOT_WARMUP measured_steps=$PILOT_STEPS data-fed families=tinystories,mqar,overwrite"
  run "$PYTHON_BIN" run_all.py \
    --pilot --recurrence-backend "$RECURRENCE_BACKEND" --output "$OUTPUT_DIR" \
    --max-optimizer-steps "$MAX_OPTIMIZER_STEPS" --estimate-seeds "$ESTIMATE_SEEDS" \
    --pilot-scales "$PILOT_SCALES" --pilot-warmup "$PILOT_WARMUP" \
    --pilot-steps "$PILOT_STEPS" --pilot-data-fed \
    --pilot-parallel-scales "$PILOT_PARALLEL_SCALES" \
    --pilot-parallel-workers "$PILOT_PARALLEL_WORKERS" \
    "${parallel_qualification_args[@]}"

  smoke_output="$OUTPUT_DIR/pilot_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
  log_progress "Starting post-pilot CPU smoke suite; output=$smoke_output"
  run "$PYTHON_BIN" run_all.py --mode smoke --family all --output "$smoke_output"
fi

families=(tinystories mqar overwrite)

effective_workers_for_scale() {
  local scale="$1"
  if ((PARALLEL_WORKERS == 1 || DRY_RUN == 1)); then
    printf '%s\n' "$PARALLEL_WORKERS"
    return
  fi
  "$PYTHON_BIN" -c '
import json, sys
path, scale, requested = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(path, encoding="utf-8") as handle:
    qualification = json.load(handle)
statuses = qualification["scale_dataset_statuses"][scale]
print(requested if all(value == "complete" for value in statuses.values()) else 1)
' "$OUTPUT_DIR/parallel_worker_qualification.json" "$scale" "$PARALLEL_WORKERS"
}

run_training_scale() {
  local family="$1" model="$2" worker_limit="$3" active=0 failed=0
  local variant seed
  local variants=(scalar channel-wise)
  if [[ "${ENABLE_PARAMETER_MATCHED_SCALAR_CONTROL:-0}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
    variants+=(scalar-matched)
  fi
  IFS=, read -r -a model_seeds <<< "${SEEDS_BY_MODEL[$model]}"
  for seed in "${model_seeds[@]}"; do
    for variant in "${variants[@]}"; do
      log_progress "Queue training task dataset=$family model=$model variant=$variant seed=$seed (worker limit=$worker_limit)"
      if ((DRY_RUN == 1)); then
        run "$PYTHON_BIN" run_all.py \
          --mode production --recurrence-backend "$RECURRENCE_BACKEND" \
          --family "$family" --models "$model" --seeds "$seed" --variant "$variant" \
          --max-optimizer-steps "$MAX_OPTIMIZER_STEPS" --output "$OUTPUT_DIR" \
          --concurrent-training-runs "$worker_limit" --train-only --skip-report
        continue
      fi
      run "$PYTHON_BIN" run_all.py \
        --mode production --recurrence-backend "$RECURRENCE_BACKEND" \
        --family "$family" --models "$model" --seeds "$seed" --variant "$variant" \
        --max-optimizer-steps "$MAX_OPTIMIZER_STEPS" --output "$OUTPUT_DIR" \
        --concurrent-training-runs "$worker_limit" --train-only --skip-report &
      active=$((active + 1))
      if ((active >= worker_limit)); then
        if wait -n; then :; else failed=1; fi
        active=$((active - 1))
      fi
    done
  done
  while ((active > 0)); do
    if wait -n; then :; else failed=1; fi
    active=$((active - 1))
  done
  if ((failed != 0)); then
    log_progress "One or more training workers failed for dataset=$family model=$model; stopping before evaluation."
    return 1
  fi
}

for family in "${families[@]}"; do
  echo
  log_progress "Starting training stage for dataset=$family"
  for model in "${MODEL_LIST[@]}"; do
    model_workers="$(effective_workers_for_scale "$model")"
    if ((model_workers == 1 && PARALLEL_WORKERS > 1)); then
      log_progress "Training dataset=$family model=$model sequentially; requested worker count failed at least one VRAM capacity gate."
    else
      log_progress "Training dataset=$family model=$model with up to $model_workers workers."
    fi
    run_training_scale "$family" "$model" "$model_workers"
  done

  echo
  log_progress "Starting evaluation/completion stage for dataset=$family"
  for model in "${MODEL_LIST[@]}"; do
    model_workers="$(effective_workers_for_scale "$model")"
    variants=(scalar channel-wise)
    if [[ "${ENABLE_PARAMETER_MATCHED_SCALAR_CONTROL:-0}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
      variants+=(scalar-matched)
    fi
    IFS=, read -r -a model_seeds <<< "${SEEDS_BY_MODEL[$model]}"
    for variant in "${variants[@]}"; do
      for seed in "${model_seeds[@]}"; do
        log_progress "Evaluating dataset=$family model=$model variant=$variant seed=$seed"
        run "$PYTHON_BIN" run_all.py \
          --mode production --recurrence-backend "$RECURRENCE_BACKEND" \
          --family "$family" --models "$model" --seeds "$seed" --variant "$variant" \
          --max-optimizer-steps "$MAX_OPTIMIZER_STEPS" --output "$OUTPUT_DIR" \
          --concurrent-training-runs "$model_workers" --evaluate-only
      done
    done
  done
done

echo
echo "Research grid complete. Results are under: $OUTPUT_DIR"
