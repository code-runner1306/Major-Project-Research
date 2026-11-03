#!/usr/bin/env bash
# Sequential train-then-evaluate launcher for TinyStories, MQAR, and overwrite.
# Configure with MODELS, SEEDS_<SCALE>, MAX_OPTIMIZER_STEPS, OUTPUT_DIR, and
# RECURRENCE_BACKEND environment variables; see --help below.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODELS="${MODELS:-10M 30M 55M 100M}"
OUTPUT_DIR="${OUTPUT_DIR:-output/full_research}"
MAX_OPTIMIZER_STEPS="${MAX_OPTIMIZER_STEPS:-1000}"
RECURRENCE_BACKEND="${RECURRENCE_BACKEND:-reference}"
RUN_PILOT=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: bash ./run_research_grid.sh [--pilot] [--dry-run]

The launcher trains each selected family in its own stage, then resumes that
family for evaluation and completion before moving to the next family:
TinyStories -> MQAR -> overwrite.

Configuration is through environment variables:
  MODELS="10M 30M 55M 100M"     Scales to include (space separated)
  SEEDS_10M="42,43,44,45,46"   Seeds for each scale are configured separately
  SEEDS_30M="42,43,44,45,46"
  SEEDS_55M="42"
  SEEDS_100M="42"
  MAX_OPTIMIZER_STEPS=1000       Fixed training budget for every run
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
USAGE
}

while (($#)); do
  case "$1" in
    --pilot) RUN_PILOT=1 ;;
    --no-pilot) RUN_PILOT=0 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

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
  printf '\n+'
  printf ' %q' "$@"
  printf '\n'
  if ((DRY_RUN == 0)); then
    "$@"
  fi
}

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
  echo "Pilot gate: scales=$PILOT_SCALES warmup=$PILOT_WARMUP measured_steps=$PILOT_STEPS"
  run "$PYTHON_BIN" run_all.py \
    --pilot --recurrence-backend "$RECURRENCE_BACKEND" --output "$OUTPUT_DIR" \
    --max-optimizer-steps "$MAX_OPTIMIZER_STEPS" --estimate-seeds "$ESTIMATE_SEEDS" \
    --pilot-scales "$PILOT_SCALES" --pilot-warmup "$PILOT_WARMUP" \
    --pilot-steps "$PILOT_STEPS" --pilot-data-fed \
    --pilot-parallel-scales "$PILOT_PARALLEL_SCALES" \
    --pilot-parallel-workers "$PILOT_PARALLEL_WORKERS"

  smoke_output="$OUTPUT_DIR/pilot_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
  run "$PYTHON_BIN" run_all.py --mode smoke --family all --output "$smoke_output"
fi

families=(tinystories mqar overwrite)
for family in "${families[@]}"; do
  echo
  echo "=== Train $family across selected models and seeds ==="
  for model in "${MODEL_LIST[@]}"; do
    run "$PYTHON_BIN" run_all.py \
      --mode production --recurrence-backend "$RECURRENCE_BACKEND" \
      --family "$family" --models "$model" --seeds "${SEEDS_BY_MODEL[$model]}" \
      --max-optimizer-steps "$MAX_OPTIMIZER_STEPS" --output "$OUTPUT_DIR" --train-only
  done

  echo
  echo "=== Evaluate and complete $family ==="
  for model in "${MODEL_LIST[@]}"; do
    run "$PYTHON_BIN" run_all.py \
      --mode production --recurrence-backend "$RECURRENCE_BACKEND" \
      --family "$family" --models "$model" --seeds "${SEEDS_BY_MODEL[$model]}" \
      --max-optimizer-steps "$MAX_OPTIMIZER_STEPS" --output "$OUTPUT_DIR" --evaluate-only
  done
done

echo
echo "Research grid complete. Results are under: $OUTPUT_DIR"
