#!/usr/bin/env bash
# Layout-fix training study on a Kaggle "GPU T4 x2" notebook: two workers, one per GPU
# (scalar on GPU 0, channel-wise on GPU 1). With --push-results, every finished model pair is pushed to a
# new results branch for this session (GH_TOKEN from Kaggle secrets; never pass tokens on the command line).
#
#   bash run_kaggle_layout_fix.sh --arms A2_no_block_first,A0_control --push-results
#   bash run_kaggle_layout_fix.sh --arms A1_varied_distance,A3_mixed_rewrites --push-results
#
# One variant per notebook (a single T4 is enough; one worker, seeds trained one after another):
#   bash run_kaggle_layout_fix.sh --arms A4_distance_and_rewrites --variants scalar --push-results
#   bash run_kaggle_layout_fix.sh --arms A4_distance_and_rewrites --variants channel-wise --push-results
#
# Options: --arms LIST  --variants scalar|channel-wise|scalar,channel-wise  --seeds 42,43,44
#          --devices 0,1 (default; 0 when one variant)  --branch NAME  --push-results  --plan-only  --cpu-smoke
set -Eeuo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
ARMS=''; VARIANTS='scalar,channel-wise'; SEEDS='42,43,44'; DEVICES=''; BRANCH=''; PUSH=0; EXTRA=()
while (( $# )); do
  case "$1" in
    --arms) ARMS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;;
    --devices) DEVICES="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --push-results) PUSH=1; shift ;;
    --plan-only) EXTRA+=(--plan-only); shift ;;
    --cpu-smoke) EXTRA+=(--cpu-smoke); shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$ARMS" ]] || { echo 'Choose --arms, e.g. --arms A2_no_block_first,A0_control' >&2; exit 2; }
IFS=',' read -r -a ARM_LIST <<< "$ARMS"
IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
IFS=',' read -r -a VARIANT_LIST <<< "$VARIANTS"
if [[ -z "$DEVICES" ]]; then
  if (( ${#VARIANT_LIST[@]} == 1 )); then DEVICES='0'; else DEVICES='0,1'; fi
fi
SESSION="$(date -u +%Y%m%dT%H%M%SZ)_$(head -c4 /dev/urandom | od -An -tx1 | tr -d ' \n')"
OUTPUT="output/layout_fix_study/layout-fix-v1/${SESSION}"
command=("${PYTHON_BIN:-python3}" run_layout_fix_study.py --arm "${ARM_LIST[@]}" --seeds "${SEED_LIST[@]}"
         --variants "${VARIANT_LIST[@]}"
         --devices "$DEVICES" --output "$OUTPUT" --max-seconds "${MAX_SECONDS:-39600}" "${EXTRA[@]}")
if (( PUSH )); then
  command+=(--publish)
  [[ -n "$BRANCH" ]] && command+=(--publish-branch "$BRANCH")
  [[ -n "${GH_TOKEN:-}" ]] || echo 'WARNING: GH_TOKEN is not set; pushing relies on existing Git credentials' >&2
fi
echo "Output: $OUTPUT"
# Hard stop shortly before Kaggle's 12-hour limit; the runner itself stops admitting pairs earlier.
exec timeout --signal=INT --kill-after=60s "${SESSION_SECONDS:-42600}s" "${command[@]}"
