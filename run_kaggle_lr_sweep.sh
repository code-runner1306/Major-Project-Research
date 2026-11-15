#!/usr/bin/env bash
# Diagnostic only: LR sweep of the varied-data learnability diagnostic for ONE family.
# Session A runs FAMILY=mqar (default); session B runs FAMILY=overwrite. Each session
# writes its own research_results/lr_sweep_<family>_<stamp>/ folder and pushes it to
# its own branch sweep/<family>_<stamp>, so the two sessions never collide.
#
# Env knobs:
#   FAMILY              mqar (default) | overwrite
#   LRS                 space-separated learning rates (default "0.0003 0.001 0.003")
#   STEPS               optimizer steps per LR (default 2000)
#   EVAL_EVERY          evaluation interval (default 200)
#   TRAIN_EXAMPLES      training pool size (default 10000)
#   SEED                research seed (default 42)
#   TIME_BUDGET_SECONDS wall-clock budget for jobs, excludes git (default 36000 = 10h)
#   GIT_PUSH            1 (default) push results to a per-family branch | 0 skip
#   BRANCH              override branch name (default sweep/<family>_<stamp>)
#   GIT_REMOTE_URL      optional remote URL (may embed a token); sets origin if given
#   OUTPUT_ROOT         override output folder (must live inside this repo for push)
#   CPU_SMOKE           1 = tiny CPU-only end-to-end test (no GPU, no CUDA assert)
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FAMILY="${FAMILY:-mqar}"
LRS="${LRS:-0.0003 0.001 0.003}"
STEPS="${STEPS:-2000}"
EVAL_EVERY="${EVAL_EVERY:-200}"
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-10000}"
SEED="${SEED:-42}"
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-36000}"
GIT_PUSH="${GIT_PUSH:-1}"
CPU_SMOKE="${CPU_SMOKE:-0}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/research_results/lr_sweep_${FAMILY}_${STAMP}}"

case "$FAMILY" in mqar|overwrite) ;; *) echo "FAMILY must be mqar or overwrite (got: $FAMILY)" >&2; exit 2 ;; esac
for pair in "STEPS:$STEPS" "EVAL_EVERY:$EVAL_EVERY" "TRAIN_EXAMPLES:$TRAIN_EXAMPLES" \
            "SEED:$SEED" "TIME_BUDGET_SECONDS:$TIME_BUDGET_SECONDS"; do
  name="${pair%%:*}"; value="${pair#*:}"
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "Invalid $name: $value" >&2; exit 2; }
done
(( STEPS >= 10 && EVAL_EVERY >= 1 && TRAIN_EXAMPLES >= 8 )) || { echo 'STEPS/EVAL_EVERY/TRAIN_EXAMPLES out of range' >&2; exit 2; }
read -r -a LR_LIST <<< "$LRS"
(( ${#LR_LIST[@]} >= 1 )) || { echo "LRS is empty" >&2; exit 2; }
for lr in "${LR_LIST[@]}"; do
  [[ "$lr" =~ ^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]] && awk "BEGIN{exit !($lr > 0)}" \
    || { echo "Invalid learning rate: $lr" >&2; exit 2; }
done

# Resume support: an existing folder is only reused if this script created it.
if [[ -e "$OUTPUT_ROOT" ]]; then
  [[ -f "$OUTPUT_ROOT/.lr_sweep_marker" ]] || { echo "Refusing existing non-sweep folder: $OUTPUT_ROOT" >&2; exit 2; }
  [[ "$(cat "$OUTPUT_ROOT/.lr_sweep_marker")" == "$FAMILY" ]] || { echo "Folder belongs to another FAMILY: $OUTPUT_ROOT" >&2; exit 2; }
else
  mkdir -p "$OUTPUT_ROOT"
fi
printf '%s\n' "$FAMILY" > "$OUTPUT_ROOT/.lr_sweep_marker"

if [[ "$CPU_SMOKE" == "1" ]]; then
  NUM_GPUS=1
else
  NUM_GPUS="$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')"
  (( NUM_GPUS >= 1 )) || { echo 'CUDA is required (or set CPU_SMOKE=1)' >&2; exit 2; }
fi

printf 'Family:   %s\nLRS:      %s\nSteps:    %s (eval every %s)\nPool:     %s examples\nGPUs:     %s\nBudget:   %ss\nOutput:   %s\n' \
  "$FAMILY" "$LRS" "$STEPS" "$EVAL_EVERY" "$TRAIN_EXAMPLES" "$NUM_GPUS" "$TIME_BUDGET_SECONDS" "$OUTPUT_ROOT"

lr_tag() { printf '%.0e' "$1"; }

run_one() { # $1 = learning rate, $2 = gpu slot
  local lr="$1" slot="$2"
  local tag out log
  tag="$(lr_tag "$lr")"
  out="$OUTPUT_ROOT/${FAMILY}_lr${tag}"
  log="$OUTPUT_ROOT/${FAMILY}_lr${tag}.log"
  if [[ -f "$out/diagnostic_summary.json" ]]; then
    printf 'SKIP complete: lr=%s\n' "$lr"
    return 0
  fi
  if [[ -d "$out" ]]; then
    # Crashed or interrupted attempt: keep it for forensics, start clean.
    mv "$out" "${out}.partial_${STAMP}"
  fi
  local cmd=("$PYTHON_BIN" fixed_learnability.py
    --family "$FAMILY" --steps "$STEPS" --eval-every "$EVAL_EVERY"
    --train-examples "$TRAIN_EXAMPLES" --seed "$SEED" --learning-rate "$lr" --output "$out")
  if [[ "$CPU_SMOKE" == "1" ]]; then
    cmd+=(--cpu-smoke)
  fi
  printf 'LAUNCH: lr=%s gpu=%s -> %s\n' "$lr" "$slot" "$log"
  if [[ "$CPU_SMOKE" == "1" ]]; then
    "${cmd[@]}" > "$log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$slot" "${cmd[@]}" > "$log" 2>&1
  fi
}

RC=0
START_EPOCH="$(date +%s)"
idx=0
while (( idx < ${#LR_LIST[@]} )); do
  elapsed=$(( $(date +%s) - START_EPOCH ))
  if (( elapsed > TIME_BUDGET_SECONDS )); then
    remaining=("${LR_LIST[@]:idx}")
    printf '%s\n' "${remaining[@]}" > "$OUTPUT_ROOT/pending_jobs.txt"
    printf 'TIME BUDGET reached after %ss; %s job(s) deferred to pending_jobs.txt: %s\n' \
      "$elapsed" "${#remaining[@]}" "${remaining[*]}" >&2
    RC=1
    break
  fi
  wave=("${LR_LIST[@]:idx:NUM_GPUS}")
  pids=()
  for j in "${!wave[@]}"; do
    run_one "${wave[$j]}" "$(( j % NUM_GPUS ))" &
    pids+=($!)
  done
  trap 'kill "${pids[@]:-}" 2>/dev/null || true' INT TERM
  for pid in "${pids[@]}"; do
    wait "$pid" || { echo "Job failed (lr tag hunt in logs): see $OUTPUT_ROOT/*.log" >&2; RC=1; }
  done
  trap - INT TERM
  idx=$(( idx + NUM_GPUS ))
done

printf '\nAGGREGATING\n'
"$PYTHON_BIN" - "$OUTPUT_ROOT" "$FAMILY" "$STEPS" <<'PY' || RC=1
import json, pathlib, sys
root, family, steps = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
lrs = {}
for d in sorted(root.glob(f"{family}_lr*")):
    if not d.is_dir() or ".partial_" in d.name:
        continue
    tag = d.name.split("_lr", 1)[1]
    path = d / "diagnostic_summary.json"
    if not path.is_file():
        lrs[tag] = {"status": "failed_or_incomplete"}
        continue
    s = json.loads(path.read_text())
    held = s.get("held_out_curve") or {}
    train = s.get("train_sample_curve") or {}
    lrs[tag] = {
        "status": s.get("status"),
        "learning_rate": s.get("learning_rate"),
        "epochs_completed": s.get("epochs_completed"),
        "held_out_final": held.get("final"),
        "held_out_best": held.get("best"),
        "held_out_best_step": held.get("best_step"),
        "held_out_points": held.get("points"),
        "train_sample_final": train.get("final"),
        "shortcut_audit_passed": s.get("shortcut_audit_passed"),
        "wall_seconds": s.get("total_wall_seconds"),
    }
complete = {t: v for t, v in lrs.items()
            if v.get("status") == "complete" and v.get("held_out_final") is not None}
verdict, notes = "incomplete", []
if complete:
    finals = {t: v["held_out_final"] for t, v in complete.items()}
    max_final, min_final = max(finals.values()), min(finals.values())
    def still_rising(v):
        pts = v.get("held_out_points") or []
        return v["held_out_final"] >= 0.10 and len(pts) >= 2 and pts[-1]["exact_query_accuracy"] > pts[-2]["exact_query_accuracy"]
    def collapsed(v):
        best, final = v.get("held_out_best"), v["held_out_final"]
        return (best or 0) >= 0.10 and (v.get("held_out_best_step") or 0) < 0.8 * steps and final < 0.5 * best
    if any(still_rising(v) for v in complete.values()):
        verdict = "heldout_rising"
        notes.append("continue the rising LR; it has not plateaued")
    elif any(collapsed(v) for v in complete.values()):
        verdict = "rise_then_collapse"
        notes.append("degenerate-collapse signature; investigate collapse, do not just train longer")
    elif max_final - min_final > 0.10 and max_final >= 0.10:
        verdict = "lr_discriminative"
        notes.append("optimization sensitivity is central, consistent with Okpekpe & Orvieto")
    elif max_final < 0.02:
        verdict = "all_floor"
        notes.append("all LRs near zero; next is an attention control or a pipeline audit")
    else:
        verdict = "inconclusive"
        notes.append("no LR crossed 10% and none floor-clean; extend budget or tighten eval")
    if len(complete) < len(lrs):
        notes.append("some jobs failed; verdict uses complete LRs only")
summary = {
    "purpose": "exploratory LR-sweep diagnostic; not a production research result",
    "family": family, "steps": steps, "verdict": verdict, "notes": notes,
    "learning_rates": lrs,
}
(root / "sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"VERDICT: {verdict}")
for note in notes:
    print(f"  - {note}")
print(f"{'LR':>10} {'final':>8} {'best@step':>14} {'epochs':>8}  status")
for tag, v in sorted(lrs.items()):
    if v.get("held_out_final") is None:
        print(f"{tag:>10} {'-':>8} {'-':>14} {'-':>8}  {v.get('status')}")
    else:
        print(f"{tag:>10} {v['held_out_final']:>8.3f} "
              f"{v['held_out_best']:>6.3f}@{v['held_out_best_step']:<7} "
              f"{v['epochs_completed']:>8.1f}  {v.get('status')}")
PY

if [[ "$GIT_PUSH" == "1" ]]; then
  case "$OUTPUT_ROOT" in
    "$ROOT"/*) inside_repo=1 ;;
    *) inside_repo=0 ;;
  esac
  if (( inside_repo == 0 )); then
    echo "SKIP push: OUTPUT_ROOT is outside the repo ($OUTPUT_ROOT)" >&2
    RC=1
  elif ! git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    echo "SKIP push: $ROOT is not a git repository" >&2
    RC=1
  else
    if [[ -n "${GIT_REMOTE_URL:-}" ]]; then
      if git -C "$ROOT" remote get-url origin >/dev/null 2>&1; then
        git -C "$ROOT" remote set-url origin "$GIT_REMOTE_URL"
      else
        git -C "$ROOT" remote add origin "$GIT_REMOTE_URL"
      fi
    fi
    git -C "$ROOT" config user.name >/dev/null 2>&1 || git -C "$ROOT" config user.name "lr-sweep"
    git -C "$ROOT" config user.email >/dev/null 2>&1 || git -C "$ROOT" config user.email "lr-sweep@localhost"
    BRANCH="${BRANCH:-sweep/${FAMILY}_${STAMP}}"
    git -C "$ROOT" checkout -q -B "$BRANCH"
    # Commit only the sweep output even if unrelated changes are already staged.
    git -C "$ROOT" add -A -- "$OUTPUT_ROOT"
    git -C "$ROOT" commit -q -m "sweep ${FAMILY} ${STAMP}: LRS='${LRS}' steps=${STEPS} eval_every=${EVAL_EVERY}" -- "$OUTPUT_ROOT" \
      || { echo "Nothing to commit for $OUTPUT_ROOT" >&2; }
    if git -C "$ROOT" push -q -u origin "$BRANCH"; then
      printf 'PUSHED: branch %s\n' "$BRANCH"
    else
      echo "PUSH FAILED for $BRANCH (check remote/credentials; results remain in $OUTPUT_ROOT)" >&2
      RC=1
    fi
  fi
fi

printf '\nDONE rc=%s output=%s\n' "$RC" "$OUTPUT_ROOT"
exit "$RC"
