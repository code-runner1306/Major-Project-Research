# Plan: Fix evaluation device placement and retry recovery

## Goal

Allow trained checkpoints to complete evaluation on the configured device, and make evaluation failures retryable without repeating completed training or damaging valid checkpoints.

## Findings

- Production selects `cuda` in `orchestration.run_seed`, but `construct_model` creates the evaluation-only model on CPU.
- `_load_final_checkpoint` restores weights without moving the model to the selected device. `benchmark_inference` creates prompts directly on CUDA, causing the reported CPU-model/CUDA-input embedding error.
- The evaluation path changes lifecycle status from `trained` to `evaluating`. If evaluation raises, the broad exception handler changes it to `failed`.
- On the next invocation, `evaluate_only` rejects `failed` before checking whether the completed training metrics and final checkpoint are still valid.
- Evaluation may have written some intermediate diagnostic files before failing. A retry must safely overwrite or regenerate those outputs and must not treat partial metrics as a complete result.

## Work plan

1. **Fix model placement for evaluation.** Move the newly constructed model to the selected `device` before loading the final checkpoint, or move it immediately after loading. Keep model parameters, inputs, and cache tensors on the same device. Ensure CPU smoke mode continues to work.
2. **Make evaluation retries safe.** If a run is `failed`, allow `--evaluate-only` to recover only when its manifest is compatible and `metrics.json` contains valid completed-training metadata and a valid `final_checkpoint.pt` for the configured step/token budget. Resume lifecycle at `evaluating`; reject failed runs whose training artifacts are absent, corrupt, or incompatible.
3. **Preserve training evidence.** On evaluation failure, retain the training section, final checkpoint, and resource measurements. Store the evaluation error separately or in lifecycle details; do not write a completion status or report entry until all required evaluation metrics validate.
4. **Make evaluation loop failures visible.** Update the recommended PowerShell invocation to stop at the first failed Python command (for example, test `$LASTEXITCODE` after each command and throw). Report the exact failed family/model/variant/seed so a long batch cannot silently continue through the same defect.
5. **Add focused regression coverage.** Cover GPU placement/load/evaluation when CUDA is available, CPU smoke evaluation, retry from `failed` with a valid completed-training checkpoint, refusal to retry with invalid training artifacts, and preservation of the lifecycle/checkpoint invariants.
6. **Validate before rerunning production evaluation.** Run the relevant unit suite and a short smoke/qualification path. Then retry one existing 10M TinyStories run with `--evaluate-only`, confirm it reaches `complete`, and inspect its metrics and checkpoint provenance before evaluating the remaining seeds.

## Acceptance criteria

- The model and benchmark prompts use the same selected device during production evaluation.
- A device mismatch no longer occurs at `model.prefill` in the inference benchmark.
- A failed evaluation with valid completed-training artifacts can be retried without retraining.
- Invalid, incomplete, or incompatible checkpoints remain blocked from evaluation.
- A run is marked `complete` only after its mandatory evaluation metrics and checkpoint validation pass.
- A batch evaluation stops on the first error and identifies the run that failed.

## Risks and safeguards

- Existing runs may be in `failed` lifecycle state even though their trained checkpoints are intact. Retry eligibility must validate artifacts, not simply reset every failed run.
- The traceback shows evaluation reached the cached/prefill inference benchmark. Other evaluation sections may expose additional device assumptions after this fix, so first retry one checkpoint before launching the full batch.
- Do not delete or overwrite `final_checkpoint.pt` or the training metrics while repairing evaluation state.

## Out of scope

- Mid-training optimizer-state recovery. This plan resumes evaluation from a completed training checkpoint; it does not resume an interrupted optimizer run.
- Changing benchmark workloads or research metrics.
