# A6000 held-out decay evaluation

This evaluates existing checkpoints. It does not train, overwrite checkpoints, publish, or push to GitHub. Run after the joint training job releases the A6000 if it uses the same GPU.

In the existing CUDA-enabled Python environment, from the repository root:

```powershell
powershell.exe -File .\run_A6000_decay_boundary.ps1
```

Default: both tasks, seeds 42/43/44, 24 complete legacy exploratory checkpoints, 512 fresh discovery examples per cell, fixed gates fitted on the first 256 exact training examples, batch 32, **two evaluation workers sharing GPU 0**. The full default comprises **360 evaluations**: 288 stress/intervention executions plus 72 paired-update executions. Results are saved under a unique `research_results/dynamic_decay_boundary_v1/<session>/` folder.

Each worker receives disjoint scalar/channel initialization pairs, balanced by scored-cell workload. Separate worker folders/logs prevent concurrent writes to shared results. On successful completion, the coordinator checks exact coverage and prediction hashes, then produces one combined report and artifact set at the session root. Wall time is coordinator elapsed time, not the sum of concurrent worker durations. Two workers are not guaranteed to halve runtime; GPU throughput for this concurrent evaluator remains unmeasured.

Use the existing training environment's CUDA PyTorch and repository dependencies. This launcher does not install or replace PyTorch. A6000-family CUDA devices are supported; the code does not require a particular GPU name. GPU runtime for this new suite has not yet been measured.

## Useful commands

```powershell
# Inspect selected models and workload; no evaluation or GPU allocation
.\run_A6000_decay_boundary.ps1 -PlanOnly

# Explicitly request two workers (the default)
.\run_A6000_decay_boundary.ps1 -Workers 2

# Retain the single-worker option
.\run_A6000_decay_boundary.ps1 -Workers 1

# Run only overwrite
.\run_A6000_decay_boundary.ps1 -Family overwrite

# Run one seed
.\run_A6000_decay_boundary.ps1 -Seeds 42

# Explicit Python environment
.\run_A6000_decay_boundary.ps1 -Python .\venv\Scripts\python.exe

# Use a different checkpoint aggregation root
.\run_A6000_decay_boundary.ps1 -CheckpointRoot 'D:\research\calibrated_recipe_study'

# Resume an interrupted run: repeat all original selections and examples/batch settings
.\run_A6000_decay_boundary.ps1 -OutputRoot 'research_results\dynamic_decay_boundary_v1\YOUR_SESSION' -Resume
```

No joint checkpoints are required for the default command. After the joint run has completed, include its complete selected matrix explicitly:

```powershell
.\run_A6000_decay_boundary.ps1 -JointRoot 'output\joint_recipe_study\joint-timescale-sensitivity-v1'
```

That creates a new combined evaluation session with 36 models and 540 evaluations. It uses the identical data seeds and intervention policy; legacy models are reevaluated so the combined report is self-contained. Do not add joint models to an existing session via resume: checkpoint selection is immutable. If multiple sessions contain duplicate conditions, use `-RunId` and/or `-JointRunId` to name the intended session-folder IDs. Missing, incomplete, or incompatible models fail preflight rather than silently disappearing.

## Included experiments

- MQAR: lengths 64/128 crossed with 4/8/12 associations.
- Overwrite: baseline, eight keys, distance 24, and two/three/four rewrites.
- Channel checkpoints: original dynamic decay, fixed distinct channel means, and per-token channel averaging. Scalar checkpoints supply original-model references.
- Paired overwrite tests at three rewrites: change an obsolete initial value, change the latest value, or consistently rename the target key. Oracle labels and obsolete values are independently checked. Pair correctness is reported alongside output changes; identical wrong answers do not count as success.

Earlier gold answer tokens are erased. This is scored-query evaluation, not autoregressive generation. Discovery offset 700000 is used; offset 900000 remains untouched for future confirmation. Key/position swapping and duplicate-write controls, schedule replay, layer localization, mixed-rewrite training, and final confirmation are later stages of the plan and are not implemented by this launcher.

## Audits and output

Preflight checks completed training counters, full training recipe, source identities, paired data, and shared backbone initialization. Production reconstructs the 100000-example training pool, checks its original manifest, uses its exact prefix for static-gate fitting, and rejects overlap with training and earlier baseline development/confirmation examples. It does not prove disjointness from every possible historical diagnostic or run. Each active gate intervention is checked against the serial reference on actual examples before the full pass. Checkpoint parameters are verified unchanged.

- `REPORT.md`: completion/failure status and scope.
- `metrics.csv`: all per-model, per-cell metrics and evaluation times.
- `paired_metrics.csv`: correctness-aware counterfactual summaries.
- `contrasts.json`: per-seed prespecified dynamic/static contrasts. The practical thresholds are descriptive, not p-values; compare all three seeds before claiming a repeatable effect.
- `results.json`, `protocol.json`, `environment.json`: results, frozen selections/hashes, environment.
- `audits/`, `fitted_gates/`, `predictions/`: generator/overlap/numerical proofs, fitted constants, raw predictions and target/index mappings.

Results are saved after every cell in each worker's folder. Resume verifies unchanged configuration, source/checkpoint hashes, data identities, and raw-prediction hashes. Repeat the original worker count on resume; changing it or the batch size requires a new session. Sessions created before this source change cannot be resumed with the new evaluator. Console output gives total progress, and `worker_0.log`/`worker_1.log` contain detailed worker output. A worker failure stops its sibling and preserves completed cells for resume; consult `coordinator_status.json`. Partial artifacts remain in worker folders until successful final aggregation.

## Local verification and limits

The small CPU infrastructure check uses real default checkpoints, two examples per cell, two fitting examples, and the cached exact training prefix. It skips the production full-training-pool overlap audit and is not research evidence:

```powershell
.\run_A6000_decay_boundary.ps1 -CpuSmoke -Seeds 42
```

The production CUDA path fails explicitly if CUDA is unavailable. CPU tests cannot certify A6000 timing or CUDA execution; numerical checks also run on the actual device when you launch production. Training remains 16 epochs/3136 steps as saved; this evaluation changes no training settings. The joint recipe is an adapted comparison, not a claim to reproduce all DAMI architecture choices.
