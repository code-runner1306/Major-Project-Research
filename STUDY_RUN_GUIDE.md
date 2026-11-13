# Running the granularity and initialization study

The current study protocol and task matrix live in [`RESEARCH_IMPLEMENTATION_PLAN.md`](RESEARCH_IMPLEMENTATION_PLAN.md) and [`study_configs/granularity_init_study.json`](study_configs/granularity_init_study.json). The launcher writes to `output/granularity_init_study` by default, keeping the historical `output/full_research` runs separate.

## Commands

### Two-day exploratory subset

The PowerShell launcher defaults to paired seeds 42 through 46. To extend the
completed seed-42 experiment, keep `-OutputRoot output\granularity_exp_1`;
validated seed-42 tasks are skipped and only missing seeds are run.

For a two-worker CPU plumbing smoke, use `-Family both -SmokeTest -Workers 2`
with a fresh output folder. To check two workers on the VM's actual GPU and
production backbone, use the capacity pilot instead:

```powershell
powershell.exe .\run_48h_exploratory_study.ps1 -Family both -QualifyWorkers 2 -QualificationSteps 20 -QualificationRoot output\workers2_pilot_1 -Python .\venv\Scripts\python.exe
```

This runs all eight family/condition combinations in waves of up to two workers,
then evaluates those pilot checkpoints. It uses separate pilot seeds, 20 optimizer
steps per task, and 64 validation/evaluation examples, keeping research artifacts
separate. It records runtime, VRAM, utilization, and competing GPU processes. If
two workers fail qualification, it tries one worker and reports the qualified
limit; inspect `qualified_workers` in `qualification_report.json`. A short pilot
checks capacity and execution, not long-run stability or a guaranteed speedup.
Use a new qualification root for each attempt. A longer pilot can be requested
with `-QualificationSteps 100`.

If the report qualifies two workers, continue the experiment with:

```powershell
powershell.exe .\run_48h_exploratory_study.ps1 -Family both -Workers 2 -QualificationReport output\workers2_pilot_1\qualification_report.json -OutputRoot output\granularity_exp_1 -Python .\venv\Scripts\python.exe
```

If you have only about two days, run one family on each machine using the separate exploratory launcher. Each run covers **10M**, the four endpoint conditions (scalar/channel-wise crossed with default/calibrated initialization), **five paired seeds by default**, and **1,000 optimizer steps per seed**. All four runs use fixed learning rate **3e-3**, a literature-informed shared choice that is **not tuned on these runs**. Each family trains all four conditions before evaluating them. Synthetic validation and held-out evaluation are capped at **512 examples per run**; training still uses 50,000 examples. `-Family` is required and restricts each machine to one family. Without `-OutputRoot`, it selects a family-specific default folder.

Machine A: MQAR

```powershell
.\run_48h_exploratory_study.ps1 -Family mqar -DryRun
.\run_48h_exploratory_study.ps1 -Family mqar -Python .\.venv\Scripts\python.exe
```

Machine B: overwrite recall

```powershell
.\run_48h_exploratory_study.ps1 -Family overwrite -DryRun
.\run_48h_exploratory_study.ps1 -Family overwrite -Python .\.venv\Scripts\python.exe
```

Their default output folders are `output/granularity_48h_exploratory_lr3e-3_mqar` and `output/granularity_48h_exploratory_lr3e-3_overwrite`, so the VMs do not write into the same artifacts. After both finish, copy both complete folders to the analysis machine, preserving the folder names and contents. Each has its own `run_summary.json`, run artifacts, evaluations, and logs; keep the family labels when comparing or combining the results. If you pass a custom `-OutputRoot`, use a different one on each VM.

To exercise the PowerShell launcher, including all eight training and evaluation tasks, run `powershell.exe -ExecutionPolicy Bypass -File .\run_48h_exploratory_study.ps1 -Family both -SmokeTest -Python .\.venv\Scripts\python.exe`. This is a two-step tiny CPU smoke test, not a CUDA or research qualification. Use a fresh `-OutputRoot` to repeat it. For the 1,000-step PowerShell production run, intermediate validation is at steps 300 and 650, followed by final validation after training and held-out evaluation. If rerunning after a failed or older experiment, choose a fresh `-OutputRoot`: partial training cannot resume, and the changed validation schedule is part of the artifact identity.

The production script checks that PyTorch imports and CUDA is available, writes a master log plus per-task logs, and validates completed checkpoints before reusing them. It stops rather than overwriting a partial or incompatible run directory. If an evaluation failed after training, retry only evaluation on that machine with `-Family mqar -EvaluateOnly` or `-Family overwrite -EvaluateOnly`; interrupted training is not resumable because the pipeline does not save optimizer state. The **9 h 20 min per family** figure is only an old-timing baseline extrapolated from prior 10M TinyStories runs. It covers training only and excludes validation, calibration, and evaluation, so finishing within 48 hours is **not guaranteed**. This one-seed, 1,000-step subset is exploratory evidence, **not** the full multi-seed, 100M/300M-token study or its E0/E1 qualification.

### Linux and Kaggle launcher

The Bash launcher runs the same family-specific matrix and writes the same output structure. It requires `--family mqar` or `--family overwrite`; task JSON is created with Python's standard library, so `jq` is not needed. In a Linux checkout with the virtual environment activated:

```bash
bash ./run_48h_exploratory_study.sh --family mqar --dry-run
bash ./run_48h_exploratory_study.sh --family mqar --python ./.venv/bin/python
```

On the other Linux VM, run overwrite recall instead:

```bash
bash ./run_48h_exploratory_study.sh --family overwrite --dry-run
bash ./run_48h_exploratory_study.sh --family overwrite --python ./.venv/bin/python
```

In Kaggle, the notebook input area is read-only. Put the checkout and environment in a writable location when possible, or send output explicitly to `/kaggle/working`:

```bash
bash /kaggle/working/Major-Project-Research/run_48h_exploratory_study.sh \
  --family mqar \
  --python /kaggle/working/Major-Project-Research/.venv/bin/python \
  --output-root /kaggle/working/granularity_48h_exploratory_lr3e-3_mqar
```

Use `--family overwrite` and a matching `_overwrite` output path on the other VM. Kaggle sessions can expire or reset, and the training code cannot resume an interrupted optimizer run. Completed checkpoints persist only if they remain in `/kaggle/working` and are downloaded or copied before the session storage is cleared. Preserve the two family-specific output roots when moving both result sets to one analysis machine; they contain matching `run_summary.json`, checkpoint, evaluation, and log structures.

Run a dry run first to inspect the expanded tasks and paths without starting training:

```powershell
.\run_granularity_init_study.ps1 -DryRun
```

Run E0 and E1 qualification, then stop with the measured recipe and protocol lock recorded:

```powershell
.\run_granularity_init_study.ps1 -StopAfter E1
```

Run the full study after E0/E1 qualification passes. Qualification includes training pilots and may take substantial time; do not assume it will finish within a short smoke-test window:

```powershell
.\run_granularity_init_study.ps1
```

After a successful `-StopAfter E1` run, use `-StartAt E2` to continue directly to the main grid without rerunning qualification. Starting from E0 may run qualification helpers again; compatible completed work is reused where supported. Evaluate existing checkpoints without starting missing training with `-EvaluateOnly`:

```powershell
.\run_granularity_init_study.ps1 -StartAt E2
.\run_granularity_init_study.ps1 -StartAt E2 -EvaluateOnly
```

The default training worker count is one. Increase it only after the production-style worker qualification succeeds and its settings are recorded in the protocol lock:

```powershell
.\run_granularity_init_study.ps1 -TrainingWorkers 2
```

Use a separate output root for a reduced or revised experiment:

```powershell
.\run_granularity_init_study.ps1 `
  -StudyConfig study_configs/granularity_init_study.json `
  -OutputRoot output/granularity_init_study_revision
```

## Study configuration

Edit a copy of `study_configs/granularity_init_study.json` to change the experiment matrix. The `datasets` field selects TinyStories, MQAR, and overwrite recall. `conditions` defines gate granularity and initialization. `sizes`, per-size `seeds`, and `budgets` define training runs; `retrieval` defines the synthetic generator version and difficulty grids; `evaluation` and `qualification` define the evaluation subset and E0/E1 criteria.

Keep the same config and output root when resuming a locked protocol. A change to conditions, seeds, budgets, or data settings is a new protocol: use a new config identity and output root so existing manifests remain intact. Do not remove old run manifests to make a changed config fit an existing run.

The reduced plumbing profile is `study_configs/granularity_init_smoke.json`. It is explicitly marked smoke-only and does not establish synthetic-task learnability or authorize the production grid.

## Synthetic data and query distances

The new study selects `mqar-overwrite-v3`; legacy profiles keep `mqar-overwrite-v2` unless they explicitly select another version. V3 uses strict difficulty settings and rejects infeasible length, key, distance, or rewrite combinations instead of clamping them.

The configured `query_distances` value is an anchor. For MQAR it is the token-position gap from the anchor key's latest value to the first query marker. The generator answers every key in a randomized query order; each answer records its own query marker position, latest relevant value position, and measured distance. Later answers can have different distances, so report accuracy using the per-answer distance buckets and the joint key-count-by-distance table. For overwrite, the configured distance anchors the latest-value query; the example also asks for an unrelated association, whose measured distance is recorded separately. For distances of at least three tokens, an unrelated write follows the latest target write before the anchor query, so the target cannot be answered by copying the globally newest value.

For the current V3 layout, the minimum sequence length is `5 * key_count + query_distance - 1` for MQAR. For overwrite with more than one key and distance at least three, it is `2 * (key_count + overwrite_count) + query_distance + 3`. The shipped difficulty grid fits within its shortest 128-token sequence. Recheck feasibility before changing the grid; the generator reports any impossible combination.

V2 and V3 runs have different generator identities and answer structures. Keep them labeled separately when comparing historical results with the new study.
