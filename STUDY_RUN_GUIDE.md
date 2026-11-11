# Running the granularity and initialization study

The current study protocol and task matrix live in [`RESEARCH_IMPLEMENTATION_PLAN.md`](RESEARCH_IMPLEMENTATION_PLAN.md) and [`study_configs/granularity_init_study.json`](study_configs/granularity_init_study.json). The launcher writes to `output/granularity_init_study` by default, keeping the historical `output/full_research` runs separate.

## Commands

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
