# Research implementation plan: gate granularity, initialization, and memory behavior

## 1. Research question and scope

**Main question:** How much independent decay control is useful when initial memory timescales are controlled, and does the answer differ between retaining associations and overwriting them?

The contribution is a controlled study of the interaction between gate granularity and initialization. It is not the invention of grouped gates or geometric initialization, and it is not the first controlled factorial study of gate granularity: see Das, [“Delta-Rule Writes in Linear Attention: A Controlled Factorial Study of Decay Design”](https://tmlr.infinite-conf.org/paper_pages/FdXB4u27LD.html). Discuss that study alongside [DAMI](https://arxiv.org/abs/2610.41994), which studies dual-axis memory initialization, and [Systolic Linear Attention](https://arxiv.org/abs/2611.06828), which groups gates as part of a TPU hardware/kernel co-design.

Use 10M as the main experimental scale. Use 30M for confirmation of the central comparison. Keep TinyStories, MQAR, and overwrite recall. Do not add FineWeb-Edu, layer-allocation sweeps, head-dimension sweeps, or GDN-2 to the core study.

This document is an implementation plan. The new grouped training and initialization options do not yet exist in the current CLI. The PowerShell launcher described below is a deliverable of implementation, rather than a command that can already run the new experiments.

## 2. Shared experimental rules

- Primary seeds: 42, 43, 44, 45, 46, paired across conditions.
- Train each dataset family from a fresh checkpoint. Synthetic training is not TinyStories fine-tuning.
- Keep depth, width, attention layout, state dimensions, convolution, tokenizer, data, optimizer, and token budget fixed within each comparison. Record actual parameter counts, including gate parameters.
- Granularity is the number of independent decay outputs per head: scalar = 1, grouped-4 = 4, channel-wise = key dimension. Both current scales have key dimension 40, so four groups contain ten channels each.
- Retain the current backbone layout for all arms and report which layers use recurrent attention and softmax. Do not silently switch to a pure recurrent model between conditions.
- Use separate training, validation, and final test seeds/splits. Select recipes using validation only; do not select conditions using final test results.
- Use a new output root, `output/granularity_init_study`, with distinct identities for experiment, family, size, granularity, initialization, recipe, and seed. Preserve existing `output/full_research` artifacts as historical baselines.
- Record validation loss every 100 optimizer steps and at the final step. Persist history immediately. Use the same fixed validation subset across paired conditions; record its batch count and sampling seed.
- Keep cached decode latency, throughput, state memory, short prefill/TTFT measurements, cache equivalence, CPU/GPU usage, memory, and available energy measurements. Full-prefix performance benchmarking remains disabled; the short correctness comparison remains enabled.
- Default execution is sequential. Optional parallel execution uses separate processes, unique run directories, and recorded concurrency. Benchmark inference in isolation to avoid interference.

## 3. Numbered experiments

### Experiment E0 — Correctness and synthetic-task learnability

**Purpose:** Establish that the implementation and training recipe can learn a simple task before interpreting capacity or gate effects.

1. Validate loss masking, causal target alignment, train/test separation, and vocabulary IDs.
2. In generator V3, `query_distances` sets an anchor distance: for MQAR, the first query's anchor key is the final association in the randomized memory block; for overwrite, the anchor is the latest-value query. Record each answer's query-marker position, latest relevant value position, and measured distance. MQAR still queries every key in randomized order, so later answers can have different actual distances; use `AnswerPosition.distance` for per-answer analysis and joint key-count-by-distance results. For overwrite, also record obsolete target writes, the latest correct write, and the separate unrelated-association answer distance.
3. Generate examples that actually satisfy the requested distance. Reject impossible length/key/distance combinations instead of silently relabeling them.
4. Check overwrite answers against an independent last-write-wins oracle and MQAR answers against a lookup oracle.
5. Check grouped endpoints against scalar and channel-wise implementations with mapped weights; check cache equivalence and finite gradients.
6. Train short 10M scalar and channel-wise runs on easy MQAR and overwrite: start with 4 keys at length 128 and feasible short distances. Use seed 42 initially.
7. Use validation answer accuracy, rather than overall token accuracy, as the learnability measure. Proposed acceptance criterion: at least 90% answer accuracy for two consecutive validations on the easy task. Record and freeze the threshold before running E0; it is an engineering criterion, not a literature claim.

**Decision:** If either endpoint cannot learn, diagnose the task/recipe before starting the main grid. Adding more model sizes is not the remedy.

**Artifacts:** generator audit, tiny-example traces, endpoint/cache checks, validation curves, and a machine-readable E0 pass/fail result.

### Experiment E1 — Recipe and runtime qualification

**Purpose:** Select an affordable, functioning protocol without spending the main seed budget on exploratory runs.

- Use seed 42 and validation data only. Screen scalar and channel-wise endpoints under equal tuning effort.
- Start with the existing learning rate plus 1e-3 and 3e-3. Try 1e-2 only as a recorded extension if this range fails; do not assume it is safe or necessary.
- For synthetic tasks, compare direct difficulty training with a simple curriculum only if direct training fails. Specify stages, stage lengths, and the final difficulty distribution. Apply the selected recipe to every granularity/initialization arm.
- Time real training, validation every 100 steps, checkpoint writes, and evaluation separately. Include actual initialization calibration cost.
- Use a provisional LM budget of 100M tokens at 10M and 300M at 30M, subject to the measured resource budget. At batch 8 × accumulation 4 × length 512, these correspond to 6,104 and 18,311 optimizer steps after rounding up. Save actual processed tokens and handle packing/padding explicitly.
- These token targets are planning values, not a guarantee of convergence. Inspect training/validation curves and choose a fixed final budget before E2. Report results as fixed-budget comparisons if convergence is not established.
- Choose synthetic budgets from successful learning curves and performance on the final difficulty distribution. Do not convert the LM token budget into an automatic synthetic epoch requirement.
- If parallel execution is requested, qualify the requested worker count using separate production-style Python processes and the intended storage location. Report total throughput, per-run slowdown, GPU memory, CPU load, and input wait. High CPU utilization alone does not prove CPU throttling.

**Selection rule:** Select a shared recipe from the aggregate validation performance of the endpoints, with deterministic tie-breaking and recorded failures. Report that the main interaction is conditional on this shared recipe. If an arm later fails, use a separately labeled sensitivity check rather than silently retuning that arm.

**Artifacts:** `recipe_selection.json`, `training_time_estimate.md`, and `protocol_lock.json` containing all final budgets, calibration settings, validation subsets, and evaluation grids.

**Decision:** Continue automatically only after E0 passes and a valid protocol lock exists. Support a stop-after-E1 option so the budget can be reviewed before the expensive stages.

### Experiment E2 — Main 10M granularity × initialization factorial

**Purpose:** Measure whether the benefit of finer gates depends on initialization.

| Condition | Granularity | Initialization |
|---|---|---|
| C1 | Scalar, one gate per head | Existing default |
| C2 | Grouped-4, four gates per head | Documented default extension |
| C3 | Channel-wise | Existing default |
| C4 | Scalar | Calibrated geometric timescales across heads |
| C5 | Grouped-4 | Same calibrated head timescales; initially matched within each head |
| C6 | Channel-wise | Same calibrated head timescales; initially matched within each head |

For C4–C6, use the same calibration examples and target half-life range. Choose that range from the training context/distance protocol, not test performance. Calibrate realized decay behavior and input sensitivity, rather than merely assigning equal bias values. Save measured initial gate distributions and calibration residuals.

The matched initialization should make within-head initial gate behavior comparable while allowing grouped/channel-wise gates to learn different responses. Check whether symmetry persists during training. If a perturbation is needed, specify and freeze it before the main grid and measure how it changes the matched initial behavior. Do not label equal biases as equal gate functions.

The default extension for grouped-4 must preserve the existing parameterization's scaling and initialization rule, and document any differences in realized gate statistics.

**Run count:** 6 conditions × 5 seeds × 3 families = **90 fresh training runs**.

**Primary endpoints:** TinyStories validation/test loss; MQAR and overwrite answer accuracy on prespecified in-distribution grids. Report the change in the scalar/channel-wise gap between the two initialization settings as the main interaction. Use grouped-4 to locate an intermediate point.

**Interpretation:** A smaller gap after calibration supports an initialization-dependent advantage under this protocol. It does not prove that timescale diversity alone explains all differences.

### Experiment E3 — Retention and overwrite capacity curves

**Purpose:** Explain the E2 results using memory demand, without another full training sweep.

- Reuse all E2 checkpoints.
- Evaluate answer accuracy across key counts, actual query distances, and sequence lengths; for overwrite, also vary number of rewrites.
- Start from the current ranges, add easy 4-key cases, and retain only feasible combinations. Freeze the final grid in E1.
- Clearly distinguish training-distribution settings from extrapolation settings.
- Report per-seed curves, aggregate uncertainty, and accuracy on the obsolete value in overwrite tests.
- Define a capacity summary before evaluation, such as the largest tested key count achieving 90% accuracy at a fixed distance. If no setting meets the threshold, report that explicitly; do not extrapolate beyond measured points.
- Keep head dimensions fixed. This measures behavioral capacity under the chosen state size, not a universal memory-capacity law.

**Additional training:** None.

### Experiment E4 — Post-training grouping and scalarization

**Purpose:** Test how strongly trained models rely on differences between channel gates.

- Reuse channel-wise checkpoints from both E2 initialization settings.
- Evaluate the original gates, then impose 8, 4, 2, and 1 shared gate groups per head at inference.
- Explicitly specify whether averaging occurs over effective retention factors or log-decay values; these are different interventions. Use the existing convention only after documenting it, and include one alternative as a bounded sensitivity check if the conclusion depends on the convention.
- Use a fixed held-out subset for initial intervention screening; evaluate prespecified primary comparisons on the final evaluation grid.
- Report changes relative to each checkpoint's unmodified result, paired across seeds.
- Save channel diversity, realized decay/half-life summaries, saturation, and change from initialization alongside performance. Explain that decay-only half-life is a diagnostic; delta writes also affect memory retention.
- Keep the model and gate projection computation intact. An inference intervention does not establish the runtime savings or quality of a model trained with grouped gates.

**Additional training:** None. Existing historical checkpoints may receive this analysis but must be labeled as the old training protocol.

### Experiment E5 — Targeted 30M confirmation

**Purpose:** Check whether the principal finding persists beyond 10M.

- Prespecify scalar/channel-wise × default/calibrated initialization: C1, C3, C4, C6.
- Use seeds 42, 43, 44 on all three families, with the locked 30M budgets.
- Repeat E3 and the primary E4 interventions on these checkpoints.
- Do not select the 30M conditions based on which 10M seed produced the largest effect.

**Run count:** 4 conditions × 3 seeds × 3 families = **36 fresh training runs**.

**Combined core count:** E2 + E5 = **126 training runs**, plus E0/E1 qualification. Evaluation interventions do not add training runs. If the pilot shows this exceeds budget, reduce the scope before E2 and record the revision; do not reduce seeds after inspecting final results.

### Experiment E6 — Statistical analysis and paper outputs

**Purpose:** Translate the measurements into defensible claims.

- Report every seed, failures, exclusions, training tokens, actual model parameters, and wall time.
- Report paired effect sizes and uncertainty. For synthetic tasks, distinguish variation across examples from variation across training seeds; do not treat thousands of test examples as thousands of independent trained models.
- Make the granularity × initialization interaction the primary comparison. Label other slices and correlations as secondary or exploratory, and account for multiple comparisons where appropriate.
- Include learning curves versus tokens and wall time, granularity/quality curves, recall curves, overwrite error breakdowns, intervention effects, and isolated cached inference efficiency.
- Avoid a superiority claim when both models fail the task or differences are within seed variability.
- Keep aggregate reports compact: summary statistics in the top-level JSON, raw examples and large diagnostic arrays in separately referenced artifacts.

## 4. Implementation work packages

1. **Configuration and identity:** Add independent granularity, initialization, calibration, recipe, and experiment-ID fields. Include them in manifests and run paths. Keep compatibility checks meaningful and evaluation of old checkpoints supported.
2. **Grouped gate model:** Implement independently trained group outputs and broadcasting, retaining the existing scalar/channel-wise endpoints and recurrent state shape. Verify support explicitly for each recurrence backend; reject unsupported combinations.
3. **Initialization:** Implement calibrated geometric head timescales and matched within-head initialization. Store calibration data provenance, targets, realized behavior, and residuals.
4. **Synthetic data:** Implement audited query placement, oracle checks, explicit difficulty settings, and optional curriculum. Version the generator and label old/new results separately.
5. **Training controls:** Add configurable learning rate, budgets, validation interval/subset, curriculum, and calibration options to the CLI or a validated study-config file. Preserve the new immediate validation-history writes.
6. **Evaluation:** Implement E3/E4 evaluation plans, fixed subsets, overwrite error categories, and compact artifacts.
7. **Study orchestrator:** Create a Python study entry point that expands the locked matrix, executes individual existing training/evaluation operations, and records task status. Implement preflight, dry-run, lifecycle-aware skipping, retries, and protocol locks.
8. **PowerShell launcher:** Create `run_granularity_init_study.ps1` as specified below.
9. **Verification:** Check generator oracles, grouped endpoints, initialization behavior, gradients, cache equivalence, interrupted/restarted execution, and worker isolation. Run a CPU smoke workflow and a short GPU production workflow before launching the research grid.
10. **Analysis/reporting:** Implement E6 tables and figures, update the proposal and related work, and link the actual protocol lock in the final report.

## 5. PowerShell launcher specification

**Deliverable:** A new root-level `run_granularity_init_study.ps1`. Keep the existing launcher available for historical runs.

Proposed usage after implementation:

```powershell
# Print expanded tasks, run counts, budgets, and output paths without training.
.\run_granularity_init_study.ps1 -DryRun

# Run qualification and create the measured budget/protocol documents.
.\run_granularity_init_study.ps1 -StopAfter E1

# Run E0 through E6 back to back, reusing completed compatible work.
.\run_granularity_init_study.ps1

# Use two training processes after production-style qualification.
.\run_granularity_init_study.ps1 -TrainingWorkers 2

# Resume at a selected stage using its existing locked protocol.
.\run_granularity_init_study.ps1 -StartAt E2

# Run evaluation/analysis from existing study checkpoints.
.\run_granularity_init_study.ps1 -EvaluateOnly
```

Parameters: `-OutputRoot`, `-Python`, `-TrainingWorkers` (default 1), `-StartAt`, `-StopAfter`, `-EvaluateOnly`, `-DryRun`, and a validated `-StudyConfig` path. The study config owns conditions, model sizes, per-size seed lists, datasets, and budgets. Overrides must be recorded and checked against the protocol lock.

Execution order:

1. Preflight interpreter, dependencies, data availability, hardware, backend support, configuration, and artifact paths. Print the entire intended grid and estimated cost.
2. E0, then E1; create or validate the protocol lock. Stop if correctness or learnability fails.
3. E2: train **all selected 10M TinyStories conditions/seeds**, wait for every worker, then evaluate that entire family. Repeat for MQAR, then overwrite.
4. Run E3 and E4 from the completed 10M checkpoints, without retraining.
5. E5: train **all selected 30M TinyStories conditions/seeds**, then evaluate that family. Repeat for MQAR, then overwrite, including its E3/E4 analyses.
6. E6: create the combined report and completion summary.

Operational requirements:

- Use process exit codes reliably on Windows. Print queued, started, completed, skipped, and failed events with experiment/family/size/condition/seed identifiers.
- Stream progress to the console and save one master log plus per-task logs. Include training and validation steps, evaluation buckets, and intervention progress.
- Skip training only after validating a compatible completed checkpoint and sufficient completed optimizer steps. A directory's existence is insufficient. Preserve completed training if evaluation fails, and retry evaluation independently.
- On any failure, stop launching new tasks, handle active workers cleanly, preserve checkpoints, record the original error, and return a nonzero exit code. Do not mark an experiment complete after partial execution.
- On restart, skip completed tasks and retry failed evaluation from valid trained checkpoints. Interrupted training may resume only if a genuine resumable checkpoint includes optimizer, scheduler, scaler, RNG, and data/curriculum position; otherwise report that limitation explicitly.
- Use separate Python processes for parallel training, unique task outputs, and a bounded worker queue. Training data may intentionally overlap across paired experimental conditions; this is required for control. Prevent train/test leakage and unintended worker-state sharing, not legitimate reuse of the same training split.
- Run evaluation benchmarks sequentially after the family training barrier. Record actual concurrency/resource telemetry for training.
- `-EvaluateOnly` must never start missing training. List missing checkpoints and exit with a clear incomplete status.
- `-DryRun` must work before GPU allocation and must expose every expanded command/configuration.

## 6. Completion criteria

Implementation is ready when the new CLI/config, grouped model, calibrated initialization, audited generators, study orchestrator, PowerShell launcher, restart behavior, and smoke/short-GPU checks work together.

The study is complete only when all selected tasks and final evaluations succeed, the seed-level results are available, and E6 reports both findings and limitations. A working pilot demonstrates a functioning short path; it does not guarantee that a long run will never encounter resource or numerical failures.

No novelty score or positive result is promised. The intended contribution is a well-controlled answer to the initialization–granularity question, including a credible null result if that is what the measurements support.
