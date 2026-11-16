# Matched calibrated gating study

For exact parameter counts, model-size selection, and the distinction between
16/32/64/128-token decay targets and fitted offsets, see
[RESEARCH_METHODOLOGY.md](RESEARCH_METHODOLOGY.md).

The current clean-task experiments prove learnability, not a calibration benefit.
V3 suggests an initialization effect but has task shortcuts and a different
architecture. Neither replaces a calibrated comparison on the new recipe.

## Smallest useful exploratory comparison

Run overwrite, one seed, all four conditions for16epochs (3136steps). This tests
the interaction where the existing default models show a retention-speed gap.
The cosine schedule still has T_max64; do not compress it to16epochs.

```powershell
.\run_calibrated_recipe.ps1 -Python .\.venv\Scripts\python.exe -Stage explore -Family overwrite -Seeds 42 -Devices '0' -Publish
```

This launches scalar/channel default together, then scalar/channel calibrated
together: four models, approximately2hours on the single A6000 or a two-T4
session. Allow2.5..3hours for calibration and conservative margin. One seed can
reject broken calibration or show an interesting effect; it cannot establish
reproducibility. Prefer adding seed43 (eight models, about4hours total) before
committing to the full study. Use `-SeedCount 2 -StartSeed 42` instead of `-Seeds`.

Compare the normalized area under unrelated-retention accuracy over epochs1..16,
plus steps to three consecutive validations at95% overall and in all subgroups.
Do not judge worth solely by final near-perfect accuracy. An executable calibration
passes when its target errors are<=1e-6 and final three pilot validations exceed
80% overall/subgroups. An exploratory research signal is a reproducible paired
change in learning curves or scalar/channel contrast, not a significance claim
from two seeds. A null interaction is valid evidence; do not tune confirmation
data or silently change targets to manufacture a positive result.

## GPU qualification and full runs

Run the dedicated pilot on seed41 for both tasks, calibrated models:

```powershell
.\run_calibrated_recipe.ps1 -Python .\.venv\Scripts\python.exe -Stage pilot -Family mqar,overwrite -Seeds 41 -Initialization calibrated -Devices '0' -Publish
```

On Kaggle use `notebooks/kaggle_calibrated_recipe.py` as one cell. Set GPU T4x2,
Internet on, attach GH_TOKEN, set STAGE='pilot', FAMILIES=['mqar','overwrite'],
SEEDS=[41], INITIALIZATIONS=['calibrated'], then Save & Run All. This is about
two hours if tasks run consecutively, or about one hour in separate notebooks.
The runner repeats numerical backend checks and calibrated five-step optimizer
trajectory checks on the actual device. A full study requires matching successful
GPU pilot evidence for the selected tasks, hardware layout, PyTorch, and source.
CPU smoke evidence never authorizes a GPU study.

After qualification, a full study command can choose a task, seed list or count,
and initialization pair:

```powershell
.\run_calibrated_recipe.ps1 -Python .\.venv\Scripts\python.exe -Stage study -Family overwrite -Seeds 42 -Initialization calibrated -Devices '0' -QualificationRoot 'output/calibrated_recipe_study/matched-decay-v1/PILOT_RUN_ID' -Publish
```

Use `-Devices '0,1'` for two physical GPUs. `-Devices '0'` launches two workers
sharing one GPU. Family defaults to both; initialization defaults to both; seed
count defaults to one starting at42. Explicit seed lists and counts are mutually
exclusive. Use `-PlanOnly` to inspect selections without running them. No dependencies
are silently installed by the PowerShell launcher. Use the CUDA Python environment.

Full study defaults: seeds42,43,44; two tasks; four conditions;64epochs;12544steps;
LR .0021544346900318843; FP32; effective batch512/micro32;512development examples.
Flags let you select the seeds explicitly; pass `-SeedCount 3` for all three.
All models start fresh. The four-layer targets16/32/64/128 are identical for
scalar and every channel. Fit only existing dt_bias on the first256training
examples, eval-mode sequential layer passes, preserving RNG and other parameters.
Training retains the verified optimizer/weight decay and64epoch scheduler.
This is a same-backbone comparison, not exact parameter matching: scalar has
1,585,416 parameters and channel-wise1,651,456 (about4.2%more). Report that capacity
difference when interpreting a granularity effect.

Each invocation has an11hour workload ceiling. Queues that cannot fit are listed
as `deferred` in session_status.json, not declared trained. Launch those conditions
in a fresh session with flags; there is no implicit checkpoint resume. The Kaggle
wrapper also has an11.5hour outer timeout. Step50 speed guards reject a run whose
projected remaining training cannot fit its budget. GPU calibration timing is
still unmeasured; all existing runtime estimates remain provisional for calibrated
models.

## Per-run folders, checkpoints, and collection

Local master: `output/calibrated_recipe_study/matched-decay-v1/<session-id>/`.
Inside: `<family>/seed_<seed>/<initialization>/<variant>/`, with settings,
calibration, data audits/manifests, epoch curves, summary and latest_checkpoint.pt.
Checkpoints are written atomically each epoch; latest replaces previous. Logs,
protocol, numerical proofs and pair status are retained. On failure a surviving
checkpoint is partial evidence, not a completed model.

Publication is enabled by default (`-Publish:$false` disables it; CPU smoke never
publishes). Each model pushes to its own orphan result branch:
`research/calibrated-recipe/matched-decay-v1/<family>/<seed>/<initialization>/<session-id>/<variant>`.
It includes checkpoints and hash manifests. The branch has a common folder:
`results/calibrated_recipe_study/matched-decay-v1/<family>/seed_<seed>/<initialization>/<session-id>/<variant>`.
Session IDs prevent overwrite across notebooks. Publication happens after each
two-worker pair exits, including partial artifacts on failure. A push failure
fails the session; raw local output remains available. Each file must be below90MiB.
For Kaggle use GH_TOKEN; Windows can use existing Git credentials.

Collect all independent result branches, including checkpoints, without merging:

```powershell
python recipe_study_artifacts.py --output research_results/aggregated_recipe_study
```

Existing conflicting files are refused. The collector writes aggregate_index.json
and preserves the common master folder. Do not average pilot, explore and study
runs together; the index records each stage and identity.

After all selected study models finish, evaluate the frozen confirmation matrix:

```powershell
python confirm_recipe_study.py --root research_results/aggregated_recipe_study --output research_results/calibrated_confirmation --family mqar overwrite --seeds 42 43 44
```

It refuses incomplete/duplicate conditions (use --run-id to disambiguate repeats),
checks checkpoints/source/dataset identities, and evaluates512examples per task
and seed with offset400000. It checks overlap with train,development,and previous
confirmation. No confirmation is generated by training. Final reports include
paired per-seed AULC contrasts and the initialization-by-granularity interaction.
With three seeds report descriptive results; do not treat queries as independent
training replicates or infer significance from saturated final accuracy.

## Measured-budget estimates

| Work per seed | Two T4s, two workers | One A6000-12Q, two workers |
|---|---:|---:|
| One task, one initialization,64epochs:2models |3h54m..3h57m|about3h43m|
| One task, both initializations:4models |about7h50m|about7h25m|
| Both tasks, all four conditions:8models |about15h42m|about14h51m|
| Overwrite exploratory16epochs,4models |about1h59m|about1h51m|

These sum sequential pair sessions and include measured baseline setup/validation/
checkpoint times, but not new calibration overhead, dependency installation, or
uploads. Allow about20% plus setup/publication headroom; a complete seed must be
split across Kaggle sessions. Three full seeds total about47two-T4 session-hours
or44.5A6000hours, before extra overhead. Four simultaneous independent Kaggle
sessions could finish one seed in about4hours wall-clock if quota permits.

For plumbing checks only:
`./run_calibrated_recipe.ps1 -Python ./.venv/Scripts/python.exe -CpuSmoke -Stage explore`.
CPU smoke shrinks data/model and runs2epochs; it is never learning or GPU evidence.
