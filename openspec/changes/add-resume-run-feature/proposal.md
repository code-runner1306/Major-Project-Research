## Why

Currently, running the 24-configuration Gated DeltaNet grid search via `run_all.py` generates a random run UUID on each execution and starts all configurations from step 1 without checking for existing progress. If an experiment is interrupted mid-way, the entire multi-hour grid search must be restarted from the beginning, resulting in lost compute time.

Additionally, model checkpoints (`.pt` files) can reach up to ~388 MB for 100M parameter models and total over ~4.4 GB across all runs, which exceeds GitHub's 100 MB single-file limit and blocks standard git pushes.

Furthermore, while the pipeline evaluates models across 3 seeds per configuration, the dataset loader currently streams the dataset in a static order without seed-specific deterministic shuffling. To ensure valid statistical comparison, each seed must use a distinct, reproducible dataset split, and all model scales and variants evaluated under the same seed must train on the **exact same dataset split**. Persisting these per-seed dataset split configurations ensures that even when resuming an interrupted run, models load the identical dataset split for their respective seed.

Adding run resumption, deterministic per-seed dataset split persistence, strict `.gitignore` rules for checkpoint weights, multi-seed statistical significance testing (**Wilcoxon signed-rank tests**), and publication-ready visualization graphs will make the repository robust, scientifically controlled, lightweight for Git, and fully reproducible.

## What Changes

* **Deterministic / Fixed Output Directory Option**: Add support for specifying a fixed output run directory name (or run ID) so progress is stored and tracked predictably across restarts.
* **Seed & Scale Progress Recovery**: Modify `run_all.py` to check for completed seed/scale metric outputs (`seed<seed>_metrics.json` or `aggregate_metrics.json`) before executing training, skipping already completed runs.
* **Deterministic Per-Seed Dataset Splitting & Persistence**: Update `dataset.py` to accept a `seed` parameter, creating distinct, deterministic dataset shuffles for each seed (e.g. seeds 42, 43, 44). Ensure all models (10M, 30M, 55M, 100M for both Variant A & B) assigned to seed $S$ train on the **identical dataset split for seed $S$**. Save per-seed split manifests (`seed_<seed>_split.json`) to disk so resumed runs load the exact same data stream.
* **Git Exclusions for Checkpoints**: Update `.gitignore` to exclude `.pt` and `.ckpt` files from git tracking to prevent push failures due to GitHub's 100 MB per-file limit.
* **Rigorous Multi-Seed Statistical Hypothesis Testing**: Integrate `scipy.stats` to execute non-parametric (Wilcoxon signed-rank test, Mann-Whitney U) and parametric (paired t-test) hypothesis tests comparing Scalar Gating (Variant A) vs. Channel-wise Gating (Variant B) across parameter scales. Compute $p$-values, effect sizes (Cohen's $d$), and 95% confidence intervals.
* **Enhanced Research Visualization Suite**: Expand matplotlib figure generation to include error bars (95% CI / standard deviation), statistical significance annotations ($p < 0.05$), loss convergence curves across training steps, and memory/throughput scaling comparisons.

## Capabilities

### New Capabilities
- `run-resumption`: Support resuming interrupted grid-search runs, skipping finished seed/scale combinations, and managing checkpoint lifecycle safely without exceeding Git size limits.
- `deterministic-data-splits`: Generate, persist (`seed_<seed>_split.json`), and enforce reproducible per-seed dataset splits such that all model scales and variants sharing a seed train on the identical data stream across runs and resumptions.
- `statistical-hypothesis-testing`: Compute Wilcoxon signed-rank tests, $p$-values, confidence intervals, and effect sizes comparing Gating Variant A vs Variant B across multi-seed evaluations.
- `advanced-research-visualization`: Generate publication-grade plots with error bars, statistical significance markers ($*, **, \text{ns}$), and step-wise loss trajectories.

### Modified Capabilities
*None*

## Impact

* **Files Affected**:
  * `dataset.py`: Add seed-based dataset shuffling (`dataset.shuffle(seed=seed, buffer_size=...)`) and per-seed split manifest creation/loading.
  * `pyproject.toml`: Add `scipy>=1.11.0` dependency for statistical tests.
  * `config.py`: Add resume configuration toggles and dataset split directory settings.
  * `run_all.py`: Pass seed parameter to dataloaders, add progress recovery, statistical testing invocation, and error-bar plotting.
  * `train.py`: Receive seed-initialized dataloaders.
  * `.gitignore`: Exclude `.pt` model checkpoint files.
* **APIs / Dependencies**: Requires `scipy` for statistical calculations.
