## 1. Configuration & Dependency Updates

- [x] 1.1 Add `scipy>=1.11.0` to `pyproject.toml` dependencies.
- [x] 1.2 Add `RESUME_RUN_ID` and `SPLITS_DIR = "data/splits"` settings to `config.py`.

## 2. Per-Seed Dataset Splitting & Split Persistence

- [x] 2.1 Update `TinyStoriesDataset` in `dataset.py` to accept a `seed` parameter and apply `.shuffle(seed=seed, buffer_size=10000)`.
- [x] 2.2 Implement split manifest saver/loader `save_or_load_split_manifest(seed)` in `dataset.py` to write `data/splits/seed_<seed>_split.json`.
- [x] 2.3 Update `get_dataloader(split, batch_size, seed)` to retrieve seed-controlled dataloaders for training and validation.

## 3. Orchestrator Resume Logic & Git Exclusions

- [x] 3.1 Update `.gitignore` to add `*.pt` and `output/**/*.pt` to prevent committing model checkpoints exceeding GitHub's 100 MB limit.
- [x] 3.2 Update `run_all.py` to check for existing run directory when `RESUME_RUN_ID` is defined.
- [x] 3.3 Add completion verification helper in `run_all.py` to check if `seed<seed>_metrics.json` exists and contains valid metrics.
- [x] 3.4 Add skip logic in the grid loop of `run_all.py` so completed runs are restored from disk rather than re-trained.

## 4. Multi-Seed Statistical Hypothesis Testing Module

- [x] 4.1 Implement `compute_statistical_tests` function in `eval.py` using `scipy.stats` to execute Wilcoxon signed-rank tests and paired t-tests between Variant A and Variant B across seeds for each scale.
- [x] 4.2 Compute Cohen's $d$ effect sizes, mean differences, and 95% confidence intervals for Perplexity, Throughput, Needle Retrieval Accuracy, and MQAR Accuracy.
- [x] 4.3 Output structured `statistical_analysis.json` and generate `statistical_report.md` with significance indicators ($p < 0.05$).

## 5. Advanced Research Visualization Suite

- [x] 5.1 Update `generate_plots` in `run_all.py` to include standard deviation error bars (`yerr`) on all metric scaling charts.
- [x] 5.2 Add statistical significance markers ($*, **, \text{ns}$) to scaling charts based on calculated $p$-values.
- [x] 5.3 Implement step-wise loss convergence trajectory plotting (`loss_convergence_<scale>.png`) with shaded standard deviation error bands.

## 6. Verification

- [x] 6.1 Execute smoke test run with `seed=42`, verify `data/splits/seed_42_split.json` is created, check that both Variant A and B load the exact same data split, test run resumption, and confirm `statistical_analysis.json` and error-bar figures are generated properly.
