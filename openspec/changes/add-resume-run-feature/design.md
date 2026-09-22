## Context

The current benchmark orchestrator in `run_all.py` executes 24 model training and evaluation runs sequentially. When running in full production mode (`SMOKE_TEST = False`), this process can take several hours to complete on GPUs (or days on CPU). If a run is interrupted, all progress in the current execution folder is lost because `run_all.py` creates a brand new `run_<uuid>` directory each time.

Furthermore, model checkpoints `.pt` files range from 40 MB (10M scale) to ~388 MB (100M scale). A full 24-run suite generates ~4.4 GB of checkpoint weights, exceeding GitHub's single-file limit of 100 MB.

Finally, scientifically valid comparison across model architectures requires that each random seed corresponds to a distinct, controlled dataset split. All model scales (10M–100M) and gating variants (Scalar vs Channel-wise) evaluated under seed $S$ must train on the exact same token stream for seed $S$. These per-seed dataset split configurations must be persisted to disk so that resumed runs reproduce the exact data stream.

## Goals / Non-Goals

**Goals:**
* Provide run directory persistence and configurable target run IDs so output folders can be targeted predictably across restarts.
* Implement automatic checking and skipping of previously completed seed/scale configurations in `run_all.py`.
* Exclude `.pt` files in `.gitignore` while keeping lightweight JSON and PNG metric files tracked in Git.
* Implement deterministic per-seed dataset shuffling in `dataset.py` with `seed` parameters and save dataset split manifests (`data/splits/seed_<seed>_split.json`) for resumption.
* Integrate `scipy.stats` for Wilcoxon signed-rank tests, paired t-tests, $p$-value calculations, Cohen's $d$ effect sizes, and 95% confidence intervals across multi-seed evaluations.
* Generate enhanced publication-grade matplotlib plots featuring error bars, statistical significance markers ($*, **, \text{ns}$), and loss convergence band charts.

**Non-Goals:**
* Implementing partial within-epoch micro-step checkpointing for incomplete steps inside `train.py` (seed-level granular resumption is sufficient and simple to maintain).

## Decisions

### Decision 1: Seed-Level Run Resumption vs. Step-Level Checkpointing
* **Choice**: Check for completed `seed<seed>_metrics.json` files per seed/scale configuration to decide whether to skip execution.
* **Rationale**: Each seed run is an atomic task in the grid. If a seed completes, its metrics are saved. Skipping completed seeds makes `run_all.py` robust against interruptions without adding complex step-level state serialization.

### Decision 2: Per-Seed Deterministic Dataset Splits & Manifest Persistence
* **Choice**: Update `TinyStoriesDataset(split="train", seed=seed)` to invoke `dataset.shuffle(seed=seed, buffer_size=10000)` on Hugging Face streaming dataset. Save `data/splits/seed_<seed>_split.json` storing seed, buffer size, and dataset name.
* **Rationale**: This guarantees that for seed 42, all 8 models (10M A/B, 30M A/B, 55M A/B, 100M A/B) receive the **identical token sequence**. Saving the manifest ensures that even if training is paused and resumed weeks later, seed 42 will reload the exact same data split configuration.

### Decision 3: Run ID Override in `config.py`
* **Choice**: Introduce `RESUME_RUN_ID` setting in `config.py`. If set (e.g. `RESUME_RUN_ID = "run_dae0c498"`), `run_all.py` targets that directory instead of generating a new random UUID.

### Decision 4: Explicit Git Exclusion of `.pt` Checkpoint Artifacts
* **Choice**: Add `*.pt` and `output/**/*.pt` to `.gitignore`.
* **Rationale**: Model checkpoints for 30M, 55M, and 100M models (110MB, 200MB, 388MB) exceed GitHub's 100MB per-file upload limit and will fail standard `git push`. Lightweight JSON metrics and PNG plots remain tracked in Git.

### Decision 5: Statistical Testing Framework (`scipy.stats`)
* **Choice**: Implement a dedicated `compute_statistical_tests` helper in `eval.py` using `scipy.stats.wilcoxon` (non-parametric) and `scipy.stats.ttest_rel` (paired parametric). Compute Cohen's $d$ and 95% confidence intervals for every primary metric.

### Decision 6: Error-Bar and Loss Convergence Visualizations
* **Choice**: Extend `generate_plots` in `run_all.py` using `matplotlib.pyplot.errorbar` to display $\mu \pm \sigma$ across seeds, annotate statistically significant differences ($p < 0.05$), and render shaded `fill_between` loss convergence curves across training steps.

## Risks / Trade-offs

* **Risk**: Hugging Face streaming dataset shuffling with buffer size 10,000 depends on HF library reproducibility.
  * **Mitigation**: Set explicit PyTorch and Python random seeds before initializing dataset iterations, and record `seed_<seed>_split.json` manifests in `data/splits/`.
