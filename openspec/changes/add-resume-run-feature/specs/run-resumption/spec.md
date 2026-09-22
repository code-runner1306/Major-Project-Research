## ADDED Requirements

### Requirement: Skip completed configurations on resume
The orchestrator `run_all.py` SHALL inspect the target run directory for existing `seed<seed>_metrics.json` artifacts before launching a model training run. If the metrics file already exists and is valid, the orchestrator SHALL skip training for that specific configuration and log that the run was restored.

#### Scenario: Orchestrator encounters completed run artifact
- **WHEN** `run_all.py` processes a (scale, variant, seed) tuple whose `seed<seed>_metrics.json` file already exists in the target run folder
- **THEN** `run_all.py` skips execution of `train_model`, `benchmark_inference`, and retrieval probes for that seed, loading the existing metrics directly into the summary aggregation pipeline.

### Requirement: Deterministic run output directory reuse
The system SHALL support configuring a fixed run identifier or directory path in `config.py` (or via environment variables) instead of generating a random UUID on every execution.

#### Scenario: User specifies a target run folder
- **WHEN** `config.RESUME_RUN_ID` or `config.TARGET_RUN_DIR` is set to an existing run folder name
- **THEN** `run_all.py` reuses the specified run directory for output generation rather than creating a new UUID directory.

### Requirement: Exclude model checkpoints from Git tracking
The project `.gitignore` file SHALL explicitly exclude PyTorch model checkpoint files (`*.pt`, `*.ckpt`, `output/**/*.pt`) to prevent tracking large binary files that exceed GitHub's 100 MB per-file limit.

#### Scenario: Model checkpoint files created during training
- **WHEN** `train.py` saves `model_checkpoint_seed<seed>.pt` to disk
- **THEN** git status ignores the `.pt` files, preventing them from being staged or committed to the repository.
