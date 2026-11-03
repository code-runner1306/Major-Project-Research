## ADDED Requirements

### Requirement: Configurable concurrent model worker pool
The system SHALL support configuring `PARALLEL_TRAININGS` in `config.py` to specify the maximum number of distinct model architecture configurations to train concurrently in parallel.

#### Scenario: Running multiple models in parallel
- **WHEN** `config.PARALLEL_TRAININGS` is set to $N > 1$ and sufficient GPU memory is available
- **THEN** the orchestrator launches $N$ worker threads/processes concurrently, each processing its assigned model configuration across seeds.

### Requirement: Tagged worker console output
Console log outputs from concurrent worker tasks SHALL be prefixed with a clear worker and model identifier tag.

#### Scenario: Printing progress from parallel workers
- **WHEN** Worker 1 outputs training loss for 10M Scalar and Worker 2 outputs loss for 10M Channel-wise
- **THEN** console lines are printed as `[Worker-1 | 10M-scalar] Step X/Y - Loss: Z` and `[Worker-2 | 10M-channel-wise] Step X/Y - Loss: Z`.

### Requirement: Model memory unloading upon seed completion
Once a model configuration finishes all its seed runs, all associated model weights, optimizers, and gradient buffers SHALL be explicitly dereferenced and deleted, followed by `gc.collect()` and `torch.cuda.empty_cache()`.

#### Scenario: Completing all seed runs for a model architecture
- **WHEN** all seeds for a specific (scale, variant) model finish training
- **THEN** the worker invokes memory cleanup to free GPU VRAM before picking up the next model task.

### Requirement: Thread-safe progressive tracking
Updates to `global_tracker.json` across concurrent workers SHALL be protected by a synchronization lock (`threading.Lock`).

#### Scenario: Concurrent workers updating global tracker
- **WHEN** two parallel worker tasks finish a seed run simultaneously
- **THEN** global tracker file writes are serialized safely without data corruption.

### Requirement: Pilot-based concurrency qualification
The V100 pilot SHALL measure bounded concurrent optimizer workloads for configured low-tier model scales and report aggregate throughput and per-worker speedup against a single-worker baseline before recommending production concurrency.

#### Scenario: Comparing low-tier worker counts
- **WHEN** a pilot requests worker counts 1, 2, or 3 and the VRAM pre-check passes
- **THEN** the pilot records completed optimizer steps per second, wall time, and sampled CPU/GPU utilization for each trial, and reports a recommendation without silently changing production execution settings.

### Requirement: Pilot progress reporting
The V100 pilot SHALL print timestamped, flushed progress updates while running resident model trials, data-fed dataset trials, and parallel worker trials. Updates SHALL identify the active scale, variant, dataset, worker count, and phase, and include completed/total steps during measured loops.

#### Scenario: Monitoring the pilot launched by the research-grid script
- **WHEN** the operator runs `run_research_grid.sh --pilot`
- **THEN** the terminal and run log show progress through model setup, resident warmup and measurement, each data-fed dataset warmup and measurement, and each parallel trial, including explicit VRAM skips or failures.

#### Scenario: Long measured pilot loop
- **WHEN** a pilot trial has enough measured steps to run for an extended period
- **THEN** it emits periodic step-count updates rather than waiting until the trial or full pilot completes.

### Requirement: Configurable production training workers
The research-grid launcher SHALL accept `PARALLEL_WORKERS` as the maximum number of independent training-run processes active at once, defaulting to 1. Each process SHALL own one family/scale/variant/seed run, including its model, optimizer, dataloader iterator, and run directory. The launcher SHALL wait for all training runs for a family to finish before starting that family's evaluation, and SHALL keep evaluation sequential to preserve inference timing measurements.

#### Scenario: Running with the default worker count
- **WHEN** the operator runs the grid without setting `PARALLEL_WORKERS`
- **THEN** training runs execute sequentially and retain the existing train-then-evaluate family ordering.

#### Scenario: Running with multiple workers
- **WHEN** the operator sets `PARALLEL_WORKERS` to an integer greater than 1
- **THEN** the launcher runs up to that many isolated training processes concurrently, labels progress by dataset/model/variant/seed, and waits for the stage before evaluation.

#### Scenario: Matched data across model variants
- **WHEN** scalar and channel-wise variants share the same research seed
- **THEN** each process independently loads the same deterministic data ordering for the paired comparison; model weights, optimizers, iterators, and output artifacts remain separate.

#### Scenario: Recording concurrent measurement context
- **WHEN** multiple training processes overlap on the GPU
- **THEN** each run's manifest and metrics record the configured concurrency limit and identify device-level resource telemetry as aggregate across concurrent work.
