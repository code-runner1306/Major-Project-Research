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
