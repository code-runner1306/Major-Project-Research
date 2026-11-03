## ADDED Requirements

### Requirement: GPU memory capacity inspection
Before launching worker tasks, the orchestrator SHALL inspect free GPU memory via `torch.cuda.mem_get_info()` and calculate whether the GPU has enough usable memory to safely support `PARALLEL_TRAININGS` concurrent models.

#### Scenario: Pre-checking VRAM capacity before execution
- **WHEN** `run_all.py` starts execution with `PARALLEL_TRAININGS = N` ($N > 1$)
- **THEN** the system estimates total VRAM required for $N$ models and compares it against 85% of usable free VRAM.

### Requirement: Automatic fallback to sequential execution
If estimated VRAM required for $N$ parallel models exceeds 85% of usable free VRAM, the orchestrator SHALL log a clear warning and automatically default to sequential execution (`PARALLEL_TRAININGS = 1`).

#### Scenario: Running on GPU with tight VRAM capacity
- **WHEN** free VRAM is insufficient to safely run $N$ models in parallel
- **THEN** the system logs `[Parallel Check] Insufficient VRAM! Needed ~X GB, but ~Y GB free. Defaulting to PARALLEL_TRAININGS = 1.` and executes sequentially without crashing.

### Requirement: Pilot concurrency capacity gate
The pilot SHALL skip a requested concurrent trial when the measured single-worker peak reserved memory multiplied by the requested worker count exceeds 85% of currently free VRAM.

#### Scenario: Pilot declines an unsafe worker count
- **WHEN** the selected GPU does not have sufficient free VRAM for a pilot worker count
- **THEN** the pilot records the capacity estimate and skip reason in its report without attempting the concurrent allocation.
