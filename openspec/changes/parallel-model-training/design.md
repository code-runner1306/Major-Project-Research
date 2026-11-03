## Context

The current orchestrator in `run_all.py` executes 8 model architecture configurations across 3 seeds sequentially. With our recent VRAM memory optimizations (micro-batching with `BATCH_SIZE = 8` and `GRAD_ACCUM_STEPS = 4`), single-model peak logits VRAM footprint has dropped to ~0.82 GB, making concurrent model execution feasible on GPUs.

To maximize training throughput without causing OOM errors or output collisions, we need a configurable `PARALLEL_TRAININGS` knob, dynamic VRAM pre-checking, worker-tagged logging, synchronization locks for shared output trackers, and explicit GPU memory unloading after a model's seeds finish.

## Goals / Non-Goals

**Goals:**
* Add `PARALLEL_TRAININGS` knob in `config.py`.
* Pre-check free VRAM via `torch.cuda.mem_get_info()` using an 85% safety threshold, falling back to `PARALLEL_TRAININGS = 1` if capacity is tight.
* Execute parallel model worker tasks using `concurrent.futures.ThreadPoolExecutor` (or `ProcessPoolExecutor`).
* Format worker logs with tags (e.g. `[Worker-1 | 10M-scalar] Step X/Y - Loss: Z`).
* Protect `global_tracker.json` writes with `threading.Lock()`.
* Explicitly unload model instances (`del model`, `gc.collect()`, `torch.cuda.empty_cache()`) upon finishing all seeds for a given model.
* Maintain 100% compatibility with `RESUME_RUN_ID` and per-seed split manifests.

**Non-Goals:**
* Multi-node distributed cluster training (single GPU setup is targeted).

## Decisions

### Decision 1: Isolated process workers
* **Choice**: The Bash launcher starts one `run_all.py` process for each family/scale/variant/seed training run and bounds active processes with `PARALLEL_WORKERS`.
* **Rationale**: Separate processes isolate Python, NumPy, and PyTorch random-number generators, model/optimizer state, and CUDA contexts. Each process writes to its unique run directory; worker training processes skip aggregate report writes, and the launcher evaluates runs sequentially afterward.

### Decision 2: Dynamic VRAM Capacity Pre-Check (85% Safety Margin)
* **Choice**: Calculate `Required_VRAM = Max_Scale_VRAM * N`. Compare against `0.85 * Free_VRAM`.
* **Rationale**: If $N$ models require more VRAM than available, falling back to `PARALLEL_TRAININGS = 1` prevents OOM crashes before any model training starts.

### Decision 3: Output Isolation & Synchronization Lock
* **Choice**: Keep each model's output strictly inside `output/run_<uuid>/run_<variant>_<scale>/`. Synchronize global tracker writes using `tracker_lock = threading.Lock()`.
* **Rationale**: Prevents file collisions across concurrent workers while maintaining progress traces in `global_tracker.json`.

## Risks / Trade-offs

* **Risk**: High GPU compute or CPU input contention if $N > 1$ on the target host.
* **Mitigation**: Do not assume a universal parallel default. Use the bounded pilot's measured speedup and capacity report before selecting a production worker count.

## Pilot Qualification

Before production concurrency is selected, the V100 pilot measures resident and data-fed optimizer steps for TinyStories, MQAR, and overwrite. For configured low-tier scales, it may run bounded concurrent optimizer work at worker counts 1–3 only when measured single-worker reserved memory predicts that the requested workers fit below the 85% free-VRAM threshold. One shared resource sampler records process/system CPU and whole-device GPU utilization during each trial. The report compares aggregate completed optimizer steps per second and speedup against the single-worker baseline; it labels CPU saturation as a contention indicator, not proof of thermal throttling. These measurements inform the operator and do not automatically change production worker count.

### Decision 4: Pilot capacity gate and measurement policy
* **Choice**: For counts above 1, qualify the requested count across every selected scale and dataset before production training. Evaluation remains sequential.
* **Rationale**: A scale falls back to one worker if any dataset trial fails the VRAM capacity gate; other trial failures stop the launch. Sequential evaluation avoids collecting inference latency and throughput under concurrent GPU load.
* **Measurement caveat**: Per-run wall time may reflect concurrent GPU load, while GPU utilization and NVML energy are device-wide and overlap sibling processes. The requested concurrency limit is stored in each manifest and metrics artifact, with aggregate telemetry identified explicitly.
