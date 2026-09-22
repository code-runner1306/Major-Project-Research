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

### Decision 1: ThreadPoolExecutor vs Multiprocessing Pool
* **Choice**: Use `concurrent.futures.ThreadPoolExecutor(max_workers=parallel_jobs)` combined with PyTorch's native CUDA stream management and explicit memory cleanup.
* **Rationale**: Avoids multi-process serialization overhead for model weights and dataset streaming objects while permitting concurrent PyTorch kernel launches on CUDA device 0.

### Decision 2: Dynamic VRAM Capacity Pre-Check (85% Safety Margin)
* **Choice**: Calculate `Required_VRAM = Max_Scale_VRAM * N`. Compare against `0.85 * Free_VRAM`.
* **Rationale**: If $N$ models require more VRAM than available, falling back to `PARALLEL_TRAININGS = 1` prevents OOM crashes before any model training starts.

### Decision 3: Output Isolation & Synchronization Lock
* **Choice**: Keep each model's output strictly inside `output/run_<uuid>/run_<variant>_<scale>/`. Synchronize global tracker writes using `tracker_lock = threading.Lock()`.
* **Rationale**: Prevents file collisions across concurrent workers while maintaining progress traces in `global_tracker.json`.

## Risks / Trade-offs

* **Risk**: High GPU compute contention if $N > 4$ on low-end GPUs.
  * **Mitigation**: Setting `PARALLEL_TRAININGS = 2` by default provides optimal speedup without CPU/GPU bottlenecking.
