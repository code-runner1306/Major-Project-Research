## Why

Currently, `run_all.py` executes the 8 model architectures across 3 seeds (24 total training runs) in a single-threaded sequential loop. On GPUs with sufficient memory capacity, training multiple model configurations concurrently can significantly reduce total experiment runtime.

However, executing multiple models in parallel requires careful GPU memory management, output directory isolation, pre-execution VRAM capacity checks, thread-safe global tracking, and graceful memory cleanup upon completing each model's seed runs.

Adding a `PARALLEL_TRAININGS` configuration knob with automatic VRAM pre-check fallbacks, isolated process worker pools, worker-prefixed console logging, and model memory unloading will accelerate the research workflow while maintaining full compatibility with run resumption and single-GPU setups.

## What Changes

* **Parallel Trainings Knob**: Introduce `PARALLEL_TRAININGS` setting in `config.py` (e.g. `PARALLEL_TRAININGS = 2`).
* **VRAM Safety Pre-Check**: Add automatic GPU memory capacity inspection in `run_all.py`. If usable free VRAM is insufficient to safely run $N$ models concurrently without OOM risks, log a warning and fall back to sequential execution (`PARALLEL_TRAININGS = 1`).
* **Concurrent Model Worker Pool**: Re-architect `run_all.py` to launch up to `PARALLEL_TRAININGS` concurrent model execution tasks using a thread/process pool.
* **Worker Console Log Prefixes**: Format console logs with model/worker tags (e.g. `[Worker-1 | 10M-scalar] Step 1000/15000 - Loss: 3.21`) for clear real-time monitoring.
* **Graceful Model Unloading**: Ensure that once a model configuration finishes all its seed runs, its model instance, optimizer, and memory buffers are explicitly deleted, followed by `gc.collect()` and `torch.cuda.empty_cache()`.
* **Thread-Safe Global Tracking**: Use thread/process locks when updating `global_tracker.json` across concurrent workers.
* **Resumption & Split Isolation**: Maintain full compatibility with `RESUME_RUN_ID` and per-seed dataset split manifests.

## Capabilities

### New Capabilities
- `parallel-execution`: Support concurrent training of multiple model configurations using worker pools while preserving thread-safe file isolation and clear tagged console logging.
- `vram-capacity-check`: Dynamically inspect available GPU memory and automatically fall back to sequential execution if VRAM capacity is tight.

### Modified Capabilities
*None*

## Impact

* **Files Affected**:
  * `config.py`: Add `PARALLEL_TRAININGS` setting.
  * `run_all.py`: Add VRAM capacity check, worker pool execution, thread locking for global tracker, tagged logging, and model memory unloading.
  * `train.py`: Add explicit memory cleanup helper (`cleanup_model_memory`).
* **APIs / Dependencies**: Uses Python standard `concurrent.futures`, `threading`, and PyTorch `torch.cuda.mem_get_info()`.
