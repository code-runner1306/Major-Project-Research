## 1. Configuration Settings

- [x] 1.1 Add `PARALLEL_TRAININGS = 2` setting to `config.py`.

## 2. VRAM Capacity & Parallel Worker Infrastructure

- [x] 2.1 Implement `check_parallel_capacity(num_requested, scales_to_run)` in `run_all.py` to inspect VRAM via `torch.cuda.mem_get_info()` and apply 85% safety threshold fallback.
- [x] 2.2 Add thread synchronization lock (`tracker_lock = threading.Lock()`) in `run_all.py` for safe `global_tracker.json` updates.
- [x] 2.3 Implement worker-tagged logger helper `log_worker(worker_id, model_tag, message)` for formatted output.

## 3. Parallel Worker Execution & Model Memory Unloading

- [x] 3.1 Refactor model processing in `run_all.py` into worker function `process_model_config(scale, variant, worker_id, run_base_dir)`.
- [x] 3.2 Add explicit memory unloading (`del model`, `gc.collect()`, `torch.cuda.empty_cache()`) after a model configuration completes all its seeds.
- [x] 3.3 Integrate `ThreadPoolExecutor(max_workers=parallel_jobs)` in `main()` of `run_all.py` to manage parallel worker jobs.

## 4. Verification

- [x] 4.1 Run smoke test with `PARALLEL_TRAININGS = 2`, verify worker log prefixes, confirm VRAM pre-check succeeds, test fallback behavior, and verify `statistical_analysis.json` and plots are generated properly.

## 5. Pilot Capacity Qualification

- [x] 5.1 Add data-fed optimizer timing for TinyStories, MQAR, and overwrite on each selected pilot scale.
- [x] 5.2 Add a bounded 1/2/3-worker low-tier pilot sweep guarded by measured per-worker peak reserved VRAM and the 85% free-memory safety threshold.
- [x] 5.3 Capture one shared CPU/GPU resource timeline per concurrency trial and report aggregate throughput, speedup, CPU saturation indicators, GPU utilization, and a recommended worker count without changing production settings.
- [x] 5.4 Write a pilot Markdown/JSON report with per-family per-scale estimates, 1/3/5-seed totals, and concurrency findings/caveats.
- [x] 5.5 Print timestamped progress through resident model, data-fed dataset, and parallel worker trials; persist launcher output to a timestamped run log.

## 6. Production Training Concurrency

- [x] 6.1 Add `PARALLEL_WORKERS` to the Bash launcher with sequential default and per-scale worker limits.
- [x] 6.2 Qualify requested worker counts across selected scales and all three datasets; fall back to one worker on VRAM capacity skips and fail before training on other trial failures.
- [x] 6.3 Launch isolated per-seed/variant training processes, keep evaluation sequential, and prevent concurrent aggregate report writes.
- [x] 6.4 Record concurrency and telemetry scope in run manifests/metrics and label live training progress by run identity.
