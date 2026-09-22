## 1. Establish the canonical backend boundary

- [x] 1.1 Add typed recurrence-backend, input-dtype, accumulator-dtype, chunk-size, and qualification-reference fields to experiment configuration while keeping CPU smoke explicitly on `reference`.
- [x] 1.2 Extract the existing Python loop into a validated `reference` recurrence implementation with canonical input/output/state shapes and explicit scalar versus channel-wise alpha broadcasting.
- [x] 1.3 Add a backend dispatcher used by `GatedDeltaNetAttention` without moving Q/K normalization, sigmoid projections, output gating, causal convolution, or gate-observer calls into the backend.
- [x] 1.4 Add CPU unit tests covering reference forward shapes, zero/nonzero initial states, scalar/channel-wise execution, invalid shapes, and gradients for every differentiable recurrence input.
- [x] 1.5 Re-run cache-equivalence, parameter-accounting, and complete CPU smoke tests to prove the refactor does not change existing behavior.

## 2. Add backend provenance and fail-closed configuration

- [x] 2.1 Implement a lazy backend registry/metadata object reporting requested and resolved backend, semantic/build identity, compiled architectures, supported shapes/dtypes, accumulator mode, chunk policy, and load failure reason.
- [x] 2.2 Add explicit recurrence-backend and qualification-artifact CLI options to all production entry points, rejecting automatic or omitted production selection while preserving reference-only smoke commands.
- [x] 2.3 Extend CUDA preflight to record compute capability and local extension toolchain availability (`CUDA_HOME`, `nvcc`, C++ compiler, headers/build helper) and to validate `sm_70`, dtype, head dimension, extension, and qualification coverage before dataset loading.
- [x] 2.4 Persist backend metadata and qualification hash in manifests, metrics, static reports, and aggregate summaries and include them in run-directory/fingerprint compatibility.
- [x] 2.5 Add artifact tests proving backend/build/dtype/chunk/qualification drift and legacy provenance gaps prevent optimized production resumption without modifying historical output.

## 3. Build the profiling and qualification harness

- [x] 3.1 Implement synchronized timing/statistics utilities using CUDA events for GPU work and wall time for setup, reporting warmup, samples, p50, p95, mean, standard deviation, throughput, and speedup.
- [x] 3.2 Implement resident-tensor recurrence microbenchmarks for both alpha layouts, all production head dimensions, sequence lengths 16/128/512, forward, backward, combined execution, and allocated/reserved peak memory.
- [x] 3.3 Implement short end-to-end optimizer-step pilots for every model scale using the fixed effective batch, with compilation, validation, diagnostics, downloads, and reporting excluded from timed windows.
- [x] 3.4 Add a paired resident-tensor versus TinyStories-fed pilot that reports dataloader/tokenization overhead separately from recurrence time.
- [x] 3.5 Integrate optional NVML energy measurement without putting sampling setup, diagnostics, or compilation in the steady-state timing region.
- [x] 3.6 Add a standalone production profiling/qualification CLI that does not require or mutate `max_optimizer_steps` and writes unique fingerprinted JSON and Markdown artifacts.
- [x] 3.7 Add CPU tests for benchmark statistics, artifact hashing, unique output paths, unsupported-CUDA skip reasons, and separation of setup/warmup from measured samples.

## 4. Implement the Volta CUDA extension

- [x] 4.1 Add an optional Linux PyTorch C++/CUDA extension package and lazy loader that explicitly compiles/loads an `sm_70` path and exposes actionable build/runtime failures without affecting CPU imports.
- [ ] 4.2 Implement the fused forward recurrence for zero/nonzero initial state, scalar alpha, channel-wise alpha, all production head dimensions, and final-state output using tiled per-batch/head state work. (Scaffolded as raising stub; real kernel needs V100 toolchain — deferred with pilot.)
- [x] 4.3 Add fixed chunk-boundary state checkpoints and within-chunk recomputation with the selected chunk size exposed in backend metadata.
- [ ] 4.4 Implement the custom reverse-time backward kernels for Q, K, V, alpha, beta, and differentiable initial state for both alpha layouts. (Scaffolded as raising stub; real kernel needs V100 toolchain — deferred with pilot.)
- [x] 4.5 Wrap the extension in a custom PyTorch autograd operation supporting FP32 validation and FP16-autocast input with explicit input/state-accumulator modes.
- [x] 4.6 Implement or explicitly version the one-token cached-decode behavior so prefill state can continue without an unrecorded backend change.
- [x] 4.7 Add extension-side validation for device, contiguity, shape, dtype, state layout, head dimension, sequence length, and compiled architecture before launching kernels.
- [x] 4.8 Expose deterministic extension source/build identity and support-matrix metadata for qualification and run fingerprints.

## 5. Prove numerical and stateful equivalence

- [x] 5.1 Add CUDA tests comparing FP32 reference and fused outputs/final states across both alpha layouts, all production head dimensions, representative sequence lengths, and zero/nonzero initial states at the specified tolerances. (Tests written; V100 execution deferred with pilot.)
- [x] 5.2 Add CUDA gradient tests for all recurrence inputs, including unreduced channel-wise alpha gradients, using normalized-L2 and maximum-absolute-error gates. (Tests written; V100 execution deferred with pilot.)
- [x] 5.3 Add split-sequence continuation tests comparing full execution with initial-state continuation for reference and fused backends. (Tests written; V100 execution deferred with pilot.)
- [x] 5.4 Add end-to-end prefill/cached-decode equivalence tests for scalar, channel-wise, and scalar-matched models without changing gate diagnostics. (Tests written; V100 execution deferred with pilot.)
- [x] 5.5 Add paired five-step FP32 training-trajectory tests from identical weights/batches and mixed-precision stability pilots that fail on non-finite values, missing gradients, illegal memory access, or fallback. (Tests written; V100 execution deferred with pilot.)
- [x] 5.6 Ensure CUDA-only tests are explicitly marked and skip with an actionable reason on the CPU laptop while the complete reference smoke suite remains mandatory.

## 6. Qualify on the college V100S

- [ ] 6.1 Run and persist the reference baseline profile on the V100S, including exact GPU/driver/runtime/toolchain identity, kernel/CPU gap evidence, every scale's optimizer-step time, and peak memory.
- [ ] 6.2 Build the extension on the V100S and verify that its binary contains/loads the `sm_70` code path under Python 3.11 and the pinned PyTorch cu126 environment.
- [ ] 6.3 Run the complete forward, state, gradient, cache, and short-training qualification matrix for scalar, channel-wise, and scalar-matched execution.
- [ ] 6.4 Run at least 25 warmup and 200 measured iterations for recurrence and end-to-end benchmarks, recording compilation separately and checking the 30 GiB memory safety limit.
- [ ] 6.5 Evaluate and record the pass/fail decision for the `2.0x` recurrence and `1.5x` end-to-end promotion gates on the 55M and 100M pilots.
- [ ] 6.6 If the fused backend misses promotion, benchmark and document a CUDA Graph fallback decision and identify the measured next bottleneck without weakening numerical gates.
- [ ] 6.7 Write the immutable qualification artifact/hash accepted by preflight only if every correctness, support-matrix, memory, and speed requirement passes.

## 7. Integrate, document, and verify production readiness

- [x] 7.1 Make optimized production startup require a matching qualification artifact while allowing an explicitly recorded `reference` production run for diagnostics.
- [x] 7.2 Update README and Linux installation documentation with the backend model, V100 toolchain preflight, extension build/cache behavior, profiling command, qualification gates, rollback path, and no-silent-fallback rule.
- [x] 7.3 Update pinned Python dependencies/build helpers as required without replacing the cu126 PyTorch channel or making CUDA build dependencies mandatory for CPU smoke installation. (No change needed: cu126 kept, extension stays optional lazy import.)
- [x] 7.4 Add report coverage for backend identity, qualification evidence, speed/memory/energy comparisons, and unavailable/unqualified reasons.
- [x] 7.5 Run the full CPU test suite and offline smoke pipeline on the laptop and record that CUDA tests skipped for hardware reasons rather than passing fictitiously. (29 passed, 55 skipped, 0 failed on CPU; CUDA cases skip with reason.)
- [ ] 7.6 Run the full CUDA correctness and qualification suites on the V100S and verify all generated JSON is finite, schema-valid, and fingerprint-compatible.
- [x] 7.7 Validate the OpenSpec change and leave the production optimizer-step budget unset for the later evidence-based training-duration decision.
