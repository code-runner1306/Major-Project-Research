## ADDED Requirements

### Requirement: Pre-implementation V100 baseline profile
The system SHALL provide a standalone baseline profiler that runs without setting the full production training budget and records the active device/toolchain, recurrence kernel activity, synchronized forward/backward timing, optimizer-step timing, peak memory, GPU utilization evidence, and data-fed versus resident-tensor timing.

#### Scenario: V100 baseline captured
- **WHEN** the profiler runs on the college V100S
- **THEN** it writes a versioned machine-readable artifact containing GPU identity, compute capability, driver/runtime, PyTorch version, local toolkit/compiler availability, warmup count, measured iteration count, shape, dtype, timing summaries, and peak allocated/reserved memory

#### Scenario: Compilation and warmup accounting
- **WHEN** a candidate backend performs build, JIT, or autotuning work
- **THEN** setup time is recorded separately and excluded from steady-state p50 and p95 timings

### Requirement: Forward and final-state equivalence
An optimized backend MUST pass deterministic comparisons against the reference backend for outputs and final states across scalar and channel-wise decay, zero and nonzero initial states, all production head dimensions, and representative sequence lengths before qualification succeeds.

#### Scenario: FP32 equivalence
- **WHEN** both backends receive identical deterministic FP32 inputs on a supported short or medium sequence
- **THEN** output and final-state maximum absolute error is at most `1e-4` and all values are finite

#### Scenario: Production mixed-precision equivalence
- **WHEN** both backends receive identical inputs under the candidate production autocast and accumulator configuration at sequence length 512
- **THEN** output and final-state errors remain within the versioned mixed-precision tolerance of at most `2e-2` absolute and relative error and all values are finite

### Requirement: Backward-gradient equivalence
An optimized backend MUST pass gradient comparisons against the reference backend for Q, K, V, alpha, beta, initial state when differentiable, and representative upstream model parameters in both gating variants.

#### Scenario: FP32 recurrence gradients
- **WHEN** identical deterministic FP32 inputs and upstream gradients are backpropagated through both backends
- **THEN** each required gradient is present and finite, has normalized L2 relative error at most `1e-3`, and has maximum absolute error at most `1e-3`

#### Scenario: Channel-wise alpha gradients
- **WHEN** channel-wise decay is differentiated
- **THEN** the optimized backend returns a distinct gradient for every `[B,T,H,D]` alpha element and agrees with the reference tolerance without head-wise reduction

### Requirement: Stateful inference equivalence
Qualification SHALL verify recurrent initial/final state continuity and end-to-end full-prefix versus cached decoding for scalar, channel-wise, and scalar-matched models.

#### Scenario: Split-sequence recurrence
- **WHEN** a sequence is processed once in full and once as two segments using the first segment's final state as the second segment's initial state
- **THEN** concatenated outputs and final states agree within the dtype-specific qualification tolerance

#### Scenario: End-to-end cache test
- **WHEN** each model variant performs prefill followed by token-by-token cached decode
- **THEN** its logits agree with full-prefix logits within the existing cache-equivalence tolerance or a stricter documented dtype-specific tolerance

### Requirement: Short-training equivalence and stability
Qualification SHALL run paired short training trajectories from identical initial weights and batches and SHALL reject an optimized backend that introduces non-finite values, missing gradients, systematic loss divergence, or variant-specific execution differences.

#### Scenario: Paired FP32 trajectory
- **WHEN** reference and optimized tiny models train for at least five optimizer steps from identical weights on identical batches
- **THEN** all losses and gradients remain finite and the per-step loss difference remains at most `1e-3`

#### Scenario: Mixed-precision stability trajectory
- **WHEN** each production variant completes the versioned mixed-precision pilot
- **THEN** no backend-related NaN, Inf, illegal memory access, or unsupported fallback occurs and the stability result is persisted

### Requirement: Reproducible performance qualification
Performance qualification SHALL compare reference and optimized execution on the same V100S using synchronized, paired, fixed-shape trials with at least 25 warmup iterations and 200 measured iterations and SHALL report p50, p95, mean, standard deviation, throughput, speedup, peak memory, and energy when NVML supports it.

#### Scenario: Recurrence microbenchmark
- **WHEN** the qualification suite benchmarks each production head dimension at sequence lengths 16, 128, and 512 for both alpha layouts
- **THEN** it reports separate forward, backward, and combined timings plus allocated/reserved memory for both backends

#### Scenario: End-to-end optimizer benchmark
- **WHEN** the suite benchmarks the fixed effective batch and gradient-accumulation policy for each model scale
- **THEN** it records optimizer-step seconds and training tokens per second without validation, diagnostics, downloads, or extension compilation inside the timed region

#### Scenario: Data-fed bottleneck check
- **WHEN** the same short model pilot is run once with resident tensors and once through the TinyStories dataloader
- **THEN** the report identifies the observed data-pipeline overhead separately from recurrence/backend performance

### Requirement: Promotion gates
The system MUST NOT mark `cuda-fused` qualified for production unless all correctness scenarios pass, all production shapes fit within the V100S memory safety limit, recurrence forward-plus-backward p50 is at least `2.0x` faster than reference, and end-to-end optimizer-step p50 is at least `1.5x` faster than reference for the representative 55M and 100M pilots.

#### Scenario: Candidate passes all gates
- **WHEN** all numerical, stateful, stability, shape, memory, and speed thresholds pass on the target V100S
- **THEN** the suite writes a signed-by-hash qualification artifact that production preflight can accept

#### Scenario: Correct but insufficiently faster
- **WHEN** numerical checks pass but either speed threshold fails
- **THEN** the candidate remains available for development diagnostics but is not promoted as the production optimized backend

#### Scenario: Memory limit exceeded
- **WHEN** a production shape exceeds `30 GiB` peak allocated memory or triggers an out-of-memory error on the 32 GB V100S
- **THEN** qualification fails that shape and production preflight rejects the backend/configuration combination

#### Scenario: Qualification mismatch
- **WHEN** the active GPU, backend build, dtype, chunk policy, or production shape is not covered by the qualification artifact
- **THEN** production preflight refuses to treat the backend as qualified

### Requirement: Qualification reporting and auditability
The qualification command SHALL write immutable JSON and concise Markdown summaries that include every test case, tolerance, raw timing sample summary, excluded setup time, failure reason, and artifact hash without modifying or deleting prior qualification results.

#### Scenario: Repeated qualification run
- **WHEN** qualification is executed more than once
- **THEN** each execution writes to a unique fingerprinted directory and prior evidence remains intact

#### Scenario: CPU development environment
- **WHEN** the qualification tests are collected on the CPU-only laptop
- **THEN** CUDA-specific cases skip with explicit reasons while reference correctness and the ordinary CPU smoke suite remain runnable

