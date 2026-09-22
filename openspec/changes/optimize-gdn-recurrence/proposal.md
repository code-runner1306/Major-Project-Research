## Why

The current Gated DeltaNet recurrence is an autograd-tracked Python loop over every token in every GDN layer, causing thousands of small CUDA operations per optimizer step and making the planned 24-run TinyStories grid unnecessarily expensive. Before reducing training tokens, scales, or seeds, the project needs a production recurrence path that preserves the scalar-versus-channel-wise experiment exactly while using the Volta V100S efficiently.

## What Changes

- Retain the current eager recurrence as the portable correctness reference for CPU smoke tests, debugging, and numerical comparison.
- Introduce an explicit recurrence-backend interface supporting the reference implementation and a Volta-compatible optimized CUDA implementation for training and prefill, while preserving cached one-token decoding.
- Implement the optimized CUDA recurrence for both scalar decay (`[B,T,H,1]`) and channel-wise decay (`[B,T,H,D]`), including forward state evolution and backward gradients without changing model parameters or equations.
- Add fail-closed production capability checks for GPU architecture, FP16 support, CUDA-extension availability, supported head dimensions, and backend version. Production runs must never silently fall back to a different backend.
- Add numerical-equivalence, gradient, state/cache, mixed-precision, short-training, memory, and performance qualification suites against the reference recurrence.
- Add a synchronized V100 pilot benchmark that separates compilation/setup time from steady-state recurrence and end-to-end optimizer-step timing, and records speedup, memory, throughput, and energy evidence before the fixed production budget is selected.
- Persist the requested and resolved recurrence backend, implementation version, accumulator dtype, chunk/checkpoint policy, and qualification result in run manifests and reports.
- **BREAKING**: production artifacts created without matching recurrence-backend provenance are not resumable as optimized production runs.

## Capabilities

### New Capabilities

- `optimized-gdn-recurrence`: Backend selection, exact recurrence semantics, Volta-compatible optimized execution, explicit fallback behavior, and persisted backend provenance.
- `recurrence-optimization-qualification`: Numerical-correctness gates and reproducible V100 performance/memory/energy benchmarks required before the optimized backend can be used for the research grid.

### Modified Capabilities

None. The repository has no archived baseline specs; this change adds new contracts around the already implemented experiment pipeline.

## Impact

- Primary code paths: `model.py`, `config.py`, `preflight.py`, `train.py`, `orchestration.py`, `artifacts.py`, CLI entry points, diagnostics, and reporting.
- New code: recurrence backend abstraction, optional Linux CUDA-extension sources/build loader, backend qualification utility, and V100 pilot benchmark command.
- Tests: CPU reference tests remain mandatory; CUDA-only equivalence and performance tests are explicitly marked and skipped when compatible hardware is unavailable.
- Runtime: Linux x86-64, Python 3.11, PyTorch cu126, CUDA 12.6, and V100S (`sm_70`) remain the production target. Building a repository-owned CUDA extension may additionally require a compatible local CUDA toolkit, C++ compiler, and build helper; absence is reported before production rather than discovered during a long run.
- External optimized GDN packages are not assumed to be compatible: the project requires channel-wise decay semantics and Volta support, both of which must be demonstrated before any external backend can be admitted.
