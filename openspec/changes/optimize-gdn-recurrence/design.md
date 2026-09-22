## Context

`GatedDeltaNetAttention.forward` currently projects and convolves Q/K/V, normalizes Q/K, constructs sigmoid `alpha` and `beta`, then calls `recurrent_forward`. That function allocates a `[B,H,D,D]` state and executes an ordinary Python loop over `T`; each iteration performs two state-vector products, a rank-one write, decay, and output read under PyTorch autograd. At the production settings (`T=512`, four accumulated microbatches), the 100M hybrid model invokes the loop in twelve GDN layers, so launch overhead and saved per-token autograd state can dominate otherwise modest models.

The same function is also the semantic center of several established contracts:

- scalar `alpha` has shape `[B,T,H,1]` and broadcasts over state rows;
- channel-wise `alpha` has shape `[B,T,H,D]` and independently decays state rows;
- `beta` is head-wise, Q/K are L2-normalized before the recurrence, and the SiLU output gate is applied after it;
- prefill returns the final recurrent state, while one-token decode accepts that state and updates the convolution caches;
- gate observers inspect projected `alpha` outside timed regions;
- CPU smoke tests must remain offline and CUDA-independent;
- run fingerprints currently cover the complete profile, environment, and source state.

The production target is one Linux V100S 32 GB (`sm_70`) using Python 3.11 and the PyTorch CUDA 12.6 wheel. PyTorch cu126 supports Volta, but current upstream Triton documents compute capability 8.0+ as its supported NVIDIA floor. The available Flash Linear Attention GDN operation is also not a drop-in for this study: its documented forget gate is head/value-head-wise, whereas this repository must support a row/channel-wise state decay. Therefore the design cannot assume that installing a contemporary Triton package or replacing the layer with a third-party GDN layer preserves either hardware compatibility or the manipulated variable.

## Goals / Non-Goals

**Goals:**

- Preserve one canonical recurrence equation and parameterization across reference and optimized execution.
- Provide a Volta-compatible optimized training/prefill path for scalar, channel-wise, and scalar-matched models.
- Keep the eager PyTorch implementation as the CPU-safe correctness oracle and rollback path.
- Prove forward, final-state, gradient, cached-decode, mixed-precision, and short-training agreement before promotion.
- Measure actual V100 speed, memory, energy, compilation overhead, and end-to-end impact with reproducible artifacts.
- Make backend identity and qualification status part of production provenance and resumption compatibility.
- Fail before training if an explicitly requested optimized backend cannot run.

**Non-Goals:**

- Changing the Gated DeltaNet equations, projection shapes, hybrid layer schedule, model sizes, optimizer, fixed-token comparison, or seed policy.
- Replacing the model with a third-party architecture whose normalization, convolution, gate activation, state layout, or decay semantics differ.
- Optimizing full attention, SwiGLU, tied-output cross entropy, tokenization, or dataloader workers in this change. The profiler may identify these as subsequent bottlenecks.
- Making CPU smoke tests compile or execute CUDA code.
- Claiming a speedup before it is measured on the college V100S.

## Decisions

### 1. Define one narrow recurrence contract

The existing eager loop will be renamed and isolated as the reference implementation behind a function accepting `q`, `k`, `v`, `alpha`, `beta`, and an optional initial state, and returning all token outputs plus the final state. Q/K normalization, sigmoid gate projection, output gating, and the diagnostic observer remain outside this boundary.

For each time step, the contract remains:

```text
error_t = v_t - S_(t-1) k_t
S_t     = alpha_t * S_(t-1) + beta_t * error_t k_t^T
o_t     = S_t q_t
```

Here `S` is `[B,H,D,D]`; scalar `alpha_t` broadcasts as `[B,H,1,1]`, and channel-wise `alpha_t` broadcasts as `[B,H,D,1]`. Keeping this boundary below projections prevents a backend from silently changing the architecture or parameter count.

Alternative considered: replace `GatedDeltaNetAttention` with an external layer. Rejected because it would entangle recurrence optimization with projections, convolution, normalization, and gate semantics.

### 2. Use explicit backends with no production fallback

The model/profile will expose `reference` and `cuda-fused` recurrence backends. `reference` works on CPU and CUDA and remains the smoke/debug default. `cuda-fused` is valid only after the production preflight loads the extension, confirms `sm_70` support and the requested dtype/head dimensions, and locates a matching qualification record.

`auto` is not permitted for production research runs. A command may use backend probing during qualification, but a real run must name the resolved backend so artifacts cannot vary with machine-local availability. Cached one-token decode may use a dedicated recurrent kernel or the reference primitive only when that decode choice is itself recorded as part of the backend implementation version; it may not silently vary during a run.

Alternative considered: silently use eager when the extension import fails. Rejected because paired jobs could then use different implementations and invalidate latency, energy, and resumption comparisons.

### 3. Build a repository-owned CUDA extension for Volta

The optimized implementation will use PyTorch's C++/CUDA extension interface and compile an `sm_70` code path, rather than depend on a Triton kernel whose current supported hardware floor excludes Volta. The extension will expose a custom autograd operation with forward and backward implementations for both alpha layouts.

The kernel will process independent batch/head states in parallel and tile the `D x D` state operations. It will checkpoint state at fixed chunk boundaries (initial target: 64 tokens) and recompute within a chunk during backward, avoiding storage of every complete state while retaining exact recurrence order inside each chunk. Chunk size is configuration/provenance, not an invisible tuning knob. The implementation will initially leave Q/K normalization and the output SiLU gate in PyTorch so the optimized surface is as small and auditable as possible.

The extension will support FP32 validation and the production FP16-autocast input path. State-accumulator dtype will be explicit. Qualification will compare each supported accumulator mode against a reference run using the same mode, and only a qualified mode may be selected for production.

Alternative considered: implement the entire attention block as one fused kernel. Deferred because it greatly increases correctness risk and makes it harder to attribute numerical differences.

### 4. Measure before and after the kernel, not only whole jobs

A standalone qualification command will operate without unlocking the full production budget. It will run two tiers:

1. Resident-tensor recurrence microbenchmarks isolate forward, backward, and forward-plus-backward at the exact head dimensions, batch sizes, alpha layouts, and sequence lengths used by all four scales.
2. Short end-to-end model pilots include projections, convolution, hybrid attention, optimizer, gradient accumulation, validation-disabled timed windows, peak memory, and optional NVML energy. A separate TinyStories-fed window reveals dataloader/tokenization stalls without contaminating the kernel result.

CUDA events and device synchronization will delimit steady-state measurements. Extension build/JIT time and warmup are recorded separately and excluded from p50/p95 throughput. The same initialized tensors, model weights, and batch order are used for paired reference/optimized comparisons.

The optimized backend is promoted only when all correctness gates pass and the V100 evidence reaches both a recurrence-level speedup target and an end-to-end speedup target. Failure produces a report, leaves production blocked, and does not weaken correctness tolerances.

### 5. Treat backend provenance as scientific configuration

`ExperimentConfig`, run manifests, preflight output, metrics, and aggregate reports will record:

- requested and resolved recurrence backend;
- extension/backend semantic version and source/build identity;
- GPU compute capability and compiled CUDA architectures;
- input and state-accumulator dtypes;
- chunk size and state-checkpoint/recomputation policy;
- qualification artifact hash and pass/fail result;
- reference and optimized timing/memory results when available.

These fields participate in the canonical run fingerprint. Existing smoke outputs remain readable as historical artifacts, but production resumption across different recurrence backends or unqualified backend builds is rejected.

### 6. Preserve portable development and provide a staged rollout

Extension loading is lazy and isolated from module import. On this CPU-only laptop, unit tests and the complete smoke pipeline always select `reference`; CUDA tests use explicit markers and skip with a reason when the extension or compatible GPU is absent.

Implementation proceeds through measurement and decision gates:

1. Add the backend boundary, provenance, reference tests, and benchmark harness.
2. Run the unoptimized V100 profiler to confirm recurrence share, actual memory, and toolchain availability (`nvcc`, compiler, CUDA headers).
3. Implement and qualify the fused forward path, then custom backward with chunk checkpoints.
4. Run full correctness and V100 performance qualification.
5. Promote `cuda-fused` only after thresholds pass; otherwise retain `reference` and use the evidence to choose a follow-up (CUDA Graph capture, a compatible external CUDA kernel, or reduced experimental scope).

CUDA Graph capture is retained as a measured fallback investigation because it can reduce launch overhead without changing recurrence math, but it is not the primary backend: it preserves the many small kernels, has static-buffer and memory constraints, and does not solve saved-state overhead. `torch.compile`/Inductor is also a probe rather than a production assumption on `sm_70`.

## Risks / Trade-offs

- **[The V100 host has a CUDA-capable driver but no local toolkit/compiler]** → Run toolchain qualification before implementation deployment; keep the reference path usable and document exact required system packages. Do not attempt a long production job.
- **[Custom backward silently computes incorrect alpha or recurrent-state gradients]** → Compare every input gradient and representative model-parameter gradients against the eager oracle in FP32 before mixed-precision testing; require both scalar and channel-wise cases.
- **[FP16 recurrence accumulates differently enough to change training]** → Make accumulator dtype explicit, compare against a dtype-matched reference, run a paired short trajectory, and promote only a qualified mode shared by all variants.
- **[A fused kernel speeds the recurrence but not the model]** → Require end-to-end optimizer-step improvement in addition to kernel microbenchmark speedup; profile tied-output softmax and data loading separately for follow-up work.
- **[Kernel resource use or shape specialization fails at one model scale]** → Qualify every production head dimension and alpha layout, enforce a memory safety margin below 32 GB, and fail preflight for unsupported shapes.
- **[Optimized and reference inference paths drift]** → Test full-prefix versus cached logits and final states with initial-state continuation; record any specialized decode path as backend provenance.
- **[External package updates change behavior]** → Do not make an external kernel the default; pin any admitted dependency and record its version/source hash after it passes the same qualification suite.
- **[Compilation cost is mistaken for training cost]** → Report build/JIT and warmup separately, cache build products by source/toolchain/architecture identity, and exclude them from steady-state timing while still exposing them operationally.
- **[Performance tuning harms experimental fairness]** → Use the same qualified backend, dtype, and per-scale effective batch for all paired variants; persist those settings and reject incompatible resumption.

## Migration Plan

1. Add backend configuration with `reference` as the default and include its provenance in new manifests.
2. Refactor the eager loop behind the canonical recurrence API and prove no regression with the existing CPU suite.
3. Add qualification artifacts and the V100 profiling command without changing the blocked production step budget.
4. Qualify the college machine and save its toolchain/device report.
5. Build and validate `cuda-fused`; keep it opt-in until all numerical and performance gates pass.
6. Record the accepted backend qualification hash in the production profile, then select the fixed optimizer-step budget from the measured V100 pilot.
7. Run scalar and channel-wise paired jobs with identical backend configuration.

Rollback is selecting `reference` in a new output root. Results from different backends are not merged or resumed as if they were the same run; historical artifacts remain untouched.

## Open Questions

- Does the college V100S host expose `nvcc` and a compatible C++ compiler, or only the CUDA-capable driver reported by `nvidia-smi`? This is resolved by the mandatory toolchain preflight before CUDA-extension work is promoted.
- What share of measured optimizer-step time is recurrence launch/autograd overhead versus tied-output cross entropy and zero-worker tokenization? The baseline profiler determines whether a follow-up optimization is needed after the recurrence change.
- Which state-accumulator mode gives the best acceptable FP16 stability/performance trade-off on V100? Both candidate modes are qualified, and the selected one is frozen in provenance before research training.

