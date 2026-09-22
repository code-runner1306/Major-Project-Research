## ADDED Requirements

### Requirement: Canonical recurrence semantics
The system SHALL expose one recurrence contract whose reference and optimized implementations compute the same state update, output ordering, initial-state continuation, scalar decay broadcasting, and channel-wise decay broadcasting without changing projections, convolutions, normalization, output gating, or model parameters.

#### Scenario: Scalar decay execution
- **WHEN** alpha has shape `[B,T,H,1]`
- **THEN** every backend broadcasts each head's scalar decay over all state rows and returns outputs and a final state with the canonical shapes

#### Scenario: Channel-wise decay execution
- **WHEN** alpha has shape `[B,T,H,D]`
- **THEN** every backend applies independent row/channel decay values using `[B,H,D,1]` broadcasting and does not reduce them to a head scalar

#### Scenario: Initial-state continuation
- **WHEN** a valid `[B,H,D,D]` initial state is supplied
- **THEN** the backend continues from that state and returns the final state after all supplied tokens

### Requirement: Portable reference backend
The system SHALL retain an eager PyTorch reference backend that supports CPU execution, CUDA execution, autograd, initial state, final state, scalar decay, and channel-wise decay and serves as the correctness oracle for optimized implementations.

#### Scenario: CPU smoke execution
- **WHEN** the offline CPU smoke profile is executed on a machine without CUDA or the extension toolchain
- **THEN** it selects the reference backend, imports no mandatory CUDA extension, and completes all existing smoke experiment families

#### Scenario: Reference gradient execution
- **WHEN** differentiable recurrence inputs are processed by the reference backend
- **THEN** PyTorch autograd produces gradients for Q, K, V, alpha, beta, and any differentiable initial state

### Requirement: Volta-compatible optimized backend
The system SHALL provide a `cuda-fused` backend whose compiled code explicitly supports compute capability 7.0 and whose forward and backward paths support every production head dimension, scalar and channel-wise decay, sequence length 512, and the selected production mixed-precision mode.

#### Scenario: Supported V100 execution
- **WHEN** `cuda-fused` is explicitly requested on a qualified V100S with a supported shape and dtype
- **THEN** the system executes the optimized forward/backward implementation and reports `cuda-fused` as the resolved backend

#### Scenario: Unsupported shape or dtype
- **WHEN** `cuda-fused` receives a shape, dtype, or state layout outside its qualified support matrix
- **THEN** it raises an actionable error before training instead of producing an approximate result or selecting another backend

### Requirement: Fail-closed production backend selection
Production runs MUST explicitly name a recurrence backend and MUST NOT silently fall back when the requested backend is unavailable, unqualified, or incompatible with the active GPU.

#### Scenario: Missing optimized extension
- **WHEN** a production profile requests `cuda-fused` but the extension cannot be loaded
- **THEN** preflight fails before dataset loading or run lifecycle transition and reports the missing build/runtime prerequisite

#### Scenario: Automatic selection requested for production
- **WHEN** a production command requests an automatic or unspecified optimized backend
- **THEN** the system refuses to start and instructs the operator to select a qualified explicit backend

#### Scenario: Reference deliberately selected
- **WHEN** a production profile explicitly requests `reference`
- **THEN** the run may proceed after recording that deliberate choice and does not claim optimized qualification or speedup

### Requirement: Cache and diagnostics preservation
Optimized recurrence execution SHALL preserve prefill/decode state continuity, full-versus-cached output equivalence, gate-observer inputs, and the separation of diagnostics from timed and energy regions.

#### Scenario: Cached continuation
- **WHEN** a prompt is prefetched with the optimized backend and subsequent tokens are decoded from its returned cache
- **THEN** concatenated cached logits agree with a full-prefix execution within the qualified dtype tolerance

#### Scenario: Gate diagnostics enabled
- **WHEN** a gate observer is registered while either backend processes a batch
- **THEN** the observer receives the original projected alpha tensor with the correct scalar or channel-wise shape before recurrence execution

### Requirement: Backend provenance and resumption integrity
Every run manifest and result SHALL persist the requested and resolved backend, backend semantic/build identity, compiled architecture set, input and state-accumulator dtypes, chunk/checkpoint policy, and qualification artifact hash, and these values SHALL participate in compatibility fingerprints.

#### Scenario: Matching optimized resume
- **WHEN** an optimized run is resumed with identical backend provenance and a valid qualification artifact
- **THEN** normal lifecycle and checkpoint validation may continue

#### Scenario: Backend build drift
- **WHEN** the extension source/build identity, chunk policy, accumulator dtype, or qualification hash differs from an existing run
- **THEN** the existing run is rejected as incompatible and is not overwritten or resumed

#### Scenario: Legacy production artifact
- **WHEN** a production artifact lacks recurrence-backend provenance
- **THEN** it remains readable as historical output but is not accepted as a resumable optimized production run

