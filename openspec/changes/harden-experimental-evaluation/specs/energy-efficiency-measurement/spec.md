## ADDED Requirements

### Requirement: NVIDIA GPU energy measurement
On supported NVIDIA hardware, the system SHALL measure total GPU energy for named training and inference regions using NVML total-energy counter deltas, with sampled-power integration as a fallback when the energy counter is unavailable.

#### Scenario: Energy counter is supported
- **WHEN** an instrumented region runs on a GPU exposing the NVML total-energy counter
- **THEN** the result records joules from the counter delta and identifies the counter-based method

#### Scenario: Only instantaneous power is supported
- **WHEN** the total-energy counter is unavailable but power sampling is available
- **THEN** the result records numerically integrated joules, sampling interval, average watts, peak watts, and identifies the sampling method

### Requirement: Normalized energy-efficiency metrics
The system SHALL report training joules per million processed tokens, prefill joules per processed token, and cached-decode joules per generated token in addition to total energy and wall time.

#### Scenario: Producing energy metrics
- **WHEN** a training or inference region completes with supported energy measurement
- **THEN** its metrics contain both total energy and the protocol-appropriate normalized energy value

### Requirement: Reproducible energy metadata
Energy artifacts SHALL record GPU identity, driver and CUDA versions, power limit, dtype, batch/sequence configuration, measurement method, and whether the device was shared or measurement isolation was verified.

#### Scenario: Comparing energy results
- **WHEN** energy metrics are saved
- **THEN** the artifact includes enough device and protocol metadata to reject cross-hardware or incompatible comparisons

### Requirement: Graceful unsupported behavior
CPU-only and unsupported environments SHALL complete all non-energy experiment paths and SHALL emit a structured unsupported-energy result rather than zero energy or an exception.

#### Scenario: CPU smoke test
- **WHEN** the energy collector initializes without CUDA/NVML
- **THEN** it returns `supported: false` with a reason and the smoke pipeline continues successfully

