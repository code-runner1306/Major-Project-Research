## ADDED Requirements

### Requirement: Linux Python 3.11 CUDA requirements
The repository SHALL provide `requirements.txt` for Linux x86-64 and Python 3.11 containing all production/test dependencies and a pinned CUDA-enabled PyTorch package resolved through an official PyTorch CUDA wheel index.

#### Scenario: Installing on the college V100 host
- **WHEN** the documented driver preflight confirms compatibility with the configured CUDA wheel channel
- **THEN** `python3.11 -m pip install -r requirements.txt` installs the CUDA-enabled environment without requiring the local CUDA toolkit

### Requirement: Driver compatibility preflight
Installation and runtime documentation SHALL require checking the NVIDIA driver before using the default CUDA wheel and SHALL fail clearly when the driver is below the selected CUDA runtime's minimum rather than silently falling back to CPU.

#### Scenario: Unknown or incompatible V100 driver
- **WHEN** the CUDA environment check cannot confirm driver compatibility
- **THEN** production launch stops with the detected versions and instructions to select a compatible official PyTorch wheel channel

### Requirement: CPU-only end-to-end smoke profile
Smoke mode SHALL use local deterministic toy data, one seed, tiny model dimensions, short contexts, two optimizer steps, minimal evaluation repetitions, and small checkpoints while exercising all experiment families and variants.

#### Scenario: Running the complete smoke suite without CUDA
- **WHEN** the smoke command runs on a CPU-only laptop
- **THEN** it completes scalar, channel-wise, matched-scalar, cached-inference, MQAR, overwrite, resume, reporting, and unsupported-energy paths without downloading TinyStories or instantiating production models

### Requirement: Smoke resource and identity isolation
The complete CPU smoke suite SHALL target completion within five minutes and 2 GB RAM on the development laptop, write only to a clearly identified smoke directory, and include `run_mode: smoke` in its fingerprint.

#### Scenario: Production resumes after smoke testing
- **WHEN** production orchestration scans output directories after smoke artifacts exist
- **THEN** no smoke checkpoint or metric is considered compatible with a production seed

### Requirement: Unified Python compatibility metadata
`pyproject.toml`, lock metadata, runtime checks, and `requirements.txt` SHALL agree on Python 3.11 compatibility and required package versions.

#### Scenario: Validating dependency metadata
- **WHEN** environment consistency tests run
- **THEN** they detect incompatible Python or dependency constraints before an experiment begins
