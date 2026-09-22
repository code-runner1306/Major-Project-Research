## ADDED Requirements

### Requirement: Canonical experiment fingerprints
Every run and seed SHALL have a schema-versioned SHA-256 fingerprint derived from canonical configuration covering experiment family, model/per-layer configuration, variant, seeds, budgets, data/tokenizer or generator identity, evaluation protocol, runtime versions, expected artifacts, and source revision state.

#### Scenario: Configuration changes after a smoke run
- **WHEN** a production run targets a directory containing artifacts whose fingerprint was created with smoke settings
- **THEN** those artifacts are marked incompatible and are not resumed or counted as production results

### Requirement: Atomic lifecycle state transitions
Each seed SHALL maintain an atomically written lifecycle state using `pending`, `training`, `trained`, `evaluating`, `complete`, or `failed`, with timestamps and structured failure details.

#### Scenario: Process interruption during an artifact write
- **WHEN** execution stops while updating lifecycle or metrics data
- **THEN** the previous valid file remains readable and the seed is not incorrectly marked complete

### Requirement: Strict completion validation
A seed SHALL be considered complete only if its fingerprint matches, its fixed budget was reached, its checkpoint exists and loads with a valid checksum, and every mandatory finite metric for its experiment family is present and schema-valid.

#### Scenario: Training metrics exist but evaluation is missing
- **WHEN** a metrics file contains validation results but lacks required inference, energy-support, or retrieval fields
- **THEN** completion validation rejects the seed as complete

### Requirement: Evaluation-only recovery
The orchestrator SHALL resume a compatible `trained` seed from its checkpoint and run missing evaluation stages without repeating completed training.

#### Scenario: Crash after checkpoint creation
- **WHEN** a compatible checkpoint and completed training budget exist but the lifecycle state is `trained`
- **THEN** resumption loads the checkpoint, performs required evaluations, and advances through `evaluating` to `complete`

### Requirement: Derived global reporting
Global trackers, summaries, and plots SHALL be rebuilt from validated per-seed artifacts and SHALL NOT be treated as authoritative resume state.

#### Scenario: Global tracker is stale
- **WHEN** per-seed artifacts and a pre-existing global tracker disagree
- **THEN** the pipeline reconstructs global reporting from compatible validated seeds and records excluded artifacts with reasons

