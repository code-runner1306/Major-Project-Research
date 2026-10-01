## ADDED Requirements

### Requirement: Fingerprint every decision research run
The system SHALL derive a schema-versioned canonical SHA-256 fingerprint from source state, model/variant configuration, pretrained-checkpoint checksum, data/split identity, tokenizer and serialization rules, training/evaluation configuration, seeds, runtime versions, expected artifacts, and pinned baseline identity.

#### Scenario: Compatible stage resumes
- **WHEN** the requested configuration produces the same fingerprint as an incomplete run
- **THEN** the orchestrator may reuse its validated artifacts and continue from its lifecycle state

#### Scenario: Configuration changed
- **WHEN** any fingerprinted field differs from the existing run
- **THEN** existing artifacts are preserved but rejected for resume with a field-level incompatibility report

### Requirement: Record atomic lifecycle transitions
Each variant SHALL maintain an atomically replaced lifecycle document using `pending`, `training`, `trained`, `calibrating`, `calibrated`, `evaluating`, `complete`, or `failed`, including UTC timestamps and structured failure details.

#### Scenario: Process stops during lifecycle write
- **WHEN** execution is interrupted before an atomic replacement completes
- **THEN** the previous valid lifecycle document remains readable and the run is not falsely marked complete

#### Scenario: Stage fails
- **WHEN** an unhandled stage error occurs
- **THEN** the lifecycle records `failed`, the last valid stage/step, exception type, and concise reason without deleting recoverable artifacts

### Requirement: Save two rolling resumable checkpoints
The trainer SHALL alternate `resume_A.pt` and `resume_B.pt` at configured optimizer-step and epoch boundaries, write through a local temporary file, compute a checksum, copy and validate the inactive slot, and update the latest pointer only after validation.

#### Scenario: Checkpoint save succeeds
- **WHEN** the inactive slot is fully copied, reloadable, and checksum-valid
- **THEN** the latest pointer atomically names that slot and the previous slot remains as fallback

#### Scenario: Colab disconnects during checkpoint copy
- **WHEN** the inactive slot is partial or invalid
- **THEN** the latest pointer continues to reference the previous valid slot

### Requirement: Persist exact optimizer-boundary continuation state
A resumable checkpoint SHALL contain backbone/head, optimizer, scheduler, mixed-precision scaler, completed epoch/optimizer step, gradient-accumulation boundary, sampler position, random-generator states, and best-validation state, and SHALL only be written after a completed optimizer update.

#### Scenario: Training resumes from latest
- **WHEN** a compatible valid rolling checkpoint is selected
- **THEN** training continues with the same next minibatch, option permutation, learning rate, and random state that uninterrupted execution would use

#### Scenario: Newest slot is invalid
- **WHEN** the pointer target fails checksum or load validation but the alternate slot is valid
- **THEN** the system resumes from the alternate slot and records the fallback

### Requirement: Export validated model-only checkpoints
The trainer SHALL write checksummed `best_model.pt` and `final_model.pt` artifacts containing the model/head configuration, model tensors, label/schema metadata, source run fingerprint, selected step, and validation summary without optimizer state.

#### Scenario: Model-only checkpoint is loaded for evaluation
- **WHEN** its checksum and source fingerprint match the evaluation manifest
- **THEN** the evaluator reconstructs the dynamic decision model strictly and produces predictions without the training checkpoint

### Requirement: Resume completed training at later stages
The orchestrator SHALL derive progress from validated per-stage artifacts and SHALL continue a `trained` run at calibration or a `calibrated` run at evaluation without repeating training.

#### Scenario: Training completed before interruption
- **WHEN** best/final checkpoints and the completed training budget validate but calibration artifacts are absent
- **THEN** resume starts at calibration

#### Scenario: Metrics exist without valid predictions
- **WHEN** summary metrics are present but required immutable prediction rows are missing, incompatible, or corrupt
- **THEN** completion validation fails and evaluation is rerun from the validated model/calibration artifacts

### Requirement: Support bounded external output storage
The pipeline SHALL accept a configurable output root, retain no more than two rolling resumable checkpoints per variant, and package final research artifacts without duplicating the source pretrained checkpoints.

#### Scenario: Mounted Drive output is configured
- **WHEN** a Colab run uses a mounted Drive directory
- **THEN** active downloads and minibatch data remain local while durable manifests, checkpoints, predictions, metrics, and exports are written through the crash-safe output process

### Requirement: Preserve cumulative fine-tuning resource cost
The trainer SHALL record wall-clock duration, optimizer-step range/count, training precision, physical/effective batch size, context limit, runtime/device identity, GPU activity observations, and optional GPU energy for each fine-tuning session. Resumed sessions SHALL be retained and their wall time and supported energy SHALL be aggregated per variant; completed optimizer steps SHALL not be double-counted. The record SHALL explicitly exclude energy and time for the already-completed TinyStories pretraining. Missing energy or activity data SHALL remain null/unknown and SHALL NOT block training.

#### Scenario: Training completes across one or more sessions
- **WHEN** a scalar or channel-wise fine-tune completes, including after resume
- **THEN** the training resource artifact records all sessions, cumulative fine-tuning wall time, final optimizer-step count, and cumulative energy when every session has supported readings

#### Scenario: Telemetry is unavailable
- **WHEN** a session cannot read GPU energy or activity
- **THEN** model training and checkpointing continue, and the session cost artifact preserves the unavailable reason without fabricating values
