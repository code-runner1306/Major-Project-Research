## ADDED Requirements

### Requirement: Load the selected pretrained GDN backbones exactly
The trainer SHALL load scalar seed 43 and channel-wise seed 42 from configured checkpoint paths or the supported checkpoint archive, instantiate the 30M architecture with vocabulary size 50,257, and strictly validate every pretrained model tensor before attaching the decision head.

#### Scenario: Selected checkpoint is compatible
- **WHEN** its variant, architecture, tensor names/shapes, and SHA-256 identity match the locked configuration
- **THEN** the backbone loads strictly and the manifest records its source and checksum

#### Scenario: Checkpoint does not match the selected model
- **WHEN** a checkpoint is missing, corrupt, has the wrong variant, or fails strict state loading
- **THEN** training stops before optimizer creation with a structured compatibility error

### Requirement: Score arbitrary typed option sets with one shared head
The decision model SHALL use a shared projection-based pointer head over the final decision representation and masked pooled candidate spans, with projection width configurable and defaulting to 128.

#### Scenario: A choice decision is scored
- **WHEN** a serialized choice example with K valid candidates is forwarded
- **THEN** the model returns exactly K finite logits and probabilities summing to one

#### Scenario: Score and noul decisions are scored
- **WHEN** serialized ordered-score or yes/no candidates are forwarded
- **THEN** the same shared head produces distributions without a question-specific classifier

#### Scenario: Batch padding is present
- **WHEN** candidate tensors contain padded positions
- **THEN** padded positions receive no probability mass and contribute nothing to the loss

### Requirement: Train against full soft target distributions
The trainer SHALL minimize mean soft-target cross-entropy over valid candidates and SHALL reject non-finite losses or gradients.

#### Scenario: Ambiguous gold distribution is trained
- **WHEN** a target assigns probability mass to multiple candidates
- **THEN** every target probability contributes to `-sum(y * log_softmax(logits))`

#### Scenario: Non-finite training state occurs
- **WHEN** the loss, gradient norm, parameter, or optimizer update becomes non-finite
- **THEN** the lifecycle becomes failed with the step and diagnostic values and no completion artifact is emitted

### Requirement: Apply one locked protocol to both variants
The production orchestrator SHALL use identical split identity, fine-tuning seed, minibatch order, deterministic option permutations, epoch budget, optimizer/scheduler settings, evaluation cadence, and effective batch size for scalar and channel-wise training.

#### Scenario: Paired variants are launched
- **WHEN** both production variants use the same run manifest
- **THEN** all controlled training fields are equal and only variant/pretrained-checkpoint identity and naturally resulting model parameters differ

#### Scenario: Variant-specific hyperparameters are requested
- **WHEN** a production invocation would give one variant a different controlled training field
- **THEN** the orchestrator rejects the run as an invalid paired comparison

### Requirement: Lock hyperparameters before official-test evaluation
The system SHALL support a bounded development comparison of backbone learning rates `1e-5` and `2e-5`, aggregate validation soft NLL across both variants, and persist the selected common configuration before permitting official-test evaluation.

#### Scenario: Development selection completes
- **WHEN** both candidate learning rates have compatible validation results for both variants
- **THEN** the lower aggregate validation-NLL configuration is marked locked with a deterministic tie-break rule

#### Scenario: Test evaluation is attempted before lock
- **WHEN** no complete immutable training configuration exists
- **THEN** the official-test stage refuses to run

### Requirement: Produce best and fixed-budget final models
The trainer SHALL run the configured fixed maximum epoch budget, evaluate validation soft NLL at configured boundaries, save the lowest-validation-NLL model as `best_model`, and save the completed-budget state as `final_model`.

#### Scenario: Validation improves
- **WHEN** a completed validation pass has lower finite soft NLL than every previous pass
- **THEN** the best-model artifact and its metadata are atomically updated

#### Scenario: Fixed training budget completes
- **WHEN** all configured epochs and optimizer steps finish
- **THEN** a distinct final-model artifact records the exact completed budget even if an earlier model remains best

### Requirement: Qualify the training path before a production launch
The pipeline SHALL provide a CPU-safe smoke run and a small GPU overfit/resume preflight that exercise all primitives, variable candidate counts, masked loss, mixed precision where available, checkpoint reload, and deterministic continuation.

#### Scenario: Production preflight passes
- **WHEN** the smoke tests pass, the small GPU sample shows a substantial finite loss decrease, and an interrupted run resumes successfully
- **THEN** the selected hardware/training profile is eligible for production

#### Scenario: Production preflight fails
- **WHEN** any required construction, learning, save/load, or resume assertion fails
- **THEN** the full production run is blocked and the failing gate is reported

