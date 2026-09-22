## ADDED Requirements

### Requirement: Separate deterministic MQAR experiment family
The system SHALL provide a synthetic MQAR training and evaluation entry point independent of TinyStories, using versioned deterministic generation, paired data order across variants, masked query-target loss, and held-out exact-token accuracy.

#### Scenario: Running production MQAR
- **WHEN** the production MQAR experiment is launched for a research seed
- **THEN** it uses the configured 8,192-token symbolic vocabulary, 100,000 training examples, fixed 3,000-example held-out suite, and identical generated examples for every paired variant

### Requirement: Retrieval models isolate recurrent gating
The primary synthetic retrieval experiment SHALL train models from scratch using the study backbone dimensions, synthetic vocabulary, and pure Gated DeltaNet blocks without full-attention retrieval shortcuts.

#### Scenario: Constructing an MQAR backbone
- **WHEN** a named study backbone is selected for MQAR
- **THEN** all token-mixing blocks use the selected scalar or channel-wise Gated DeltaNet mechanism and the report labels it as a backbone scale with its actual synthetic-model parameter count

### Requirement: Retrieval difficulty sweeps
MQAR evaluation SHALL break down exact query accuracy by sequence length, number of key-value pairs, and query distance/depth, including configured in-distribution and length-extrapolation cases through length 512.

#### Scenario: Evaluating a held-out checkpoint
- **WHEN** MQAR evaluation completes
- **THEN** the result contains overall accuracy and per-difficulty breakdowns using the same persisted cases for all variants and seeds

### Requirement: Required separately trained overwrite-recall experiment
The first-submission pipeline SHALL run overwrite recall as a distinct experiment in which a key receives multiple values and the target is its most recent value. It SHALL be trained separately from standard MQAR and SHALL produce independent train/test manifests, checkpoints, metrics, aggregate tables, and plots.

#### Scenario: Testing selective forgetting
- **WHEN** an overwrite-recall query follows an earlier value and a later overwrite for the same key
- **THEN** exact accuracy is computed against the later value and reported independently from standard MQAR, including whether an error selected the stale value

#### Scenario: Building first-submission artifacts
- **WHEN** the required production overwrite grid completes
- **THEN** the paper outputs report latest-value accuracy, stale-value error rate, unrelated-association retention, and breakdowns by overwrite count, overwrite-to-query distance, memory load, and sequence length

### Requirement: Persisted exploratory natural-language retrieval suite
Any TinyStories needle-in-a-haystack evaluation SHALL load a versioned persisted prompt suite and SHALL be labeled exploratory rather than merged with trained MQAR results.

#### Scenario: Comparing TinyStories variants
- **WHEN** natural-language retrieval is run on paired TinyStories checkpoints
- **THEN** each checkpoint receives identical prompts/targets and the output clearly distinguishes this zero-shot diagnostic from synthetic-task accuracy
