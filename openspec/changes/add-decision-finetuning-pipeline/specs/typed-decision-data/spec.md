## ADDED Requirements

### Requirement: Load every official Typed Decisions workflow
The data pipeline SHALL load pinned revisions of `agent_trace_observability`, `customer_service`, `invoice_processing`, and `security_incidents`, validate 300 training and 100 official-test cases per workflow, and validate five typed questions per case before producing an experiment manifest.

#### Scenario: Expected release is loaded
- **WHEN** all four pinned workflow configurations match their expected schemas and cardinalities
- **THEN** the pipeline records the dataset revision, fingerprint, workflow counts, question schemas, and content hashes

#### Scenario: Dataset identity has changed
- **WHEN** a workflow has an unexpected schema, cardinality, revision, or content identity
- **THEN** the pipeline stops before training and reports the incompatible field

### Requirement: Partition development data at case level
The pipeline SHALL deterministically divide each workflow's 300 training cases into 240 training, 30 validation, and 30 calibration cases and SHALL keep every question from a case in the same partition.

#### Scenario: Split manifest is created
- **WHEN** the configured split seed is applied to the four workflows
- **THEN** the persisted manifest contains 960 training, 120 validation, and 120 calibration case IDs with no overlap

#### Scenario: Split manifest is reused
- **WHEN** a later stage loads an existing compatible split manifest
- **THEN** it uses the stored case assignments instead of generating new assignments

#### Scenario: Leakage is detected
- **WHEN** any case ID or normalized state content appears in more than one partition
- **THEN** validation fails and no training or evaluation stage starts

### Requirement: Expand cases into typed decision examples
The pipeline SHALL expand each case into one example per question while preserving workflow, case ID, question ID, primitive, instructions, ordered option labels/descriptions, soft target distribution, and score ordering metadata.

#### Scenario: All training decisions are prepared
- **WHEN** the 960-case training partition is expanded
- **THEN** it contains exactly 4,800 decision examples covering `choice`, `score`, and `noul`

#### Scenario: Noul is normalized
- **WHEN** a `noul` question is expanded
- **THEN** it is represented as explicit yes/no candidates with a normalized two-element target distribution

#### Scenario: Invalid probability target is encountered
- **WHEN** a target contains non-finite, negative, missing, or non-normalizable option probabilities
- **THEN** preparation fails with the workflow, case ID, and question ID

### Requirement: Serialize dynamic options without changing the tokenizer vocabulary
The serializer SHALL use the pretrained GPT-2 tokenizer vocabulary, produce token IDs plus question/decision positions and candidate spans, and SHALL support variable candidate counts without adding embedding rows.

#### Scenario: Variable option counts are collated
- **WHEN** a batch contains decisions with different numbers of candidates
- **THEN** candidate logits and targets are padded with an explicit mask and padded candidates cannot affect loss or predictions

#### Scenario: Candidate span is invalid
- **WHEN** tokenization produces an empty, overlapping, truncated, or out-of-range candidate span
- **THEN** serialization rejects the example before it reaches the model

### Requirement: Preserve questions and options under the context budget
The serializer SHALL preserve the complete question, every option, and the decision marker and SHALL truncate only state content using a deterministic head-and-tail policy when the configured context budget is exceeded.

#### Scenario: Input fits within the budget
- **WHEN** the complete serialized decision is no longer than the configured maximum length
- **THEN** no state tokens are removed and the example is marked untruncated

#### Scenario: State requires truncation
- **WHEN** state plus the mandatory decision suffix exceeds the configured maximum length
- **THEN** the serializer retains deterministic state head and tail segments, preserves the complete suffix, and records original/retained lengths and a truncation flag

#### Scenario: Mandatory decision content is too long
- **WHEN** question, options, and required delimiters alone exceed the context budget
- **THEN** preparation fails rather than removing or shortening a candidate

### Requirement: Audit and freeze data identity before production training
The preparation stage SHALL emit a schema-versioned data manifest containing split identities, serialization version, tokenizer/version, context budget, length distribution, truncation rate, primitive/option-count distributions, and a canonical hash used by downstream fingerprints.

#### Scenario: The 512-token profile satisfies the fit gate
- **WHEN** at least 99 percent of training decisions fit without state truncation
- **THEN** the 512-token profile is eligible to be locked for production

#### Scenario: The fit gate fails
- **WHEN** fewer than 99 percent of training decisions fit without state truncation
- **THEN** production training is blocked until a larger context profile is explicitly qualified and recorded

