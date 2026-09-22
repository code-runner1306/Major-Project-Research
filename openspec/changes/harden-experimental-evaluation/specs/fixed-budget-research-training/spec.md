## ADDED Requirements

### Requirement: Equal fixed training budgets
Production comparisons SHALL train every paired variant for a predetermined equal number of optimizer steps and processed tokens. Validation-based early stopping SHALL NOT terminate a successful headline run before that budget.

#### Scenario: Validation stops improving
- **WHEN** validation loss fails to improve during a production seed
- **THEN** training continues to the configured fixed budget while diagnostic best-checkpoint tracking may continue

### Requirement: Final checkpoint is the primary evaluation checkpoint
The system SHALL evaluate and report the fixed-budget final checkpoint as the primary result and MAY report a best-validation checkpoint only as a separately labeled secondary result.

#### Scenario: Best validation occurs before the final step
- **WHEN** a model reaches its lowest validation loss before completing its fixed budget
- **THEN** the headline metrics still use the final checkpoint and any best-checkpoint metrics are explicitly secondary

### Requirement: Fully deterministic paired randomness
The system SHALL seed Python, NumPy, PyTorch CPU, and CUDA randomness and SHALL pair ordered training examples and persisted evaluation examples across all variants sharing a research seed.

#### Scenario: Repeating a seed manifest
- **WHEN** two variants use the same recorded data and model seed configuration
- **THEN** they consume the same ordered examples and evaluation cases, with their seed sources recorded in the manifest

### Requirement: Descriptive three-seed reporting
The reporting pipeline SHALL expose every seed result, mean, standard deviation, and paired variant differences, and SHALL NOT display null-hypothesis significance stars for the configured three-seed study.

#### Scenario: Generating the paper summary
- **WHEN** three paired seeds are aggregated
- **THEN** the report contains raw and descriptive paired results without claiming statistical significance from Wilcoxon or minimum-p-value selection

