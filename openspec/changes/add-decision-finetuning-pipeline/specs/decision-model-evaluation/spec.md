## ADDED Requirements

### Requirement: Select checkpoints using validation data only
The evaluator SHALL use validation soft-label NLL to select `best_model` and SHALL prevent official-test labels or metrics from influencing hyperparameters, context budget, stopping, checkpoint selection, or temperature fitting.

#### Scenario: Best checkpoint is selected
- **WHEN** production training has multiple completed validation evaluations
- **THEN** the finite checkpoint with minimum validation soft NLL is selected using the documented deterministic tie-break rule

#### Scenario: Test access would influence configuration
- **WHEN** an invocation requests test metrics before the run is locked and calibrated
- **THEN** the evaluator refuses access and records no test predictions

### Requirement: Fit held-out temperature calibration
The calibration stage SHALL fit positive temperature values on the 600-decision calibration partition by minimizing soft NLL, preserve raw logits, and produce separately identified raw and calibrated probabilities.

#### Scenario: Primitive-specific calibration succeeds
- **WHEN** each of `choice`, `score`, and `noul` has valid calibration rows
- **THEN** one finite positive temperature per primitive is saved with objective values and optimizer settings

#### Scenario: Calibration artifact is incompatible
- **WHEN** its model fingerprint, split hash, logits hash, or calibration schema differs from the evaluation run
- **THEN** it is rejected and refitted without consulting test data

### Requirement: Compute paper-grade decision metrics
The evaluator SHALL compute schema-versioned top-1 accuracy, soft accuracy, soft-label NLL, multiclass Brier score, ECE, and score expected-value MAE from immutable prediction rows and SHALL document all reductions and binning conventions.

#### Scenario: Official test evaluation completes
- **WHEN** predictions exist for all 2,000 decisions from 400 official-test cases
- **THEN** raw and calibrated aggregate metrics plus workflow, primitive, option-count, and truncation breakdowns are emitted

#### Scenario: Prediction coverage is incomplete
- **WHEN** any expected case/question pair is missing, duplicated, non-finite, or has probabilities that do not normalize within tolerance
- **THEN** metrics are rejected as incomplete

### Requirement: Measure option-order robustness
The evaluator SHALL score canonical option order and at least three deterministic permutations while mapping probabilities back to canonical labels.

#### Scenario: Permutation evaluation completes
- **WHEN** all configured permutations are available for one model
- **THEN** the report includes option flip rate and changes in accuracy, NLL, Brier, and ECE relative to canonical order

#### Scenario: Labels are remapped incorrectly
- **WHEN** a permuted prediction cannot be bijectively mapped to canonical candidates
- **THEN** robustness evaluation fails rather than comparing positional indices

### Requirement: Estimate paired uncertainty by case
The reporting stage SHALL use a deterministic paired bootstrap that samples the 400 test cases with replacement and keeps all five decisions from each sampled case together.

#### Scenario: Comparison report is generated
- **WHEN** compatible predictions exist for scalar, channel-wise, and Laya
- **THEN** the report includes 95 percent intervals for model metrics and paired scalar-minus-channel-wise and GDN-minus-Laya differences

#### Scenario: Decision-level bootstrap is requested for headline results
- **WHEN** a configuration would independently resample the 2,000 correlated decisions
- **THEN** the report generator rejects it as an invalid headline uncertainty protocol

### Requirement: Preserve immutable predictions and regenerate derived reports
Every prediction row SHALL identify run/model fingerprint, partition, workflow, case/question IDs, primitive, canonical labels, raw logits/probabilities, calibrated probabilities, gold distribution, option order, truncation metadata, and timing metadata where applicable.

#### Scenario: Tables and figures are rebuilt
- **WHEN** validated prediction and efficiency artifacts are supplied
- **THEN** aggregate JSON, CSV/Markdown tables, reliability diagrams, metric comparison plots, and robustness plots are deterministically regenerated without model inference

### Requirement: Distinguish feasibility from scientific success
The final report SHALL state the selected-checkpoint and synthetic-teacher limitations and SHALL emit complete results whether the GDN models outperform, trade quality for efficiency, or are dominated by the baseline.

#### Scenario: GDN underperforms Laya
- **WHEN** locally measured GDN quality metrics are lower
- **THEN** the report retains the negative result, confidence intervals, calibration, and efficiency measurements without changing the locked protocol

