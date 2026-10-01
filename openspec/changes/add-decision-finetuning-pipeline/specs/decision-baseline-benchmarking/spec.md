## ADDED Requirements

### Requirement: Execute a pinned Laya Typed-Decisions baseline locally
The baseline adapter SHALL load a pinned revision of `convaiinnovations/laya-typed-decisions`, record package/model revisions and checksums, keep the model resident, and produce predictions for the same normalized official-test decisions evaluated for GDN.

#### Scenario: Laya baseline completes
- **WHEN** the pinned model and adapter are compatible with the runtime
- **THEN** predictions cover every expected official-test case/question and use the shared canonical option labels

#### Scenario: Baseline revision is unpinned or changes
- **WHEN** model files or adapter dependencies do not match the locked baseline identity
- **THEN** baseline execution stops and the run cannot be packaged as a complete direct comparison

### Requirement: Normalize baseline outputs into the shared prediction schema
The adapter SHALL map Laya's `choice`, `score`, and `noul` outputs to complete per-option distributions and preserve both native raw/calibrated outputs and any common held-out recalibration as separately named channels.

#### Scenario: Native baseline prediction is adapted
- **WHEN** Laya returns a valid typed answer
- **THEN** every canonical candidate receives a finite probability, probabilities normalize within tolerance, and the original response metadata remains traceable

#### Scenario: Baseline omits or invents an option
- **WHEN** returned labels do not bijectively match the supplied canonical candidates
- **THEN** the decision is marked invalid and the baseline artifact fails completeness validation

### Requirement: Score every local model through one evaluator
Scalar GDN, channel-wise GDN, and Laya prediction rows SHALL be evaluated by the same metric definitions, calibration reporting rules, grouping logic, and paired case identities.

#### Scenario: Comparable quality table is produced
- **WHEN** all three complete prediction artifacts validate
- **THEN** one table reports identical metric fields and paired case coverage for every model

### Requirement: Benchmark efficiency on the same device and protocol
The benchmark SHALL run models sequentially on one recorded device/runtime with batch size 1, fixed examples, 20 warmups, at least 200 CUDA-synchronized observations, and diagnostics disabled inside timed regions.

#### Scenario: Efficiency benchmark completes
- **WHEN** all required local models run under the locked protocol
- **THEN** the report records parameters, checkpoint bytes, peak allocated/reserved memory, median and p95 latency, decisions per second, and five-question cases per second

#### Scenario: Hardware or protocol differs
- **WHEN** a result was measured on another device, precision, batch size, input set, or timing method
- **THEN** it is excluded from the direct same-hardware comparison and labeled separately

### Requirement: Preserve per-decision latency and runtime conditions
The evaluator SHALL save synchronized per-decision latency alongside primitive, full serialized input-token length, state-length/truncation metadata, and option count. It SHALL emit latency summaries overall and grouped by primitive, serialized-length bucket, and option count. The same artifacts SHALL record the model dtype/precision, batch size, warm-up and timed-decision counts, configured maximum context, software/runtime versions, device identity, and best-effort GPU utilization, memory, clock, power, and other-compute-process observations. Unavailable process/activity telemetry SHALL be explicit and MUST NOT be represented as proof that no other activity occurred.

#### Scenario: GDN and baseline latency are collected
- **WHEN** official-test inference runs for scalar GDN, channel-wise GDN, or Laya
- **THEN** prediction rows include one measured latency per decision and the report includes primitive, input-length, and option-count group summaries

#### Scenario: Device activity cannot be inspected
- **WHEN** NVML or process-list access is unavailable
- **THEN** model evaluation completes and the resource artifact records unknown/unavailable activity rather than assuming an idle device

### Requirement: Preserve optional evaluation energy telemetry
The evaluation pipeline SHOULD record GPU-device energy for canonical official-test inference by each GDN variant and Laya, and separately record GDN option-order robustness inference energy, whenever NVML exposes a supported energy counter or power sampling. Energy telemetry SHALL be written as a separate machine-readable artifact, included in the results package, and SHALL NOT be required for evaluation or alter quality metrics. Unsupported readings SHALL be recorded with a reason and null energy values; the system MUST NOT infer energy from rated power or report GPU-device readings as process-isolated energy.

#### Scenario: Energy counter is supported
- **WHEN** NVML exposes the total energy counter for the evaluation device
- **THEN** the artifact records the counter-difference joules, stage scope, device/runtime metadata, and measurement timestamps

#### Scenario: Only power sampling is supported
- **WHEN** NVML power readings are available but the total-energy counter is not
- **THEN** the artifact records joules integrated from timestamped power samples and identifies the method and sampling interval as an estimate

#### Scenario: Energy telemetry is unavailable
- **WHEN** NVML is missing, denies access, or exposes neither supported energy nor power readings
- **THEN** evaluation continues normally and the artifact records `supported: false`, a concise reason, and null joules

### Requirement: Separate direct and contextual baselines
The final report SHALL treat locally executed Laya as the required direct baseline, SHALL mark Kev execution as optional, and SHALL label published Jev values as contextual third-party results that are excluded from paired statistical comparisons.

#### Scenario: Published Jev number is included
- **WHEN** a paper table cites a Jev metric not generated by this pipeline
- **THEN** the table identifies its external source/protocol and does not compute a paired difference or significance interval against it

### Requirement: Permit report packaging only after baseline validation
The final packaging stage SHALL require complete compatible scalar, channel-wise, and Laya prediction/metric artifacts and a same-device efficiency artifact, while allowing optional baseline artifacts to be absent.

#### Scenario: Core comparison is complete
- **WHEN** all required model, metric, robustness, calibration, and efficiency artifacts validate
- **THEN** the pipeline creates a checksummed compressed research-results archive and an artifact inventory

#### Scenario: Optional Kev results are absent
- **WHEN** the required GDN and Laya artifacts are complete but no Kev artifact exists
- **THEN** packaging succeeds and records Kev as not run
