## ADDED Requirements

### Requirement: Layer-wise decay-gate summaries
The system SHALL collect online per-layer/head decay-gate summaries on a fixed persisted diagnostic corpus, including mean, standard deviation, configured quantiles, fractions near zero and one, and estimated decay half-life distributions.

#### Scenario: Evaluating gate behavior
- **WHEN** a trained scalar or channel-wise checkpoint runs on the diagnostic corpus
- **THEN** the result contains reproducible layer-wise gate summaries without storing all per-token activations

### Requirement: Channel diversity reporting
For channel-wise models, the system SHALL report within-head channel variance/diversity and SHALL label scalar-gate channel diversity as not applicable.

#### Scenario: Comparing gating granularity
- **WHEN** gate diagnostics are aggregated across paired variants
- **THEN** reports distinguish cross-time/head behavior from channel-specific variation and do not fabricate a channel-diversity value for scalar gates

### Requirement: Training-stability and static-cost diagnostics
Training SHALL record gradient norms, non-finite/loss-spike events, optimizer-step time, processed tokens, and tokens per second, while static analysis SHALL report parameter components, recurrent/cache state sizes, and theoretical gate/projection operation overhead.

#### Scenario: Completing a fixed-budget seed
- **WHEN** training and diagnostic analysis complete
- **THEN** the seed artifacts contain stability, throughput, and static-cost summaries suitable for interpreting quality and energy differences

### Requirement: Diagnostics excluded from timed regions
Gate hooks and diagnostic aggregation SHALL run outside latency, throughput, and energy benchmark regions.

#### Scenario: Measuring cached decode
- **WHEN** a performance benchmark begins
- **THEN** gate diagnostic hooks are disabled and their collection cost is not included in the benchmark result
