## ADDED Requirements

### Requirement: Same-backbone comparison remains the primary ablation
The system SHALL construct scalar and channel-wise models with identical hidden dimensions, layer counts, head counts, attention layouts, and baseline FFN widths, and SHALL report their actual trainable parameter counts and percentage overhead.

#### Scenario: Constructing a primary paired comparison
- **WHEN** a scale is scheduled with the default variants
- **THEN** the scalar and channel-wise models share all backbone dimensions while the report records the channel-wise gate parameter overhead

### Requirement: Opt-in exactly parameter-matched scalar control
The system SHALL expose `ENABLE_PARAMETER_MATCHED_SCALAR_CONTROL` and, when enabled, SHALL add a `scalar-matched` model whose useful FFN capacity is expanded so its total trainable parameter count exactly equals the paired channel-wise model without unused parameters.

#### Scenario: Enabling the matched control
- **WHEN** `ENABLE_PARAMETER_MATCHED_SCALAR_CONTROL` is true for any configured backbone
- **THEN** the orchestrator schedules scalar, channel-wise, and scalar-matched variants and asserts exact parameter equality between channel-wise and scalar-matched before training

### Requirement: Persist matched-control construction details
The system SHALL record the control strategy, actual parameter count, gate parameter count, and per-layer FFN widths in every run manifest and result artifact.

#### Scenario: Reviewing a matched-control result
- **WHEN** a scalar-matched seed completes
- **THEN** its artifacts contain sufficient parameter and per-layer configuration details to reproduce the exact matched model

