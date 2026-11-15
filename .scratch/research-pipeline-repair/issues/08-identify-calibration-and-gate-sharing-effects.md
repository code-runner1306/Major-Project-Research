# 08: Make calibration and gate-sharing conclusions identifiable

**What to build:** Calibration and inference interventions produce interpretable evidence about memory initialization and gate sharing, with treatment changes and measurement conditioning explicitly recorded.

**Blocked by:** 04: Establish a defensible architecture baseline and parameter accounting; 05: Build an overwrite task that requires keyed retrieval; 06: Build an MQAR task with meaningful query and context difficulty.

**Status:** package-treatment implementation committed; timescale decomposition not claimed; regressions pending test-seam approval

- [ ] Calibration uses only training data with explicit sampling, weighting and padding masks; write/query/filler states are reported separately.
- [ ] Artifacts identify every treatment change, including bias/timescale fitting and within-head projection-function matching. A package effect is not labeled a timescale-only causal effect.
- [ ] Realized retention distributions, layer/head residuals and input sensitivity are reported; half-life derived from mean retention is labeled a decay diagnostic rather than measured full-state memory lifetime.
- [ ] If timescale-specific claims are enabled, bounded decomposition controls separately exercise bias calibration retaining diverse rows and function matching without the complete timescale change.
- [ ] Posthoc gate sharing uses identical checkpoints and evaluation cases for all settings, including an unmodified reference.
- [ ] Effective-retention averaging and log-retention averaging have accurate descriptions, distinct identities and bounded sensitivity checks when conclusions depend on the convention.
- [ ] Reports distinguish inference-time sensitivity from training-time effects, independently trained grouped models and removal of projection cost.
- [ ] A small paired workflow verifies calibration provenance, matched behavior within declared tolerances and reproducible intervention deltas.
