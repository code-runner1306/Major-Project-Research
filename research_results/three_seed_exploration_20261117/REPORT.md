# Three-seed exploration: MQAR and overwrite

Evidence: maink39a416d, pulled17November2026. All24models (two tasks x three
model/data seeds42..44 x four conditions) completed16epochs/3136accepted steps.
The scheduler still has a64epoch horizon. These are exploratory DEVELOPMENT
results, not a completed64epoch study or untouched-confirmation evaluation.

## Integrity and measured cost

- All12two-worker pairs completed training;75,264accepted optimizer steps total.
- All24data audits pass, with zero exact train/development overlap.
- Within every task/seed, all four training/development manifests and shared
  backbone initialization hashes match. Within each gate type, default and
  calibrated base parameter hashes match. Source identities agree within each
  comparison. No duplicate conditions were found.
- All12calibrated fits reached the declared target means (largest error9.99e-8),
  and all12calibrated backend trajectory qualifications passed.
- All24final checkpoints are present. Per-model reports indicate final16epoch
  counters, not partially trained models. The reproducible analysis additionally
  checks saved checkpoint counters/settings against the summaries.
- Summed pair training wall time:11.28A6000-12Qhours, approximately11h17m.
  Individual models took54.2..57.0minutes; summed model time22.32hours reflects
  two workers, not twice the elapsed GPU session budget. Installation/upload
  overhead is outside those pair training totals. Earlier root publication
  failures do not invalidate completed pair reports; maink now contains all runs.

## Primary learning-curve results

Scores below are normalized trapezoidal AULC over epochs1..16, expressed on a
0..100scale. MQAR uses answer-free query accuracy; overwrite uses unrelated
association retention. These are different outcomes and should not be pooled.

| Task | Condition | Mean score | Sample SD across3seeds |
|---|---|---:|---:|
| MQAR | Scalar default |93.71|3.91|
| MQAR | Channel default |91.16|2.90|
| MQAR | Scalar calibrated |93.63|3.18|
| MQAR | Channel calibrated |93.73|2.51|
| Overwrite | Scalar default |91.00|10.36|
| Overwrite | Channel default |83.80|14.22|
| Overwrite | Scalar calibrated |82.23|4.22|
| Overwrite | Channel calibrated |82.51|14.96|

All12MQAR models finish between99.90%and100%query accuracy. Mean final accuracy
is99.935%,99.951%,99.951%,99.967%in the condition order shown. These ceiling
scores are not a strong final-performance separation.

Overwrite mean final overall accuracies, in that order, are99.32%,99.12%,97.79%,
97.75%. All overwrite models finish with100%updated-value accuracy and0%stale
errors except calibrated scalar seed44:99.805%updated accuracy,0.195%stale error
(one stale answer among512latest-value queries). Most differences concern
unrelated retention, not recovering the updated value.

## MQAR: the repeatable signal

| Seed | Scalar calibration effect | Channel calibration effect | Factorial interaction |
|---|---:|---:|---:|
|42|+0.37|+2.22|+1.85|
|43|-1.81|+2.45|+4.26|
|44|+1.21|+3.05|+1.83|

All values are AULC percentage-point differences. Interaction is
`(channel_cal - scalar_cal) - (channel_default - scalar_default)`.

Default scalar learns faster than default channel in all three seeds. Calibration
improves channel AULC in all three seeds, with mean improvement2.57points; scalar
changes are mixed and average-0.08points. The default mean scalar advantage2.55
points becomes a near tie after calibration (channel+0.10points). The interaction
is positive in all three seeds, mean+2.65points. This supports a limited
hypothesis that the early scalar/channel gap in this easy MQAR cell depends on
initialization, rather than establishing a general granularity ranking.

The gain is principally an early-learning effect. At epoch2, channel accuracy
changes26.90->59.91%in seed42 and35.01->71.63%in seed43. In seed44, the main
transition moves at epoch3 from28.12%default to74.41%calibrated. Nearly all
conditions saturate by epochs3..4. The coarse three-consecutive95%criterion
starts at the same epoch within each seed (epoch3for42/43,epoch4for44), so it
does not independently separate these configurations. AULC was the declared
primary metric; do not replace it with a post-hoc favorable threshold.

## Overwrite: seed44 overturns the two-seed story

| Seed | Scalar calibration effect | Channel calibration effect | Factorial interaction |
|---|---:|---:|---:|
|42|+6.74|-6.44|-13.18|
|43|-13.71|-15.30|-1.59|
|44|-19.34|+17.88|+37.23|

The earlier observation that calibration slowed channel learning in both first
seeds does not generalize to the third. Seed44's default channel finishes at
97.46%overall/94.92%unrelated retention; calibrated channel finishes99.80%/
99.61%, with much faster retention learning. Scalar seed44 moves in the opposite
direction: default99.22%overall/98.44%retention versus calibrated98.14%/96.48%.

The mean interaction is numerically+7.49points, but it is dominated by the large
positive third-seed value while two seeds are negative. Reporting just that mean
would conceal the reversal. Current evidence supports strong model/data-seed
sensitivity, not a consistent beneficial or harmful overwrite calibration effect.
Seeds change both model initialization and generated datasets, so we cannot
attribute cross-seed variation solely to initialization.

## What this permits us to say

The fixed mean-decay treatment is numerically successful, improves early channel
learning consistently in this MQAR setting, and has heterogeneous effects on
overwrite retention. Matching initial average decay does not equalize all
learning behavior or guarantee improvement. A task-dependent mechanism is a
research hypothesis, not yet demonstrated by these measurements.

Limits: three seeds, one small same-backbone capacity (channel+4.17%parameters),
one easy task cell per family, repeated development evaluation, and no new
confirmation for this24model matrix. Queries are not independent training
replicates. Do not claim statistical significance or broad superiority from
three same-direction MQAR effects. Do not call mean alpha a measured full-state
memory lifetime, or infer specific optimization mechanisms without diagnostics.

## Next action

Do not automatically repeat all24models for64epochs. The main MQAR difference
is early and longer training mostly adds saturated checkpoints; it cannot fix
the interpretation of seed-sensitive overwrite effects.

First evaluate the existing final16epoch checkpoints on a frozen fresh
confirmation set, matching the three-seed/four-condition matrix. The current
confirm_recipe_study.py is deliberately restricted to stage=study/64epochs, so
it needs an explicit16epoch exploration-compatible protocol before this step;
do not bypass its checks or call the old64epoch baseline confirmation sufficient.
Confirmation validates endpoints, not retrospectively the development AULC.

Then inspect initial decay variation and input sensitivity from the existing
models/initializations, using training-side diagnostic data. This can investigate
why matching means leaves differences, without another large training sweep.
If a stronger generalization claim is needed, predeclare one harder MQAR cell
after freezing the current analysis rather than tuning on confirmation outcomes.

The current defensible narrative is: "Initial mean-decay calibration reduces an
early-learning granularity gap on MQAR, while overwrite retention remains strongly
seed-sensitive." This is narrower and better supported than "calibration helps
scalar gates" or "calibration harms channel-wise gates."

Reproduce tables/checkpoint checks/figures with `python` running `analyse.py`
from the repository root. CSVs preserve all24individual runs and six paired
factorial contrasts, not only the condition means.
