# Overwrite exploration: completed seeds42 and43

Pulled maink4baf6a2. Seed43 calibrated session20261116T153034494_5014216e is
complete, with no deferred runs. Its two models completed16epochs/3136steps in
56m57s together. Defaults came from session20261116T140317869_6f95b296; their
training completed despite the later publishing cleanup failure. Seed42's four
conditions are in session20261116T111608181_a2524782. These are exploratory
development results, not final64epoch or untouched-confirmation results.

## Results

Retention score is normalized trapezoidal AULC over epochs1..16, multiplied by100.

| Seed | Condition | Final overall accuracy | Final unrelated retention | Retention score |
|---|---|---:|---:|---:|
|42|Scalar default|98.73%|97.46%|79.04|
|42|Channel default|99.90%|99.80%|73.05|
|42|Scalar calibrated|99.80%|99.61%|85.77|
|42|Channel calibrated|94.04%|88.09%|66.61|
|43|Scalar default|100.00%|100.00%|97.06|
|43|Channel default|100.00%|100.00%|99.92|
|43|Scalar calibrated|95.41%|90.82%|83.35|
|43|Channel calibrated|99.41%|98.83%|84.62|

All eight finish at100%latest-value accuracy, with zero stale-value errors.
Each evaluation has512development examples/1024answers. Both seeds pass their
data audits with zero exact train/development overlap. Within each seed, all four
data manifests and shared-backbone initialization hashes match; default/calibrated
base hashes also match within each gate variant. Source identities match across
conditions. All calibrated fits have maximum absolute mean-decay error<1e-7,
and both seeds' calibrated backend optimizer-trajectory checks pass.

## What repeats and what does not

Calibration changes scalar retention AULC by+6.74percentage points in seed42,
but-13.71points in seed43. It changes channel AULC by-6.44points in seed42 and
-15.30points in seed43. Thus the channel slowdown repeats; the scalar improvement
does not. Scalar beats channel under calibration in seed42, but the ranking
reverses slightly in seed43. No universal gate ranking follows.

The exploratory factorial interaction (channel-minus-scalar calibrated gap minus
default gap) is-13.18points for seed42 and-1.59points for seed43. The direction
matches but magnitude is highly variable; two paired seeds do not establish a
stable effect or significance. Across-seed mean retention scores are88.05scalar
default,86.48channel default,84.56scalar calibrated,75.61channel calibrated.
These descriptive means must not hide the individual seed outcomes.

Default seed43 channel is already nearly perfect at epoch1; default scalar reaches
the95%window starting epoch2. Calibrated channel reaches that window starting
epoch7, while calibrated scalar never has three consecutive validations at95%
in all subgroups within16epochs. The latter's unrelated retention is around89..93%
over much of epochs4..16. This is not evidence that a longer run cannot improve
it, nor justification for comparing a best checkpoint against other final models.

## Decision

The calibration treatment is executable and numerically correct on this hardware,
but this chosen mean-decay target allocation is not a consistently beneficial
training recipe. That is compatible with the research question: matching initial
means need not help all granularities or seeds. Do not claim successful numerical
calibration guarantees better learned memory, or change targets after these
development results and call the amended treatment prospectively confirmed.

The default recipe is demonstrably learnable; current evidence does not justify
calling scalar calibration an improvement or launching a large64epoch matrix to
prove that narrative. Continue the controlled16epoch scope: a third overwrite
seed checks stability, and the corresponding MQAR comparison is needed before
any claim that the effect differs between tasks. Freeze metrics and target values.
Use final confirmation only after choosing the final prospective scope.

All seed changes currently change both model initialization and generated data;
the large cross-seed variation cannot be attributed to initialization alone from
these results. The comparison also remains same-backbone, not equal-parameter.
No new GPU training or confirmation evaluation was performed for this review.
