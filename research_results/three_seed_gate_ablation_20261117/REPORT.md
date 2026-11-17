# Post-hoc shared-mean decay intervention

Status: complete. All12trained channel-wise models were evaluated on CPU, using
256fixed training examples outside the calibration-fitting sample. This is NOT
fresh confirmation and no retraining occurred. The output of every layer's decay
function was temporarily replaced by its own per-token channel average, expanded
back across channels. Local instantaneous mean is preserved for the current
inputs, but upstream changes also change later activations. Every other learned
parameter is fixed; the original function is restored afterward. Parameter
fingerprints remain unchanged. Runtime:1.29minutes.

| Task | Seed | Initialization | Original accuracy | Shared-mean accuracy |
|---|---:|---|---:|---:|
| MQAR |42|Default|100.00%|97.95%|
| MQAR |42|Calibrated|100.00%|82.52%|
| MQAR |43|Default|100.00%|94.43%|
| MQAR |43|Calibrated|100.00%|30.57%|
| MQAR |44|Default|100.00%|54.10%|
| MQAR |44|Calibrated|100.00%|95.80%|
| Overwrite |42|Default|100.00%|85.94%|
| Overwrite |42|Calibrated|98.63%|78.52%|
| Overwrite |43|Default|100.00%|100.00%|
| Overwrite |43|Calibrated|99.80%|99.41%|
| Overwrite |44|Default|98.83%|65.04%|
| Overwrite |44|Calibrated|100.00%|86.91%|

All six MQAR solutions depend on channel-specific gate structure to varying
degrees. Overwrite seed43is much less affected than seeds42/44. The perturbed
overwrite errors can involve updated values as well as unrelated retention;
for example calibrated seed44latest accuracy drops100->74.22%while unrelated
retention stays99.61%. Do not interpret the intervention as only a retention test.

This is a functional reliance test, NOT a fair scalar-versus-channel training
comparison. The trained channel weights co-adapted to their gates; a shared-gate
model trained from scratch can adapt differently and already solves these tasks.
The intervention is out of the trained model's normal operating conditions and
does not establish that more gate parameters are intrinsically necessary or that
the observed dependence caused earlier learning-speed differences. Report every
seed, not just the large100->30.57%example.
