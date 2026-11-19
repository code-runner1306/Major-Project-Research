# Calibrated overwrite exploration: seed42, 16 epochs

Evidence pulled from maink ac30150, session20261116T111608181_a2524782.
This is stage=explore, not the final64epoch study or untouched confirmation.
All four models completed3136accepted steps on NVIDIA RTXA6000-12Q, with the
unchanged64epoch cosine scheduler. Both pair_status files are complete.

| Condition | Final answer-free accuracy | Final unrelated retention | Retention AULC epochs1..16 |
|---|---:|---:|---:|
| Scalar default |98.7305%|97.4609%|0.790365|
| Channel default |99.9023%|99.8047%|0.730469|
| Scalar calibrated |99.8047%|99.6094%|0.857747|
| Channel calibrated |94.0430%|88.0859%|0.666081|

All four finish with100%updated-value accuracy and zero stale-value errors. The
differences concern unrelated association retention rather than simply copying
the latest value. Scores use512development examples /1024answers, not new
confirmation data. AULC is normalized trapezoidal area over epochs1..16, not an
arithmetic mean of checkpoints or a final accuracy metric.

## Checks

All audits pass, with zero exact train/development overlap. All four data
manifests match. Shared-backbone initial hashes match across conditions; within
each variant, default and calibrated base parameter hashes match. Calibration
reaches target means with maximum errors below1e-7. Both calibrated reference/
chunked five-step optimizer trajectory checks pass. Calibration costs0.93seconds
scalar and9.99seconds channel; no material runtime penalty is visible.

## Interpretation

Calibration changes scalar retention AULC by+0.067383 and channel retention AULC
by-0.064388. The channel-minus-scalar gap changes from-0.059896 under default
initialization to-0.191667 under calibrated initialization: exploratory interaction
-0.131771. This is a meaningful candidate for the study's question, but one
seed is not replicated evidence or a significance test. It does not establish
general superiority of scalar gating or demonstrate that calibration always helps.

Default channel finishes better despite its lower AULC because it learns later
and overtakes scalar. Calibrated channel is still improving at epoch16, so its
lower endpoint is not proof it cannot eventually solve the task. Scalar calibrated
has an early epoch2dip, followed by faster retention learning; do not hide that
dip by reporting only its endpoint. Primary learning-curve metrics are essential.

Earlier default T4 seed42 curves differ from this A6000 seed42 exploration.
Same integer seed does not establish cross-hardware bitwise reproduction. Do not
pool those curves with this four-condition comparison or claim the earlier
channel learning-speed advantage is already replicated.

## Completion and next action

Default pair took57m02s, calibrated56m36s. Total session119m21s includes
qualification/publication overhead. The root session status says failed_or_incomplete
because Windows temporary-directory cleanup raised WinError32 during publishing.
Training and its saved checkpoints are complete; the manual maink upload now
preserves the missing artifacts. This is an infrastructure issue, not a failed
calibrated optimization run. Fix publishing cleanup before relying on unattended
multi-session uploads; no retraining of these four models is needed for that issue.

Next run the exact same overwrite exploration with seed43, all four conditions,
16epochs and scheduler64. Estimated cost about2hours. Keep targets, learning rate
and metrics unchanged. Check whether the interaction direction and learning-curve
pattern repeat. If they do, test the corresponding MQAR calibrated comparison
before committing to the full three-seed matrix. Do not automatically enlarge the
budget or tune targets merely because calibrated channel finishes lower here.
