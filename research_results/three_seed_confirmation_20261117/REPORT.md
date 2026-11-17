# Fresh CPU confirmation: all 24 final 16-epoch checkpoints

Status: complete, 17 November 2026. All checkpoints existed locally. The evaluator
selected the complete exploratory matrix explicitly: two tasks, seeds42/43/44,
default/calibrated, scalar/channel-wise,16epochs/3136steps. No best-checkpoint
selection, training, offset fitting, or checkpoint modification occurred.

## Frozen protocol

512fresh examples per task and seed, generator seed400000+training seed. MQAR
uses its confirmation label; V4 overwrite uses the supported held_out domain.
All four conditions share the exact confirmation examples within each task/seed.
Content overlap with regenerated full training, development and previous
offset300000confirmation data is zero in all six task/seed audits. Source identity,
saved settings, final step/epoch counters and paired initialization hashes pass.

All inference ran on CPU in FP32. Primary is answer-free; teacher-forced scores
are secondary. Answer-free erases earlier gold answer inputs and does not feed
the model's own predictions back in. These are endpoint confirmations; they do
not independently confirm intermediate development learning curves.

## Mean final primary accuracy across three seeds

| Condition | MQAR | Overwrite | Overwrite unrelated retention |
|---|---:|---:|---:|
| Scalar default |99.9349%|99.3164%|98.6328%|
| Channel default |99.9674%|98.7956%|97.6563%|
| Scalar calibrated |99.9023%|97.9818%|95.9635%|
| Channel calibrated |99.9837%|97.3633%|94.8568%|

MQAR remains near ceiling: all individual models score99.7559..100%. Differences
between condition means amount to only a handful of answers and should not be
presented as reliable superiority at three seeds.

## Overwrite: individual results remain heterogeneous

| Seed | Scalar default | Channel default | Scalar calibrated | Channel calibrated |
|---|---:|---:|---:|---:|
|42|98.24%|99.61%|99.51%|92.97%|
|43|100.00%|100.00%|95.80%|99.51%|
|44|99.71%|96.78%|98.63%|99.61%|

The principal endpoint pattern from development survives fresh examples: seed42
calibrated channel is the weakest condition, seed43 calibrated scalar is weaker,
and seed44 calibrated channel beats default channel. This is not a universal
calibration improvement or a stable scalar/channel ranking.

Updated-value accuracy is99.9512%aggregated over the12overwrite models. There
are3wrong updated-value answers among6144such queries:2for calibrated channel
seed42 and1for default channel seed44. Only1is a stale-value answer. Most residual
errors concern unrelated-association retention.

Input conditioning matters for some models. Seed42 calibrated channel scores
92.97%answer-free but97.46%teacher-forced overall; unrelated retention changes
86.33->94.92%. Seed44 default channel scores96.78->99.41%overall. The primary
answer-free result remains the declared endpoint, but this suggests conditioning
sensitivity as well as memory behavior. It is not an evaluation of actual
autoregressive prediction feedback. Do not conflate the two protocols.

## Execution and reproducibility

Recorded evaluation wall time:9.12minutes, including overlap regeneration and a
resumed execution. An initial tuple-versus-JSON-list metadata equality check
incorrectly halted overwrite before its scores. The comparison was corrected to
canonical JSON equality while retaining actual content-hash equality. The same
frozen protocol resumed, reusing the completed MQAR scores; no new seed or dataset
was selected. Existing model-result checkpoint hashes were rechecked on resume.

`confirmation_protocol.json`, the six data-audit JSON files, per-model checkpoint
hashes, `confirmation_results.json`, and `models.csv` preserve the evidence.
The cached training prefixes are for the separate training-side diagnostics and
contain synthetic data only. This confirmation set is now observed and must not
be used to select a revised target recipe and then called untouched confirmation.
