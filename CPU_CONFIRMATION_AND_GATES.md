# CPU confirmation and gate diagnostics

Both procedures use inference only. They do not resume training, fit new target
values, or modify the original checkpoints.

## Explicitly select the completed 16-epoch matrix

```powershell
python confirm_recipe_study.py --root output/calibrated_recipe_study/matched-decay-v1 --output research_results/three_seed_confirmation_20261117 --stage explore --device cpu
```

The default stage remains `study` (64 epochs). `--stage explore` explicitly
requires 16 completed epochs / 3,136 steps for every one of the 24 conditions.
The evaluator checks checkpoint existence, source identity, settings, dataset
pairing and initial-parameter hashes. It uses all final checkpoints, not selected
best checkpoints. CPU and GPU FP32 reduction order can differ slightly.

The frozen confirmation set contains 512 examples per task and seed with seed
offset 400000. Overwrite uses the generator's `held_out` domain. The evaluator
checks exact content overlap with the regenerated full training pools,
development sets and previous offset-300000 confirmation sets. Metadata equality
uses its JSON representation, so tuple/list serialization differences do not
cause false failures; content hashes must still match.

The output directory must be new. If interrupted, use the SAME command with
`--resume`. This verifies the frozen protocol and existing checkpoint hashes,
preserves completed scores and reuses completed overlap audits. It does not draw
a different confirmation set or repeat completed model evaluations. Once the
results are inspected, this set is not untouched data for tuning another recipe.

`confirmation_results.json` contains each model's answer-free primary and
teacher-forced secondary endpoint metrics. The contrasts marked as AULC still
come from development curves; fresh endpoint evaluation does not retroactively
confirm those learning trajectories.

## Diagnose gate behavior after confirmation completes

```powershell
python diagnose_recipe_gates.py --root output/calibrated_recipe_study/matched-decay-v1 --confirmation research_results/three_seed_confirmation_20261117 --output research_results/three_seed_gate_diagnostics_20261117
```

The exact first 512 rows of each full 100,000-example training pool are cached
during confirmation's overlap checks. This matters for MQAR: building a smaller
pool with the same seed does not reproduce a prefix of the original larger pool,
because its RNG call order draws all keys, then all values, then all gaps.

Diagnostics use training-side inputs only, not confirmation examples:

- First 256 original training examples, to verify reconstructed calibration.
- Next 256 training examples under original input conditioning.
- Those next 256 overwrite examples with previous gold answer inputs erased,
  matching the answer-free evaluation contract. MQAR has no earlier answer inputs
  to erase, so this duplicate probe is omitted there.

Initial models are reconstructed with the original seed and source and checked
against the saved base-parameter fingerprint. Only a verified replay is probed
as an initial model. On a mismatch, diagnostics use the actual trained checkpoint
and the original recorded calibration means, standard deviations and offsets;
they do not label a newly reconstructed initial model as exact. Offset changes
can still be measured against saved calibrated offsets. No calibration refit
occurs. Dropout is disabled; no gradients or optimizer updates run.

For each layer, report alpha/beta mean, variation, extrema, quantiles, saturation,
within-token channel dispersion, token variation within gates, gate-variation
covariance effective rank and active-gate correlations. Means/standard deviations/
extrema/fractions use all probe values; large-tensor quantiles use a fixed
16,384-value subsample. Correlations use a separate fixed 1,024-token sample.

The sensitivity statistic is the absolute analytic derivative of alpha with
respect to the raw projection or offset. It is checked against autograd, but is
not the full loss gradient. Position groups are filler, supervised query-key
positions, and other input tokens; `other` is not asserted to mean writes only.

Mean-derived half-life is an isolated-decay diagnostic, not measured memory
lifetime. Alpha statistics alone omit delta-rule writes and readout. Initial/final
comparisons cannot identify intermediate gate dynamics because only final
checkpoints were retained. Correlations with learning scores are descriptive,
not causal mechanisms. Reports and CSVs retain all per-seed observations.
