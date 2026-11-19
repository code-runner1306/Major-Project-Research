# Partial joint-initialization review

Source: `origin/maink` commit **226235b**, committed **17 November 2026, 14:24:10 IST**. Uploaded session: `20261117T095630656_b951c742`. This is a report-based snapshot review, not live observation or checkpoint evaluation.

## Progress and verification

- Eight of twelve models are complete: both scalar/channel variants for MQAR and overwrite, seeds 42 and 43.
- MQAR seed 44 has saved curves through epoch 11 / step 2156 for scalar and epoch 10 / step 1960 for channel-wise.
- No overwrite seed-44 reports have been uploaded yet. Absence from this snapshot does not establish current live status.
- All ten available joint calibration reports pass; maximum sensitivity error is approximately 1.1e-8, and maximum mean-log-decay error approximately 5.2e-9. Available per-model backend checks pass.
- The session's training source hashes match the local joint runner. For every available joint condition, recipe fields, training/development manifests, and complete base initialization manifests match the corresponding legacy default condition. Eight final summaries contain 16 epochs / 3136 optimizer steps.
- No joint `.pt` files were uploaded. The reports are sufficient for this analysis, but gate interventions and new held-out tests need the actual VM checkpoints or downloaded copies.

Partial log tails report only steps 1200/1250 while the development curves contain later completed epochs. The export is not atomic. Runtime extrapolation below uses the later monotonic epoch curves; it does not treat old console tails as live progress.

## Completed joint-model ordinary development accuracy (%)

| Task | Seed | Scalar | Channel-wise |
|---|---:|---:|---:|
| MQAR | 42 | 99.95 | 100.00 |
| MQAR | 43 | 99.80 | 99.95 |
| Overwrite, all scored queries | 42 | 100.00 | 98.73 |
| Overwrite, all scored queries | 43 | 96.09 | 99.02 |

These are repeatedly observed training-development scores, not new confirmation or multi-rewrite stress results. Near-perfect MQAR does not answer whether dynamic decay is necessary.

## Updating and retaining are different outcomes

| Joint overwrite model | Newest-value accuracy | Unrelated retention |
|---|---:|---:|
| Seed 42 scalar | 100.00% | 100.00% |
| Seed 42 channel-wise | 100.00% | 97.46% |
| Seed 43 scalar | 99.80% | 92.38% |
| Seed 43 channel-wise | 98.05% | 100.00% |

For seed 43, channel-wise improves unrelated retention by 7.62 percentage points over scalar but reduces newest-value accuracy by 1.76 points. For seed 42, scalar is at least as good on both. This is a reason to keep the two metrics separate, not evidence that channel-wise universally improves memory.

The asymmetry is even clearer early in training. Joint seed-43 scalar reaches 100% newest-value recall at epoch one while retaining only 29.69% of unrelated facts; seed-42 channel-wise likewise has 100% newest recall but 33.40% retention. Thus “can return the update” is not equivalent to “can update without collateral loss.” These curves do not identify the causal gate or rule out a training-distribution shortcut.

## Comparison with prior initialization recipes

Retention AULC is the trapezoidal mean of development retention over epochs 1–16. Higher values indicate more reliable retention during this fixed training window; it is not final accuracy or a significance test.

| Overwrite model | Default retention AULC | Mean-alpha fit | Joint fit | Joint minus default |
|---|---:|---:|---:|---:|
| Seed 42 scalar | 79.04% | 85.77% | 82.73% | +3.70 pp |
| Seed 42 channel-wise | 73.05% | 66.61% | 78.28% | +5.23 pp |
| Seed 43 scalar | 97.06% | 83.35% | 75.72% | -21.34 pp |
| Seed 43 channel-wise | 99.92% | 84.62% | 83.97% | -15.94 pp |

The joint fit helps relative to default for seed 42 but hurts both variants substantially for seed 43. It does not currently look like a universal repair. Do not pool these opposite responses into a single success claim or decide the three-seed result before seed 44 finishes.

MQAR learning AULC also has mixed effects. Joint scalar is 98.70% versus default 96.43% at seed 42, but 92.50% versus 95.46% at seed 43. Joint channel-wise is 93.17%/94.94% versus default 92.54%/93.11%; the old mean-alpha channel recipe is higher at 94.76%/95.56%. The almost saturated final MQAR scores conceal learning-trajectory differences.

## Meaning for the new research direction

The joint experiment supplies a meaningful attributed initialization comparison: a stronger calibration does not automatically remove seed/task dependence. This supports testing the proposed boundary, but does not itself establish novelty or causation.

After completion, evaluate all three initializations on the same frozen held-out boundary protocol. The key contrast is the dynamic-versus-static per-channel difference under stress after subtracting its baseline difference. Report latest-value recall and unrelated retention independently, plus correctness-aware paired perturbation tests. Gate-averaged models are interventions on trained channel networks, not trained scalar architecture controls.

Preserve final 16-epoch checkpoints; do not select an earlier epoch because it makes the final comparison look better. If repeated-update weaknesses persist, use the planned varied-rewrite training control before describing them as intrinsic architectural limitations. The joint recipe differs from the earlier fit in both sensitivity and decay-matching statistic, so this comparison cannot isolate sensitivity alone or refute published DAMI generally.

## Runtime estimate

Measured scalar/channel pair durations, including setup, fitting, validation, and checkpoints:

| Family | Seed 42 | Seed 43 |
|---|---:|---:|
| MQAR | 56.97 min | 57.49 min |
| Overwrite | 56.92 min | 57.01 min |

From the uploaded epoch counts:

- Remaining seed-44 MQAR pair: approximately **21.5 minutes**, governed by the slower channel model.
- Full seed-44 overwrite pair afterward: approximately **57.0 minutes**.
- Total remaining from those uploaded counts: **78.4 minutes**, approximately **80 minutes**. Allow around **90 minutes** for variation and overhead.
- Full six-pair runtime extrapolation: **5h43m**, plus coordinator startup/gaps. A practical overall expectation is roughly **5h45m–6h**, assuming uninterrupted execution on the same RTXA6000-12Q with two workers.

If the job continued at the same speed after the approximately 14:24 upload, a rough finish window is **15:40–16:00 IST** on 17 November. This is conditional, not a live ETA; subtract only actual continued runtime from the snapshot estimate. Concurrent evaluation, pauses, or a stale staged file can invalidate it.

The estimate includes the run's existing development validation and checkpoint saving. It excludes the separate held-out decay-boundary evaluation; reserve the previously provisional 30–60 minutes for the combined 36-checkpoint GPU evaluation, whose GPU runtime has not yet been measured.

Artifacts: `comparison.csv`, `learning_curves.json`, `runtime_estimate.json`. Reproduce report statistics with `research_notes/analyze_partial_joint_20261117.py`. No new training, checkpoint changes, or extra model evaluation was performed.
