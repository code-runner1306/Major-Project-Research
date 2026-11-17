# Memory stress tests: completed CPU study

All 24 existing 16-epoch checkpoints were evaluated on four conditions per task, 256 examples per condition. Runtime: 12.46 minutes on CPU, concurrently with the intervention study. No further training occurred. Data seeds are 500000 plus training seed; earlier answer tokens are erased. Generator audits passed, with zero exact training-example overlap.

These are zero-shot distribution shifts from training length 64, four associations, and one overwrite. They measure robustness of these checkpoints, not whether the architecture could learn the harder distributions. Conditions have now been observed and cannot serve as untouched confirmation data for future tuning.

## MQAR accuracy, mean across seeds (%)

| Length / associations | Default scalar | Default channel | Calibrated scalar | Calibrated channel |
|---|---:|---:|---:|---:|
| 64 / 4 | 99.90 | 99.97 | 99.90 | 100.00 |
| 64 / 8 | 83.33 | 97.36 | 97.80 | 98.99 |
| 128 / 4 | 99.93 | 99.97 | 99.93 | 100.00 |
| 128 / 8 | 83.45 | 97.57 | 97.98 | 98.89 |

Increasing association load exposes default scalar seed 44: 1091/2048 correct (53.27%) with eight associations. By memory ordinal its counts are 256, 256, 240, 1, 0, 4, 79, 255 out of 256 each. This is structured association failure, not uniform random degradation. Other seeds perform much better; the group difference is driven largely by this run. Calibrated scalar seed 44 recovers 2002/2048 (97.75%).

Length 128 alone is well tolerated. Actual mean query distances are about 29–31 tokens with four associations, and 36–37 with eight; maximum distances reach 123–125. This is not evidence that every query is remembered across 128 tokens.

## Overwrite accuracy, mean across seeds (%)

| Condition | Default scalar | Default channel | Calibrated scalar | Calibrated channel |
|---|---:|---:|---:|---:|
| Baseline: four keys, distance 8, one rewrite | 99.54 | 99.09 | 97.46 | 97.98 |
| Eight keys | 77.80 | 89.71 | 61.26 | 70.31 |
| Query distance 24 | 78.19 | 79.75 | 68.36 | 76.30 |
| Three rewrites | 85.94 | 87.83 | 70.51 | 75.59 |

The combined accuracy averages newest-value recall and unrelated-key retention. With three rewrites, newest-value recall alone is **74.61%, 77.60%, 52.47%, 56.64%**, respectively; unrelated retention is 97.27%, 98.05%, 88.54%, 94.53%. Predictions matching any obsolete target value occur on 25.00%, 22.40%, 46.09%, 41.67% of newest-value queries.

A post-hoc breakdown reran precisely these conditions and asserted equality with the original scores. Across twelve overwrite models, 926/1065 newest-value errors return the first replacement, 59 the initial value, 53 the second replacement, and 27 another value. Thus obsolete answers dominate, particularly the first replacement. Default seed 43 nevertheless scores 100% newest-value recall in both variants: the failure is not universal. The results suggest some runs learned the single-rewrite training pattern without a robust general rule for repeated updates; they do not establish a causal explanation by themselves.

## Implications

The clean baseline is learnable, but near-perfect baseline scores hide substantial update/load fragility. Current mean-decay calibration does not consistently improve overwrite robustness. A useful next comparison should separate initialization, channel heterogeneity, and dynamic gating, with fixed training exposure and prespecified harder validation conditions. Three seeds support descriptive patterns, not strong significance claims; no hyperparameters were tuned in these tests.

Artifacts: `models.csv`, `condition_means.csv`, `stress_accuracy.png`, `mqar_actual_query_distances.csv`, `protocol.json`, `results.json`, and `error_breakdown.json`. Scripts: `stress_recipe_checkpoints.py`, `breakdown_stress_errors.py`, `summarize_followup_cpu.py`.
