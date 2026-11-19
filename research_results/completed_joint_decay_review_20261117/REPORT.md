# Completed joint initialization and decay-boundary analysis

Reviewed 17 November 2026. Source commit **2b3d0af** from `origin/maink`. Training session `20261117T095630656_b951c742`; evaluation session `20261117T154211097_be1684fd`.

## Verdict

**The prespecified dynamic-decay necessity hypothesis is not supported in this tested regime.** Keeping distinct channel constants usually preserves MQAR performance, including the harder load/context slices. Dynamic decay also does not consistently outperform fixed-channel decay on repeated overwrite. None of the 18 per-seed primary reliance rules passes; consequently no initialization meets the required three-seed rule.

The most reproducible unresolved failure remains repeated updating: models often answer with the first replacement instead of the newest value. Joint initialization does not remove that failure. The next discriminating test is training with varied rewrite counts under matched exposure, not another larger initialization sweep.

## What was verified

- Twelve joint models completed 16 epochs / 3136 optimizer steps; all six pairs and the session report are complete, with no deferred runs.
- All twelve joint source identities, calibration reports, and reported per-model numerical checks pass. Calibration targets are achieved within 1e-6. This verifies the implemented adapted recipe, not an exact production-scale DAMI reproduction.
- The held-out evaluation contains the exact expected **36-checkpoint / 540-cell** matrix, 512 examples per cell, 256 training-prefix fitting examples, CUDA execution, and no CPU smoke flag.
- The evaluator's frozen source hashes match repository commit `e860526`. Its environment records **one worker** on **NVIDIA RTXA6000-12Q**; this is not a two-worker benchmark.
- All 540 raw-prediction files match recorded SHA-256 hashes. Recreated 45 unique datasets/pair cases match labels, kinds, example indexes, and data hashes. Exact-query/latest/retention/stale metrics were recomputed; all 108 paired metric rows match.
- Uploaded generator/overlap/numerical audit reports pass. These audits and regenerated data do not constitute a new full training-pool overlap computation or fresh model inference here. Joint weights were not uploaded, so no independent joint inference was rerun locally.

See `verification.json`, `condition_means.csv`, `obsolete_value_breakdown.csv`, and `training_comparison.csv`. The analysis script is `research_notes/analyze_completed_joint_decay_20261117.py`.

## 1. Joint training: successful learning, no universal improvement

Means across seeds 42/43/44. AULC is the trapezoidal development-accuracy average over epochs 1–16, not final test accuracy.

| Task / variant | Default accuracy AULC | Mean-alpha fit | Joint fit |
|---|---:|---:|---:|
| MQAR scalar | 93.71% | 93.63% | 93.32% |
| MQAR channel-wise | 91.16% | 93.73% | 93.50% |
| Overwrite scalar | 95.20% | 90.67% | 91.10% |
| Overwrite channel-wise | 91.01% | 91.14% | 91.64% |

Overwrite unrelated-retention AULC is default **91.00% scalar / 83.80% channel**, mean-alpha **82.23% / 82.51%**, and joint **83.68% / 83.57%**. Thus the joint recipe does not improve average retention learning over default in either variant. Seed-42 gains from the partial review did not generalize into a uniform three-seed benefit.

Joint final development MQAR accuracy averages 99.92% scalar / 99.93% channel; final overwrite averages 98.67% / 99.22%. Clean-task learnability is established for this recipe, but ordinary validation does not certify reliable repeated updating.

Do not call this a failure of published DAMI: our one-head/four-layer adaptation differs from its architecture and targets. The old mean-alpha and joint mean-log-decay fits also differ in more than sensitivity.

## 2. Held-out MQAR: channel diversity matters more than token dependence here

Accuracy for length 64 with 12 associations, means over three seeds:

| Initialization | Scalar original | Channel dynamic | Channel fixed per-channel | Channel averaged per token |
|---|---:|---:|---:|---:|
| Default | 75.15% | 87.33% | 87.78% | 39.83% |
| Mean-alpha | 89.04% | 94.80% | 94.57% | 42.18% |
| Joint | 94.09% | 92.21% | 91.83% | 70.31% |

Fixed per-channel gates remain close to dynamic gates across all six MQAR cells. At length 128 / 12 associations, dynamic versus fixed scores are default 87.74%/88.05%, mean-alpha 94.82%/94.58%, and joint 92.55%/92.27%.

Collapsing channels to a shared per-token average is much more damaging. This extends the earlier training-side probe observation to fresh held-out examples and greater load/context, while remaining a checkpoint intervention rather than a trained-static-architecture comparison.

Initialization changes average architecture rankings, but seed effects are large. Default scalar's 12-association mean is depressed by seed 44 at 35.30%, while seeds 42/43 score 94.73%/95.41%. Channel scores for those seeds are 87.96%/79.67%/94.35%. Do not interpret the default mean advantage as a consistent channel-wise benefit. In the joint condition, scalar wins two of three seeds, not all three. Parameter count also differs by approximately 4.17% between architectures.

## 3. Held-out overwrite: baseline success conceals update fragility

Newest-value accuracy after three rewrites, means over three seeds:

| Initialization | Scalar original | Channel dynamic | Channel fixed per-channel |
|---|---:|---:|---:|
| Default | 76.82% | 78.19% | 76.56% |
| Mean-alpha | 54.10% | 59.31% | 63.15% |
| Joint | 65.17% | 66.28% | 66.86% |

Original baseline newest-value accuracy is approximately 99.7–100%, so the multi-rewrite drop is substantial. Fixed decay does not reliably repair it, but neither does dynamic decay provide the predicted consistent advantage. For distance 24, default fixed-channel actually averages 90.89% latest accuracy versus 81.18% dynamic; this is another reason not to characterize dynamic gates as uniformly helpful.

There are important seed-specific exceptions. Mean-alpha channel seed 44 loses 25.98 points of baseline newest accuracy under the static intervention, yet improves from 67.19% to 76.76% on three rewrites. Default channel seed 44 has a 5.27-point dynamic advantage on three rewrites, but already a 3.32-point baseline advantage, failing the predefined baseline constraint. These are not replicated stress-specific boundaries.

For the original joint channel models on three rewrites, unrelated retention averages **96.48%** while newest-value accuracy is **66.28%**. This is selective update failure rather than wholesale forgetting. Calibration/gate conditions can trade newest-value success against unrelated retention, so preserve both metrics.

## 4. Obsolete answers and paired tests

Across the 18 original overwrite models (three initializations × two variants × three seeds), three-rewrite evaluation has 9216 newest-value queries and 3074 errors:

| Wrong answer identity | Count | Share of newest-value errors |
|---|---:|---:|
| Initial value | 117 | 3.81% |
| First replacement | **2794** | **90.89%** |
| Second replacement | 97 | 3.16% |
| Another value | 66 | 2.15% |

The first-replacement dominance holds across initialization groups: 680/691 default errors, 1085/1330 mean-alpha errors, and 1029/1053 joint errors. It is not merely one newly failed joint seed.

The paired latest-value change test leaves much of this failure intact. For joint channel models, mean newest-value accuracy after changing only the last value is 66.41%, while roughly one third of newest outputs fail to change. Pair joint-correctness is 66.28%. Changing the obsolete initial value changes only 0.33% of their newest outputs. Thus many failures are stable selection of an earlier replacement, not arbitrary sensitivity to every past value. This does not by itself identify where the wrong value is stored or prove a count-based internal algorithm.

Default seed 43 is a critical counterexample to a universal architecture limit: its scalar model scores 100% newest and unrelated accuracy for two/three/four rewrites and all three paired tests. Its channel model is 99.80% on two rewrites and 100% on three/four and the paired tests. Other seeds are much weaker despite the same architecture/data recipe. Training dynamics and coverage remain plausible explanations, not yet proven causes.

## Decision and next action

1. **Do not claim that this suite found a replicated necessity boundary for dynamic decay.** The predefined primary criterion failed; extending tests until a favorable condition appears would be a new exploratory study.
2. Retain the supported finding: trained channel heterogeneity can matter while input-dependent decay is largely dispensable for the tested MQAR conditions. Its novelty must be scoped against static-gating prior work.
3. Prioritize the planned **six-model mixed-rewrite overwrite training comparison**: default scalar/channel × three seeds, equal steps/examples/tokens/scored targets, with rewrite counts 1/2/3 and untouched count/layout tests. Keep the old generator's remaining semantics fixed; if the sampler changes more, add a matched single-rewrite control. Report differing write-event exposure.
4. Ask whether mixed training removes the first-replacement error mode and improves response to final-value changes without harming unrelated retention. That is a discriminating test of training coverage versus an inferred architecture limitation. Prespecify the decision and new confirmation examples before running it.
5. Do not spend more compute on a larger calibration sweep or claim this is a new architecture. If the repair is simple and generalizes, the contribution is a controlled training-distribution/robustness finding. If it does not, localize the write/read mechanism before proposing another fix. Strong novelty remains unestablished.

## Actual runtimes

- Joint training, all twelve models: **20655.58 seconds = 5h44m16s**, including coordinator overhead, calibration, development validation, and checkpoint saving.
- Complete 540-cell GPU evaluation: **353.98 seconds = 5m54s**, recorded with **one worker**. This includes the evaluator's recorded setup/audits and per-cell work; it is not just summed GPU-forward time.
- The prior 20–60-minute evaluation forecasts were substantially too conservative. This run supplies a measured baseline. There is still no measured two-worker speedup from this upload.

The reserved confirmation offset 900000 has not been evaluated by this discovery script. Current conclusions are discovery evidence with three training seeds, not final untouched confirmation or large-model generalization.
