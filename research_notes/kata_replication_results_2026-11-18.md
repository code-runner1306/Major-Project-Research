# KATA repeated-key overwrite: replication and rewrite-shortcut test, results

Date: 18 November 2026. Code: `run_kata_replication.py`, `eval_kata_replication.py`, `kata_model.py`,
`kata_overwrite_data.py` (commit 8afb0fe). Guide and deviations: [KATA_REPLICATION.md](../KATA_REPLICATION.md).

Results:
- **Training:** `output/kata_replication/20261118T101702039_cbe1357b/`.
- **Evaluation:** `research_results/kata_replication_eval_20261118_103946/`, commit e08a36c.

## Validity

- **Training.** Three models, one per learning rate in {1e-3, 3e-3, 1e-2}, seed 42, each 32 epochs, on
  one A6000. They ran concurrently in about 21 minutes. Early stopping did not trigger.
- **Model size.** 1,167,876 parameters, against 1,167,878 in the paper.
- **Evaluation data.** Fresh (seed + 1800000), CUDA. 0 overlapping examples with training, 0 oracle
  failures.
- **One assumption remains.** The overwrite sequence layout extends the paper's MQAR format, because
  the authors' data code is not public.

## R1: the replication is close to the paper

| | K=64 | K=128 | K=256 | K=384 | Average |
|---|---:|---:|---:|---:|---:|
| Paper, GDN dh=64 | 0.998 | 0.853 | 0.384 | 0.217 | 0.613 |
| Ours, selected lr 3e-3 | **0.995** | 0.765 | 0.284 | 0.119 | 0.541 |
| Ours, lr 1e-3 / 1e-2 | 0.965 / 0.988 | 0.589 / 0.717 | 0.197 / 0.266 | 0.083 / 0.117 | |

- **K=64 matches the paper** (0.995 vs 0.998), passing the adequacy criterion.
- **The extrapolation columns fall in the same order but sit about 0.09-0.10 lower.** Plausible
  causes: our layout choice, our training-set size, a single seed, and our portable GDN rather than
  FLA's kernel.
- **Beyond capacity, errors are confusions with other keys or random values, not stale values.** At
  K=384, 23 of 86,576 errors (0.03%) return the key's earlier write.

## R2-R5: no rewrite shortcut; the frozen predictions are falsified

| Cell (K = 4-32) | lr 1e-3 | lr 3e-3 | lr 1e-2 |
|---|---:|---:|---:|
| Two write passes (training) | 100% | 100% | 100% |
| **Three passes** | 100% | 100% | 100% |
| **Four passes** | 100% | 100% | 100% |
| Partial third pass, rewritten keys / keys kept at pass 2 | 100% / 99.9% | 100% / 100% | 100% / 100% |
| **One pass** (plain MQAR, never trained) | 100% | 100% | 100% |

- **R2 shortcut: 0 of 3 models.**
- **R3:** none of the 14 errors across all three-pass cells is a second write.
- **R4:** latest-write recall holds both for rewritten keys and for keys left at their pass-2 value.

In this setup, Gated DeltaNet learns genuine latest-write recall. It generalises to write counts it
never saw (1, 3 and 4), even though training made "second write" and "latest write" identical.

## What this means

**1. Being written exactly twice is not enough to cause the rewrite shortcut.** The A3 experiment
showed that mixed rewrite counts remove the rewrite failure in our own task. Here the same ambiguity
is present in every training example, and no failure appears. That fits the earlier observation that,
in our task, the rewrite failure was confined to block-first layouts, while other layouts already
generalised to 2-4 rewrites. The ambiguity matters only inside the position-based route that our data
let the model build.

**2. The KATA layout gives position no predictive value.** Every key is queried, so there are K scored
answers per example; the second write's position and the query distance vary freely. There is no
sub-population where a fixed position or distance identifies the answer, so there is nothing for a
positional route to exploit. Our v4 data has such a sub-population: in block-first examples, the
answer always sits in slot 1 at distance 8.

**3. Two external checks are now negative,** and both match the conditional claim:

| Setup | Fixed positional cue in part of the data? | Shortcut found |
|---|---|---|
| Our v4 overwrite (A0, legacy and retrained) | Yes (block-first: slot 1, distance 8) | 16 of 18 legacy, 6 of 6 retrained |
| Zoology MQAR (upstream generator) | No | 1 of 14, weak, not anchored to old positions |
| KATA-style overwrite (paper's model and training) | No | 0 of 3 |

**4. The model differs as well as the data.** The KATA model is a 2-layer BaseConv + GDN hybrid with
2 heads, no output gate, a different optimiser setting and 32 epochs, against our 4-layer, 1-head,
gated GDN. This test cannot separate "data structure" from "architecture or recipe". The cleanest way
to separate them would be to train the KATA model on our v4 data (A0 layout), or our model on the KATA
layout. Each needs 3-6 models.

## Consequences for the paper

- **Do not claim** that GDN overwrite training generally produces this shortcut, or that the
  twice-written ambiguity of published benchmarks causes it. The replication says otherwise.
- **The defensible claim:** a near-perfectly accurate GDN can learn a position-based route for a
  sub-population of inputs when the training generator contains a fixed structural regularity that
  predicts the answer there. Validation on the same distribution cannot see it. Generators without
  such a regularity (Zoology MQAR, the KATA-style layout) give robust key-based recall. Our layout-fix
  arms show which cue causes which failure.
- **Benchmark-design lesson.** Our v4 generator was built specifically to remove shortcuts (per-cell
  shortcut audits, randomised queries) and still contained one, at the level of a layout
  sub-population. Layout-stratified tests are a cheap check for any generator.
- **Honest limitation.** The effect is shown on one generator family. Its generality across
  architectures is untested until the cross-training test in point 4 is run.
- **Paper score.** These controls make the paper more credible, but the scope is narrower.
  - **Mid-tier venue:** about 6.5/10, a well-controlled case study.
  - **What would raise it:** the cross-training test, and a demonstration of the shortcut in a
    second, independent, widely used generator with the same kind of regularity.
