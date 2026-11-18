# Layout-fix training study: results and conclusions

Date: 18 November 2026. Protocol `layout-fix-v1`. Plan and frozen predictions:
[training_fix_plan_and_positioning_2026-11-17.md](training_fix_plan_and_positioning_2026-11-17.md).

Sources:
- **Training:** `output/layout_fix_study/layout-fix-v1/` (commit 9860540). These files come from the
  branches `research/layout-fix/layout-fix-v1/A2_no_block_first+A0_control/20261117T151622Z_9e372e17`
  and `.../A1_varied_distance+A3_mixed_rewrites/20261117T151800Z_17deccea`. The weights-only
  checkpoints remain on those branches.
- **Evaluation:** `research_results/layout_fix_confirmation_20261118/`, produced by
  `confirm_block_layout.py` on CPU at seed offset 1400000 with extra distances 20 and 24.

## Validity

- **Training completed.** 24 models: arms A0-A3 x scalar/channel-wise x seeds 42/43/44, each 16 epochs
  and 3136 steps on Kaggle T4 x2 (one worker per GPU). Both sessions are `complete` with nothing
  deferred, in 6.3 h and 6.2 h.
- **The arms are cleanly paired.** Initial weights are identical across all four arms for every seed
  and variant, including across the two notebooks. The A0 training data reproduce the legacy training
  hash.
- **The training data match the plan.**
  - A2 contains 0 block-first examples; about 49.5k were rejected per seed.
  - A1 cycles distances 8-16.
  - A3 has rewrite counts 1/2/3 in thirds.
  - A0, A1 and A3 keep the usual third of block-first examples.
- **The evaluation data are clean.** Fresh data, overlap audit on, 0 overlapping examples in every
  cell, 256 examples per layout.
- **Not evaluated on a GPU.** The confirmation ran on CPU instead of the planned GPU. It uses the
  same code and data, so this only affects run time (1012 s).

## Pre-registered predictions

| Prediction | Result | Verdict |
|---|---|---|
| **P1:** each arm leaves at most 1 of 6 models with the shortcut (A0 expected 4 of 6) | A0 **6/6**, A1 **3/6**, A2 **4/6**, A3 **0/6** | Met for A3 only |
| **P2a:** general-layout accuracy at 2-4 rewrites within 3 points of A0 (97.5%) | A1 87.8 (**-9.7**), A2 98.5 (+1.1), A3 99.7 (**+2.2**) | A2 and A3 pass, A1 fails |
| **P2b:** every model at least 99% on the original development set | A0 5/6, A1 5/6, A2 **1/6**, A3 3/6 | Fails for all arms, including the control |
| **P3:** decision table | Only A3 removes the shortcut, giving the row "models need a signal separating second from latest" | That row applies |

P2b also fails for one control model, so the 99% threshold was too strict. What it reveals is a small
unrelated-recall cost in A3: 96.4% vs 98.9% for A0 on the original development set.

## Conclusions

### 1. The shortcut replicates on retrained models and different hardware

The new control models (A0, trained on T4) show the shortcut in **6 of 6** models:
- block-first at 3 rewrites: 10.8% correct, against 97.3% on the other layouts;
- 95.8% of those errors are the first replacement.

The legacy A6000 models showed it in 4 of 6. The default seed-43 models were clean in the legacy run;
retrained, they show it partially (31-34% on block-first). So that earlier immunity was specific to
one training run, not to the seed, which is further evidence of underspecification.

### 2. There are two separable failures, and each needs its own data fix

| Failure | Measured by | A0 | A1 varied distance | A2 no block-first | A3 mixed rewrites |
|---|---|---:|---:|---:|---:|
| **Rewrite failure:** returns the second write instead of the latest | block-first accuracy, 2-4 rewrites | 10.5% | 62.3% | 53.9% | **99.9%** |
| **Position failure:** breaks when filler separates memory from query | block-first deficit vs general, 1 rewrite, distance 12 / 16 / 20 / 24 | 66 / 29 / 29 / 27 pts | **0.0 / 0.2 / 0.1 / 0.4 pts** | 69 / 36 / 33 / 30 pts | 53 / 32 / 30 / 27 pts |

- **Mixed rewrite counts (A3) remove the rewrite failure completely.** All 6 models score 100% on
  block-first at 3 rewrites, and 100% of their answers follow the latest-write rule. They still fail
  under the distance shift: 39% on block-first at distance 12 vs 92% on general layouts.
- **Varied query distance (A1) removes the position failure completely,** including at distances 20
  and 24, which were never in training (99.6-99.9%). But 3 of 6 models keep the rewrite failure, and
  accuracy on general layouts at 2-4 rewrites drops 9.7 points. That drop is driven by seeds 42 and
  44, with 67-82% at 3 rewrites.

So the block-first route has two independent properties: *which write it returns* and *where it
looks*. Each is fixed only by training diversity along its own axis. A stress test along one axis
cannot certify the other. That is the underspecification point, shown causally rather than only by
observation.

### 3. Removing the layout is not a fix (A2)

Models never trained on block-first examples cannot answer them at all.
- **At one rewrite,** block-first is answered correctly only 56% of the time, against 98.9% on
  general layouts, and only 1 of 6 models reaches 95%.
- **The errors are mixed:** initial value 815, first replacement 652, other 607.
- **The key-based lookup they learn elsewhere does not carry over.** The block-first layout (memory
  directly before the queries, with the key's writes opening the memory) is handled as a genuinely
  different case by these models.

This strengthens the "two strategies" interpretation. Block-first examples need their own learned
route, and that route forms only from block-first training data. Which shortcut it encodes depends
on what that data leaves ambiguous.

### 4. Gate type remains irrelevant

Scalar and channel-wise models behave alike within every arm. For example, block-first at 3 rewrites
is 0-34% in both A0 variants and 100% in both A3 variants.

## What this means for the paper

The paper now has a complete causal arc:
1. A pre-registered, IID-invisible shortcut in 16 of 18 legacy models, and in 6 of 6 retrained
   controls.
2. A mechanism-level characterisation: position-anchored retrieval for one layout, coexisting with
   key-based lookup.
3. A training intervention that separates the two components. Each data cue causes its own failure,
   and each targeted diversification removes only its own failure. Removing the layout leaves it
   unlearned rather than fixed.

This is a substantially stronger paper than before. My estimate is about **7/10 for a mid-tier venue**
and a credible workshop paper at a top venue, with the limits already noted: synthetic task, 1.6M
parameters, 3 seeds, and a single architecture family.

## Recommended final experiment

**A4: varied distance and mixed rewrites together.** Prediction: it removes both failures (P1 at most
1 of 6, no distance deficit) at the general-layout and retention costs of A1 and A3 combined.
- It is the obvious reviewer question ("does combining the two fixes give a robust model?"), and it
  closes the story.
- Cost: 6 models, about 6 h on T4 x2, or about 3 h if the runner admits a single arm.
- Freeze the prediction in a protocol file first, and evaluate at a new seed offset (1500000).

Also worth reporting: A1's 9.7-point general-layout drop at 2-4 rewrites. Varied distance changes
how much filler sits between the memory and the queries, which may interact with multi-rewrite
generalisation. Check the A1 per-seed curves before interpreting it.
