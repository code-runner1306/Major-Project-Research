# Checking the proposed overall conclusion against all results

Date: 18 November 2026.

**Proposed conclusion:** "TinyStories at 10M, 30M and 55M showed minute differences; overwrite and MQAR showed
similar models; the Anatomy harness showed similar models at small scale. Therefore channel-wise gating gives no
implicit benefit at the SLM level, and neither architecture is better or worse."

## Verdict

The proposed conclusion is **broadly in the right direction, but it cannot be written as stated.** Two of its three
pillars are not what the conclusion says.

## 1. Language-model scale study (TinyStories)

| Claimed | What exists |
|---|---|
| 10M, 30M, 55M compared | **Only 30M has a real comparison.** |

- **55M was never trained.** `output/tinystories/55M/` contains the 30M runs copied verbatim: identical checkpoint
  hashes, identical losses, and 29,272,320 / 30,071,040 parameters. The 55M rows in `output/scale_results.md` and
  `output/seed_level_results.md` match no run. `seed_level_results.md` itself calls them projected.
- **10M has one seed per variant,** with validation loss 2.811 (scalar) vs 2.812 (channel-wise). The values in
  `scale_results.md` (2.408 / 2.426) do not match these runs.
- **30M:** scalar 2.366 (n=3) vs channel-wise 2.378 (n=2). Channel-wise is 0.012 nats worse, despite having about
  0.8M more parameters. One channel-wise run's lifecycle reads `evaluating`.
- **All runs** use 1000 steps and 16.4M tokens, about 0.55 tokens per parameter at 30M, so they are heavily
  undertrained. Zero-shot retrieval is 0/10 for both.

**Consequence:** a "10M/30M/55M" claim cannot appear in a paper. Report 30M (and 10M as one seed) as is, or train the
missing sizes. The projected tables must not be presented as results.

**External evidence points the other way:** Das (TMLR) finds channel-wise KDA better than scalar DeltaNet by about
0.04 nats in all 6 seed pairs at 10M and 100M, with far more tokens. Our 30M point does not refute that. It is too
small and too undertrained.

## 2. Synthetic memory tasks (1.6M GDN, overwrite and MQAR)

Paired scalar-vs-channel model comparisons on fresh held-out stress cells (decay-boundary evaluation; 3 seeds × 3
initializations = 9 model pairs per task; accuracy averaged over cells per model):

| Task | Channel better in | Sign test | Mean difference | Notes |
|---|---:|---:|---:|---|
| MQAR | 4 of 9 | p = 1.0 | +3.4 pts | driven by one failed scalar seed (+34.8); otherwise -6 to +5 |
| Overwrite stress | **8 of 9** | **p = 0.039** | +3.2 pts | typical +0.7 to +4 pts; two large swings (-22, +36) |

- **Final development accuracy** is near 100% for both variants in every study.
- **Layout shortcut:** identical in both gate types in every arm.
- **Zoology MQAR test:** both robust.
- **Mechanism:** channel-wise models do use their channel diversity. Averaging the channels per token drops their
  MQAR accuracy from 87-94% to 40-70%, while fixed per-channel decay keeps it. Scalar models nevertheless reach
  the same accuracy by other means.

**So:** there is no difference on MQAR. On overwrite stress there is a small, fairly consistent channel-wise edge.
It is exploratory (not a pre-registered hypothesis), uncorrected for the many comparisons we ran, and the pairs
are not fully independent (pairs that share a seed share data). It is a hint, not a finding. It is not evidence of
equivalence either.

## 3. Anatomy harness (dualgoose)

**Not complete.** Only the channel-wise arm has finished (4-5 seeds per cell). It reproduces the published numbers
almost exactly. No scalar run has finished yet, so this pillar cannot be cited until it does. The most likely
outcome is a null, but that is a forecast.

## Corrected conclusion that the evidence supports

> In small Gated DeltaNet models (1.6M parameters on synthetic recall and overwrite tasks; one 30M language-model
> comparison), channel-wise decay gave no large or reliable advantage over scalar decay. Final accuracies were
> indistinguishable; held-out overwrite robustness showed at most a small, exploratory channel-wise edge (8 of 9 model
> pairs, about +3 points); MQAR showed none; and at 30M the channel-wise model was slightly worse (+0.012 nats, n=2-3).
> Channel-wise models do rely on their channel diversity, but scalar models reach the same accuracy. These results
> are limited to very small, short-trained models and do not contradict reports of channel-wise gains at larger
> scale and longer training.

**Wording to avoid:**
- "SLM level": these models are 1.6M-30M, far below the usual meaning of small language model.
- "neither is better or worse": not shown; there was no equivalence test, and power was low.
- "10M/30M/55M".
- Any dualgoose result before it finishes.
