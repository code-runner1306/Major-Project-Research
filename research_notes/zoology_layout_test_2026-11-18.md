# Does the layout shortcut appear on standard Zoology MQAR?

Date: 18 November 2026. Script: `zoology_layout_test.py`. Results:
`research_results/zoology_layout_test_20261118/`. Hypotheses Z1-Z4 were frozen in `protocol.json`
before evaluation.

## Setup

- **14 models already trained on the standard Zoology MQAR generator.**
  - 12 are 16-epoch models: default and mean-alpha initialization x scalar/channel x seeds 42-44.
  - 2 are the 64-epoch published-recipe models.
  - `recipe_data.mqar` is a byte-equivalent port of pinned upstream Zoology (`random_non_queries=False`),
    so this is data we did not design. No new training was done.
- **Fresh data.** Seed offset 1600000, 512 examples per cell, and 0 overlapping examples with each
  seed's 100k training set.
- **The cells.** Each cell changes one structural regularity of Zoology:
  - memory shifted by 1, 2, 4, 8 or 16 filler tokens (an odd shift also swaps key/value parity);
  - all queries near, or all queries far;
  - queries in memory order, or in reverse order.
- **Joint MQAR models were not included,** because their weights are not on this machine. Add them
  with `--joint-root` on the GPU machine.

## Results

| Verdict | Result |
|---|---|
| **Z1** sanity: Zoology baseline >= 95% | **14/14** (mean 99.94%) |
| **Z2** shortcut: a memory shift costs >= 20 points | **1/14**: default scalar seed 44, 75.3% at shift 1 |
| **Z3** errors anchored to the old positions | **No**: 34% of shift-2/4 errors, all from one cell |
| **Z4** query near/far/order within 10 points | **14/14** in every query-layout cell |

- **Query placement has no effect at all.** Near, far, memory order and reverse order are all
  99.8-100% in every model.
- **Shifting the memory away from position 0 costs little on average:** 97.6-98.6% mean accuracy.
- **Three models lose 5-25 points under memory shifts:** default scalar 44, mean-alpha channel 43 and
  mean-alpha scalar 42.
- **Those errors are confusions with another memory pair** (535-664 per shift cell). They are not
  answers anchored to the old absolute positions.
- **Default scalar 44 is the same model that was already the weakest MQAR model** (53% with eight
  associations in the earlier stress test). Its deficit is a general fragility, not a layout rule.

## Conclusion

**The layout-conditional shortcut does not appear on standard Zoology MQAR.** Models trained on the
upstream generator do key-based retrieval that is robust to query placement and largely robust to
moving the memory. Some models show a mild dependence on absolute position, but not a shortcut of the
overwrite kind.

This is the outcome the shortcut account predicts, so it narrows the claim rather than weakening it.
- **Zoology MQAR offers no part of the training data where position alone determines the answer.**
  Each key is written once, the queried pair varies, and queries move.
- **Our overwrite data had such a part.** In block-first examples, the answer always sat in memory
  slot 1 at query distance 8, and "second write" always equalled "latest write".
- **The layout-fix study removed the failures cue by cue,** which shows each one depends on its own
  cue: mixed rewrite counts removed the rewrite failure, and varied query distance removed the
  position failure.

The defensible claim is therefore conditional:

> Gated DeltaNet models learn a position-based route for a sub-population of training inputs when a
> structural cue predicts the answer perfectly within that sub-population. Ordinary validation
> cannot detect it. Standard MQAR, which has no such sub-population, shows no such route.

MQAR is a useful negative control for that claim. The paper should state plainly that the effect is
not a general property of GDN on associative-recall benchmarks.

## Next external test

The informative published setup is an overwrite benchmark that contains a perfectly predictive cue.
- **KATA's repeated-key overwrite (arXiv 2607.17419, Section 5.2) writes every key exactly twice.**
  That makes "second write" and "latest write" the same throughout training, which is the ambiguity
  A3 showed causes the rewrite failure.
- **Prediction:** models trained on it fail when a key is written three or more times.
- **Caveat:** KATA's token format and write ordering are not in its public text. A reimplementation
  from the description would be "their task design, our generator", and must be described that way.
