# Block-first layout confirmation: conclusions

Date: 17 November 2026. Sources:
- `research_results/block_layout_confirmation_20261117/`, commit 68086c7. Produced by
  `confirm_block_layout.py` (commit 31342b7) on an NVIDIA RTXA6000-12Q in 178 s.
- `research_notes/analyze_block_layout_errors_20261117.py`, which maps errors to memory positions.

Discovery for this question was in `write_gate_mechanism_review_2026-11-17.md`. This run is the
pre-registered confirmation: the hypotheses and thresholds were frozen in `protocol.json` before any
model was evaluated.

## Validity

- **Models.** 18 overwrite checkpoints: default, mean-alpha and joint initialization x scalar/channel
  x seeds 42/43/44, all 16-epoch, with no new training.
- **Fresh data.** Seed offset 1300000 was never used before. Confirmation offset 900000 remains
  untouched.
- **No training overlap.** The overlap audit against each seed's 100k-example training pool found 0
  overlapping examples in all 81 data cells.
- **Pure layouts.** Every special-layout cell contains exactly its intended write order (for
  example, 256/256 `TTTTUUU`). The general cell mixes 11 non-block orders.
- **Same generator.** Examples come from the unchanged v4 training generator, filtered by write
  order, so they differ from training only in layout.

## Pre-registered verdicts: all confirmed

| Hypothesis | Criterion | Result |
|---|---|---|
| H1 shortcut | majority of models >= 20 points worse on block-first than general (3 rewrites) | **16 of 18** |
| H2 error identity | >= 80% of block-first errors (2-4 rewrites) are the first replacement | **97.1%** (11366 of 11702) |
| H3 invisible in training layout | block-first >= 95% at one rewrite | **18 of 18** |
| H4 needs the unbroken opening run | the three near-miss layouts within 10 points of general | **13 of 16** for each layout |
| H5 distance controls | does the deficit persist with filler between memory and query? | **yes: 14 of 16 (distance 12), 15 of 16 (distance 16)** |

## Conclusions

### 1. The shortcut is real and widespread

Mean latest-value accuracy, rewrites 2-4 pooled:

| Group | Block-first | One unrelated first | Run broken before latest | Initial, gap, run | Other layouts |
|---|---:|---:|---:|---:|---:|
| Shortcut models (13) | 5.4% | 96.5% | 97.4% | 97.0% | 99.6% |
| Broad-failure models (2) | 0.1% | 5.0% | 16.5% | 5.5% | 63.2% |
| Default seed 43 (2) | 100% | 100% | 100% | 100% | 100% |

The joint scalar seed-44 model sits between the shortcut and broad-failure groups. For 13 models,
the rule "if the key's writes form an unbroken run starting at the first memory slot, answer the
run's second write; otherwise answer the latest write" matches 90-100% of their answers at 2-4
rewrites (12 of the 13 are at or above 94%) (column `opening_run_second` in the REPORT). Moving a single unrelated write to the front,
or breaking the run just before the latest write, restores near-perfect accuracy.

### 2. Ordinary validation cannot detect it

On the training layout (one rewrite), all 18 models score at least 95% on block-first examples. In
training, "second write of the run" and "latest write" are the same token. The failure appears only
when the layout is held fixed and something else shifts.

### 3. On block-first examples, retrieval is by position, not by key

With 4 or 8 extra filler tokens between the memory and the query (distances 12 and 16), the
shortcut models fail block-first examples **even at one rewrite**: 23.6% and 53.0% correct. On every
other layout they stay at 94.6-95.3%. The wrong answers are not stale values of the queried key.
They are almost always the value of an **unrelated** key at one structural position, the
second-to-last write in the memory. Seed 42 at distance 12 is the clearest case: 1295 of 1308 errors
come from that slot.

A key-matching lookup would not return another key's value. So on block-first layouts, these models
address memory by position, while on other layouts they use a key-based lookup that survives the
same distance shift. One training distribution, one architecture and one checkpoint can therefore
hold two different retrieval strategies, and which one runs depends on where the key's writes sit.

The exact positional rule is not identified. Without a gap it returns slot 1; with a gap it returns
the second-to-last slot. It plausibly depends on the memory being adjacent to the query, which is
only true for block-first examples. Block-first examples also have the longest leading filler, so
absolute position cannot be fully separated from structure here.

### 4. Two distinct failure types

The two broad-failure models (mean-alpha channel seed 43, mean-alpha scalar seed 44) fail every
layout that has any repeated run. They agree best with "second occurrence" (60-63%), so they did not
learn general latest-write updating at all. They should be reported separately, not pooled with the
layout shortcut.

### 5. Gate type does not matter; initialization is suggestive at best

- Scalar and channel-wise gates behave the same: block-first accuracy at 2-4 rewrites is 34.9% vs
  36.8% (default), 4.0% vs 10.0% (mean-alpha) and 4.0% vs 2.4% (joint).
- The only models without the shortcut are default seed 43, both variants. That is 2 of 6 default
  models vs 0 of 12 mean-alpha or joint (hypergeometric p of about 0.10).
- Both clean models share seed 43, and so share data order and backbone initialization. They are
  effectively one independent event, so no initialization claim is supported. Both this run and the
  earlier gating study point the same way: seed matters more than gate design.

## What this means for the paper

The core claim is now pre-registered and confirmed on fresh data:

> Small Gated DeltaNet models trained to near-perfect accuracy on a keyed overwrite task learn two
> retrieval strategies within one distribution: key-based lookup for most memory layouts, and
> position-based retrieval for the roughly one third of training examples where the queried key's
> writes open the memory. The positional strategy is invisible on the training distribution (all 18
> models >= 95%), but fails deterministically under extra rewrites or small distance shifts. It
> appears in 16 of 18 models regardless of gate granularity or decay initialization.

This is a cleaner and more general contribution than the original gating question. It also fits a
known theme, shortcut learning and underspecification, applied to recurrent associative memory, so
it needs to be positioned against that literature. It does not depend on beating a baseline.

Remaining weaknesses: synthetic task, 1.6M parameters, one architecture family, three seeds, and no
demonstrated training fix yet.

## Recommended next experiment: remove the positional cue in training

The distance-control result predicts which fix should work. The block-first layout lets position
stand in for the key, because the query distance is always 8 and the memory always ends right
before the query. Planned arms, each default initialization x scalar/channel x seeds 42-44, at
16 epochs with matched examples and tokens:

1. **Control:** the current one-rewrite data (already trained; reuse those checkpoints).
2. **Varied query distance:** sample distance from 8 to 16 during training, so position no longer
   identifies the answer.
3. **Block-first removed:** the same data with block-first layouts rejected.

Rerun this confirmation script with a new seed offset. Predictions to freeze beforehand: arms 2 and 3
remove the block-first deficit (H1 fails) without lowering accuracy on other layouts. If arm 3 works
but arm 2 does not, the cue is the layout itself rather than distance. Cost: 12 new models, about
5h45m on the A6000 based on the joint run's measured rate.

Optional mechanism work: logit lens and activation patching on block-first vs non-block examples, to
locate the layer where the positional route takes over.
