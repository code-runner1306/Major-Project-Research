# Write-gate mechanism analysis: review

Date: 17 November 2026. Sources:
- `research_results/write_gate_mechanism_20261117/` (commit 7869d02): `analyze_write_gate.py` run on
  the A6000 over 18 overwrite checkpoints, with tables from `research_notes/analyze_write_gate_20261117.py`.
- `research_notes/analyze_block_layout_20261117.py`: a local CPU follow-up on the 12 legacy checkpoints.
  Joint weights are not local, so the follow-up covers default and mean-alpha models only.

The data reuse seed offset 500000, already observed in the stress study, so all of this is discovery
evidence. Confirmation offset 900000 is untouched.

## Run integrity

- 18 models: default, mean-alpha and joint initialization x scalar/channel x seeds 42/43/44.
  256 examples per cell; total wall time 249 s on the A6000.
- The serial instrumented recurrence agrees with the stock chunked model on every prediction.
- The unpatched three-rewrite outcomes reproduce `memory_stress_tests_20261117/error_breakdown.json`
  exactly for all 12 legacy models.

## Main finding: the failure is a deterministic, layout-specific shortcut

Several different models have identical scores on a seed's data. For example, seed-42 default
channel, mean-alpha scalar and joint channel all score 71.9 / 65.2 / 67.2% at 2 / 3 / 4 rewrites.
The models therefore fail on the same examples.

Splitting examples by memory layout explains it. Call an example **block-first** when every write of
the queried key comes before every unrelated write: the key's writes form one unbroken run that opens
the memory (`post_target_write_count == 3`). The table shows latest-value accuracy on block-first
examples / all other examples.

| Seed | Model | 2 rewrites | 3 rewrites | 4 rewrites |
|---|---|---|---|---|
| 42 | default scalar | 0/72 \| 180/184 | 0/89 \| 154/167 | 0/84 \| 161/172 |
| 42 | default channel | 0/72 \| 184/184 | 0/89 \| 167/167 | 0/84 \| 172/172 |
| 42 | mean-alpha scalar | 0/72 \| 184/184 | 0/89 \| 167/167 | 0/84 \| 172/172 |
| 42 | mean-alpha channel | 1/72 \| 184/184 | 10/89 \| 166/167 | 14/84 \| 171/172 |
| 43 | default scalar | 90/90 \| 166/166 | 91/91 \| 165/165 | 90/90 \| 166/166 |
| 43 | default channel | 90/90 \| 166/166 | 91/91 \| 165/165 | 90/90 \| 166/166 |
| 43 | mean-alpha scalar | 9/90 \| 166/166 | 9/91 \| 165/165 | 7/90 \| 166/166 |
| 43 | mean-alpha channel | 0/90 \| 103/166 | 0/91 \| 93/165 | 0/90 \| 93/166 |
| 44 | default scalar | 3/90 \| 166/166 | 2/95 \| 161/161 | 3/95 \| 161/161 |
| 44 | default channel | 15/90 \| 166/166 | 12/95 \| 161/161 | 5/95 \| 161/161 |
| 44 | mean-alpha scalar | 0/90 \| 32/166 | 0/95 \| 62/161 | 0/95 \| 64/161 |
| 44 | mean-alpha channel | 35/90 \| 166/166 | 5/95 \| 161/161 | 12/95 \| 161/161 |

- **Block-first examples:** 10 of 12 models are wrong on nearly all of them, and the wrong answer is
  almost always the first replacement, which is the second write of the run. Only the two default
  seed-43 models handle them.
- **All other examples:** 9 of 12 models are at or near 100% even with 4 rewrites. So in most layouts
  these models do implement last-write-wins and generalise beyond the single rewrite they were
  trained on.
- **The other failures** are concentrated in two models (mean-alpha scalar 44, mean-alpha channel 43),
  plus a few errors in default scalar 42.
- **The joint models** were not rerun locally. Their pooled first-replacement rates (28-42%) match the
  block-first share of each seed's data, which is consistent with the same pattern but not checked
  per example.

This explains the earlier plateau. First-replacement errors stay at about 31-32% for 2, 3 and 4
rewrites because the block-first share of the data stays at about 28-37%. They are not a
degradation that grows with the number of rewrites.

**Why training hides it.** In the one-rewrite training distribution, block-first examples are about
a third of the data (666 of 2000 sampled). There the key's two writes fill memory slots 0 and 1, so
"answer the second write of the opening run" and "answer the latest write" give the same answer.
Training accuracy is 99.98%. Most models appear to have learned a separate rule for this layout that
does not generalise, while using a general rule elsewhere.

What the rule is exactly remains open. It is not "answer memory slot 1" or "answer the second
occurrence" in general: layouts where the first replacement sits in slot 1 but an unrelated write
precedes the latest write are answered correctly. The failure needs the unbroken run starting at
slot 0.

## Supporting mechanism results (pooled over 18 models, three rewrites)

**The write gate is not the cause.**
- β at the value token is flat across all four writes (0.57-0.60 in every layer), the same for
  correct and wrong answers, and the same as on the training distribution.
- Forcing β = 1 on the latest write: -0.80 points (0 of 18 models improve). Copying the first
  rewrite's β onto later writes: -0.02 points.

**Aligning keys is the only large lever, but it is not a fix.**

| Patch | Layers | Latest | Unrelated | First-repl. rate | Models up/down |
|---|---|---:|---:|---:|---|
| Earlier keys set to latest key | all | +16.21 | -18.40 | -23.65 | 14 / 0 |
| Earlier keys set to latest key | 1 only | +7.66 | -5.92 | -11.61 | 10 / 0 |
| Remove first replacement's write | all | +4.12 | -6.86 | -10.24 | 6 / 1 |
| Control: β = 1 on an unrelated write | all | +0.95 | -0.39 | -0.89 | 2 / 0 |

Making repeated writes share one key lets the delta rule erase old values, which partly overrides the
shortcut. It also costs 18 points of unrelated retention, and it imposes the textbook algorithm
rather than revealing the model's own. The per-model gains vary widely (0 to +49 points), roughly in
line with how many block-first errors each model makes.

**Readout attribution.** On wrong answers, layers 2-3 draw 0.44-0.52 of the readout weight from the
target's write tokens, against 0.18-0.21 when correct. Most of it comes from the key tokens of later
rewrites. In block-first layouts these tokens form one run of repeated keys. A carry-forward
explanation fits but is untested: earlier layers look up the key at each repeat and copy a stale
value forward. Attribution shares are scalar weights on value vectors of different norms, not logit
contributions.

## Interpretation for the paper

The headline changes from "models fail repeated overwrites" to a sharper and more interesting claim:

> Small GDN models that reach 99.98% held-out accuracy implement a layout-conditional rule. On about a
> third of the training distribution they use a rule that is indistinguishable from last-write-wins
> during training but fails deterministically under extra rewrites. Elsewhere they generalise. This
> is invisible to IID validation, appears in both gate types, and only 2 of 12 locally checked models
> avoid it.

Initialization comparison: only default seed 43 (both variants) avoids the shortcut. No mean-alpha
model avoids it, and the joint models appear not to either. Two clean models out of 18 is too few to
claim that initialization causes the difference.

## Next steps

1. **Layout-stratified fresh test.** Generate block-first and non-block examples in fixed proportions
   at 1-4 rewrites with a new offset, for all 18 models, to confirm the rule on unobserved data. Add
   layouts that separate "unbroken opening run" from nearby alternatives: a run that starts after a
   single unrelated write, and a run broken only after the latest write.
2. **Training-side test.** Train with the block-first layout removed, or with mixed rewrite counts,
   at matched exposure. Ask whether the shortcut disappears and how often it forms (it appeared in
   10 of 12 models here).
3. **Circuit localisation for the block-first layout only.** Use the logit lens and activation
   patching at the repeated key tokens, comparing block-first with non-block examples.

Storage note: `diagnostics_per_write.csv` (more than 100 MB) is not in the repository. The tables use
`diagnostic_summary.json`.
