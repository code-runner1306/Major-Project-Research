# Content brief: Longer sequences and filler decay (`length`)

Question: how do the models behave on longer inputs, and what happens to stored memory across long stretches of
filler text (the decay gate shrinks memory by a product of per-token factors)?
Status: **live**. The 12 local MQAR checkpoints exist (see Local checkpoints). Results come from saved files; the try
and inside views run the checkpoints on CPU.

## Setup (verified from the scripts)
- Models: the same 4-layer Gated DeltaNet (`RecipeLM`, width 128, vocabulary 8192) trained on 64-token MQAR
  (`run_paper_mqar_length_extension.py`, `checkpoint_endpoint`: 16 epochs / 3136 optimizer steps).
  12 checkpoints = initializations {default, calibrated ("mean-decay" on the page) } × decay granularity {scalar,
  channel-wise} × training seeds {42, 43, 44}.
- Input: the 64-token MQAR example. The first 8 tokens are the four key/value writes. Then `mqar()` puts the four
  query keys at random gaps among token 0 (the filler ID), so the normal 64-token input already contains many zeros.
- Extension: the script inserts extra token-0 filler after the 8-token prefix, giving lengths 64, 256 and 1,024
  (filler counts 0, 192, 960). Data: `mqar(examples, length=64, seed=1950000 + training seed)`; the sweep used 256
  examples per seed. Training used `mqar(100000, seed=100000 + seed)`; the script asserts no sweep example is in the
  training set.
- Metric: exact answer-token accuracy (`evaluate`), 4 answers per example, so 1,024 answers per cell.

## Length sweep (verified facts; `research_results/mqar_length_extension_20261119/results.json`, status complete)
- 36 cells = 12 models × 3 lengths. `verified_results.json` matches `results.json` cell for cell (same correct count
  and checkpoint SHA-256 in all 36). Both are used; the page reads `results.json` and reports the cross-check.
- Mean across the 12 models: 99.97% at 64, 94.07% at 256, 79.52% at 1,024.
- At 256: only default/channel-wise seed 44 falls below 90% (29.4%).
- At 1,024: three models fall below 90%: calibrated/scalar seed 43 (31.4%), default/channel-wise seed 44 (24.9%),
  calibrated/scalar seed 44 (0.0%). Failures appear in both granularities and both initializations, so they are seed
  specific, not a granularity effect. Of the 6 scalar models, 4 stay at or above 99.9% at 1,024 and 2 fail.
- Checkpoint audit (`checkpoint_hash_audit.json`, status complete): 12 of 12 SHA-256 hashes match across the sweep
  record, the fresh confirmation record and the file on disk.
- Targeted replays (`targeted_replays/`): default/channel-wise seed 44 at 256 tokens = 301/1,024 correct (matches the
  sweep); calibrated/scalar seed 44 at 1,024 = 0/1,024 (matches). Both have per-query NPZ predictions.

## Filler decay diagnostic (verified; `research_results/filler_decay_diagnostic_20261119/`, status complete)
- Scope: seeds 43 and 44 only, 16 examples per seed (64 answers per cell), lengths 256 and 1,024 (filler 192 and 960),
  all 4 layers, both granularities, both initializations (16 rows in `results.json`).
- Measure: alpha (decay gate) on the inserted filler only, and the product over those tokens per channel and example.
  `alpha = exp(-exp(A_log) * softplus(a_proj(x) + dt_bias))`. Scalar has 1 channel; channel-wise has 128.
- Guardrail (from protocol.json, must appear on the page): alpha products omit the write strength (beta) and do not by
  themselves imply state retention.
- Conflict to note: the diagnostic's accuracy on its 16-example draw (e.g. seed 44 default/channel-wise at 256 = 20/64)
  differs from the 256-example sweep (29.4%); these are different example sets. Not a contradiction.
- The decay product does not separate working and failing models. Calibrated/scalar seed 43 (24/64 at 960 filler)
  has layer medians 0.107, 0.688, 0.844, 0.979, while default/scalar seed 44 (64/64) has layer 3 median 9.5e-5.
- Histogram (`product_histogram.json`): channel-wise, mean-decay, layer 4, 960 filler, seeds 43 and 44; 128 channels ×
  16 examples = 2,048 products per seed, binned into 20 bins on [0, 1]. Counts sum to 2,048 per seed.

## Live computation (from the checkpoints)
- Causal model: the decay gate at a filler position depends only on earlier tokens. One forward pass with 960 filler
  tokens therefore gives the cumulative product for every filler count up to 960. The live endpoint returns the
  survival curve from this single pass.
- Reproduction check: with 16 examples from `mqar(16, seed=1950000 + seed)`, the live median product at 192 and 960
  matches the saved diagnostic (e.g. seed 44 calibrated/scalar layer 1: 0.481 at 192, 0.0258 at 960).
- Cost: one 16 × 1,024 forward takes about 5 to 7 s on this CPU; 1 × 256 takes about 0.13 s. Examples are capped at
  16 for this reason.

## Local checkpoints
- 12 MQAR files under `output/calibrated_recipe_study/matched-decay-v1/*/mqar/seed_*/{default,calibrated}/{scalar,channel-wise}/latest_checkpoint.pt`.
- Loaded with `analyze_write_gate.load(path, variant, seed)`.

## Sources not used for numbers
- `decay_design_results.md` covers a different question (DeltaNet vs channel-wise training loss on TinyStories). It has
  no filler or extrapolation results, so no number from it appears on this page.
- `run_paper_filler_decay_diagnostic.py` and `run_paper_filler_product_histogram.py` contain figure paths outside this
  repo; the figures are not used.

## Sections
1. **question**: How do the models handle inputs 4× and 16× longer than training? Visual: mean accuracy at 64, 256 and
   1,024 tokens (AnimatedBars), with the headline count of models below 90% at 1,024.
2. **model**: the 12 models (initialization × granularity × seed, with local badges), and the input layout: 8-token
   prefix, filler run, rest of the example.
3. **code**: CodeWalkthrough over `run_paper_mqar_length_extension.py` (insertion), `recipe_data.py` `mqar` (query gaps),
   `recipe_model.py` `RecipeGDN.decay` (the gate).
4. **results**: (a) accuracy by length, all 12 models (AnimatedLines or dumbbell); (b) targeted replays; (c) audit.
5. **inside**: live survival curve for the chosen model (log10 of the decay-only product vs filler count, per layer),
   with saved values at 192 and 960 marked. Guardrail shown.
6. **try**: pick a model, filler count (0 to 960) and examples (1 to 16). Live accuracy, the example tokens, and the
   survival curve.
