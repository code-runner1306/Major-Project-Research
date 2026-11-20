# TinyStories brief (page id `tinystories`)

Verified from repo files on the day of writing. Result files win over prose.

## Question
Do the decay designs (scalar vs channel-wise decay) make a difference when models learn from short simple stories instead of synthetic tests?

## Measured here (ours), from `output/tinystories/<size>/<variant>/<run>/metrics.json`
Budget for every run: 1,000 optimizer steps, 16,384,000 processed tokens, batch 8 x grad-accum 4 x 512 context. Validation loss = `final_validation.loss` (200 eval batches, step 1000).

| Size | Variant | Seeds | Params | Val loss (per seed) |
|---|---|---|---|---|
| 10M | scalar | 42 | 10,030,880 | 2.810937 |
| 10M | channel-wise | 42 | 10,155,680 | 2.811991 |
| 30M | scalar | 42, 43, 44 | 29,272,320 | 2.372764, 2.359883, 2.365776 |
| 30M | channel-wise | 42, 43 | 30,071,040 | 2.377349, 2.378653 |

- 10M and 30M are the only measured sizes. Pages must show only these as ours.
- `output/tinystories/55M/` is a COPY of the 30M run (same run folder hashes, metrics say scale "30M"). Never show it as a 55M result.
- Parameter counts are read from `metrics.json` `parameter.parameter_count`.
- The 30M models are 10 layers, d_model 320, 8 heads, hybrid: layers 3 and 7 (0-indexed) are full softmax attention, the other 8 are Gated DeltaNet (GDN). Gate values exist only in the GDN layers. 10M is 6 layers with one attention layer (layer 3).
- Saved per-run training curves: `training.train_losses` (1,000 values per run).
- Saved gate diagnostics (seed 42 scalar and channel-wise, 10M and 30M): `gate_diagnostics.layers.<i>` with half-life quantiles, mean, channel diversity.
- Local checkpoints: `final_checkpoint.pt` in each run folder. Keys: `model_state`, `completed_steps` (1000), `processed_tokens`. Loads strictly into `model.GatedDeltaNetLM` built from `manifest.json` `model` section (d_model, n_layer, n_head, d_ff list, attention_layout hybrid), variant scalar or channel-wise. Verified: strict load succeeds, 29,272,320 params.
- Data identity per manifest: TinyStories `roneneldan/TinyStories`, split `validation`, context 512.

## Das et al. (TMLR), NOT run here
Source: `decay_design_results.md` (repo root). The file does not name Das; the user, the deck and the paper (`indicon_gate_paper/main.tex`, `das2026`) do. Sizes and values parsed from its tables:
- 10M*: DeltaNet (scalar) 2.3222 +- 0.0173; KDA (channel-wise) 2.2820 +- 0.0196 (3 seeds)
- 55M*: scalar 2.0176; channel-wise 1.9703 (1 seed)
- 100M*: scalar 1.9004 +- 0.0061; channel-wise 1.8614 +- 0.0017 (3 seeds)
- 10M* context 1024 (single seed): not shown on the chart.
Their budget (10M unique tokens, 3 epochs, about 7,300 steps) is NOT ours (1,000 steps). Footnote must say so.

## Conflicts between sources (result files win)
1. `output/scale_results.md` losses do not match `metrics.json`: 10M 2.408101 / 2.426351 (file) vs 2.810937 / 2.811991 (metrics); 30M scalar 2.359883 is seed 43 in metrics, not a mean or seed 42. Its 55M row (2.352110 / 2.374898) appears in no metrics file, and 55M was never trained. Its header says 3 seeds per size; only one 10M seed exists.
2. The deck `presentation/research_walkthrough_2026-11-17.pptx` (slides 6, 7): says 3 seeds each at 10M, 30M, 55M; says Das ran 18M and 125M; says "typical gap under 1%, scalar marginally lower". Files: 10M 1 seed, 30M channel-wise 2 seeds, no 55M run; at 30M scalar is lower on both paired seeds (channel-wise higher by 0.0046 and 0.0188); Das values in the .md are 10M/55M/100M.
3. `decay_design_results.md` section 4 claims channel-wise wins at 10M and 100M in all seeds (Das numbers). Not our result; shown only as Das.
4. Manifest `budget.status` is "pending_v100_pilot" while `lifecycle.json` says complete; `source.dirty` true, revision unavailable. Mention in More detail only if relevant; not on visible slides.
5. `decay_design_results.md` hardware row contains an unrelated URL (turnitin). Not used.

## Our headline (computed on the server from metrics.json)
- At 30M, mean scalar val loss (3 seeds) vs mean channel-wise (2 seeds): channel-wise is higher. The page must compute the numbers, not type them.
- At 10M (1 seed) the two are within 0.0011.

## Live try-it
- Models: 30M seed 42, scalar and channel-wise, both live.
- Tokenizer: GPT-2 via `server/data/tinystories/tiktoken_cache` (set `TIKTOKEN_CACHE_DIR`, as `dataset_generators._gpt2` does).
- Sample: story from `server/data/tinystories/data/validation-*.parquet`, via the existing `dataset_generators.generate('tinystories', seed, {window})`.

## Vocabulary (SPEC): decay granularity, scalar decay, channel-wise decay, decay gate (alpha), write gate (beta), recurrent state, arm, run. Never say "gating" for the scalar/channel-wise difference.
