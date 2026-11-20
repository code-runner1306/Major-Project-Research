# Content brief: Layout fix A0–A4 (`layout-fix`)

Audience: technical but new to this field. Plain English first; maths only in `MathToggle`. Use glossary terms via `<Term term="...">`.
All numbers must come from `GET /api/layout-fix/summary` (the cached fresh evaluation) or `POST /api/layout-fix/try`; never hard-code a result number.

## Facts (verified)
- Task ("overwrite task"): a sequence writes 4 keys with values into memory; one key (the *queried key*) is written again with a new value (a *rewrite*). Then the model is asked for that key's **latest** value and for an **unrelated** key's value. Vocabulary 8192 tokens; keys come from the lower half, values from the upper half; token 0 is a filler / query marker, token 1 replaces erased answers.
- Block-first layout: every write of the queried key comes before every other key's write (pattern `TTUUU`, T = queried key write, U = unrelated write). About 1 in 3 training examples are block-first by chance.
- A0 training data: query distance always 8 tokens, always exactly 1 rewrite. So in block-first training examples the latest value is always the 2nd write. Models learned "answer the 2nd write" for that layout: a position rule, not a key lookup.
- On the normal validation set the A0 models scored 97–100% (`curves[].final_dev`), so validation could not reveal it.
- With 3 rewrites in a block-first layout, A0 models answer the **first rewrite** (the 2nd write) instead of the latest.
- Arms (training-data changes, everything else identical incl. starting weights):
  A0 control (distance 8, 1 rewrite) · A1 varied distance (8–16) · A2 no block-first (those layouts removed) · A3 mixed rewrites (1–3) · A4 distance + rewrites (both A1 and A3).
- Result (fresh data, never seen in training, overlap audit = 0 in all 15 arm×seed training sets): shortcut models A0 6/6, A1 3/6, A2 4/6, A3 0/6, A4 0/6 (`shortcutModels` = block-first 3-rewrite accuracy ≥ 20 points below general-layout 3-rewrite accuracy). Compute these from the summary, do not type them.
- Both decay granularities behave the same here: gate design does not cause or cure the shortcut.
- 30 runs = 5 arms × 2 decay granularities × 3 seeds. 16 epochs, 3,136 optimizer steps each, ~1.6M parameters (scalar 1,585,416 / channel-wise 1,651,456).

## Sections (order) and what each shows
1. **question**: "Can a model that scores ~99% still be learning the wrong rule, and can changing only the training data fix it?" Visual: an animated two-panel "what the model sees": a normal layout vs a block-first layout drawn as token chips (use `showcase_trained_layout` and `showcase` from the summary with `TokenStrip`), with the queried key's writes highlighted.
2. **model**: the model is the same in every arm; only data changes. Visual: simple animated block diagram (SVG): tokens → embedding → 4 × [short convolution → Gated DeltaNet memory (decay gate α, write gate β) → output] → prediction; side note "scalar vs channel-wise: only α's width differs". Show parameter counts from `/api/runs` + the summary's `gates[variant].parameters`.
3. **code**: `CodeWalkthrough` steps:
   1 Dataset: `retrieval.py` symbol `RetrievalDataset._overwrite_v4`; notes on `rng.sample(value_pool, overwrite_count)` (picks the new values for rewrites), `latest_index = rng.randint` (where the latest write lands), `memory = [token for key, value` (builds the write stream).
   2 The one line that differs: `layout_fix_data.py` lines 20–28 (the `ARMS` dict); notes on each arm line in plain words.
   3 Model: `recipe_model.py` symbol `RecipeGDN.__init__`; notes on `self.gate_width=1 if variant=='scalar' else self.head_k` ("this is the only scalar/channel-wise difference"), `self.v_proj`, `self.a_proj`, `self.b_proj`.
   4 Training: `run_layout_fix_study.py` lines 127–150; notes on `CosineAnnealingLR`, `for micro in range(offset,offset+actual,32)`, `optimizer.step()`.
   5 Evaluation: `confirm_block_layout.py` symbol `sample_layouts` (draws fresh examples until each layout has enough) and symbol `evaluate`.
4. **results**: (a) `AnimatedBars` per arm: block-first 3-rewrite accuracy, with arm colours; (b) shortcut model count per arm as an animated "6 dots per arm, filled = has shortcut" grid; (c) `AnimatedLines` of validation accuracy per epoch for all 30 runs (dashed = channel-wise) showing every arm converges to ~100% (that is the point: validation cannot tell them apart); (d) where the wrong answers come from: stacked bars from `answer_source` (latest / first_rewrite / initial / other); (e) saved report vs today's rerun table (`saved_a4_report` vs `fresh_results`). Each with `WhatThisShows`.
5. **inside**: (prototype later in 2–3 variants) the recurrent-state "player": step through tokens of one example, Heatmap of the layer-4 state crop after each token (`state_crops`), α bars for the current token per layer, β value, and the read at the query. Compare an A0 and an A4 model side by side on the same block-first example.
6. **try**: controls: arm, decay granularity, seed, layout (block_first / general_nonblock / block_after_one_unrelated / run_broken_before_latest / initial_then_run), rewrites 1–4, distance 8–24, example seed (with a "new example" dice button). Calls `/api/layout-fix/try`; shows `TokenStrip`, each query's gold vs prediction, `which_write` in words ("returned the first rewrite"), top-5 probabilities as `AnimatedBars`, plus a compact version of the state player. Explain beforehand: "Pick an A0 model, block-first, 3 rewrites, and watch it answer the wrong write; switch to A4."
