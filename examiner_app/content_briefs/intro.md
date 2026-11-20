# Content brief: Introduction (`intro`)

Goal: a non-specialist understands, visually, what a Gated DeltaNet memory is and how scalar and channel-wise decay differ, before any experiment.

## Facts (verified)
- The model reads tokens one at a time and keeps a fixed-size memory (the recurrent state): a grid of 256 × 128 numbers per layer (value width × key width), 4 layers. Memory size never grows with the sequence.
- Each token: (1) fade the memory by the decay gate α, (2) look up what the memory currently returns for this token's key, (3) write back only the difference between the wanted value and that lookup, scaled by the write gate β. Equation (maths toggle): `S' = S·diag(α) + β (v − S̃k) kᵀ`, `S̃ = S·diag(α)`.
- Scalar decay: α is ONE number per token per layer; the whole grid fades together. Tensor shapes: decay projection `[1 × 128]`, α for a 64-token sequence `[1, 64, 1, 1]`.
- Channel-wise decay: α is 128 numbers per token per layer, one per key column; each column fades at its own rate. Decay projection `[128 × 128]`, α `[1, 64, 1, 128]`.
- Same memory size for both. Channel-wise adds 66,040 parameters (+4.17%): 1,585,416 vs 1,651,456.
- Overall finding (state neutrally, with links to the experiment pages): in these small models channel-wise decay gave no reliable recall advantage; what mattered more was the structure of the training data.
- Real tensors: `GET /api/layout-fix/summary` → `delta_rule["A4_distance_and_rewrites/scalar"]` and `["…/channel-wise"]` (state_before / state_decayed / state_after 32×32 crops, alpha, beta, read_error_before/after, max_abs_error_vs_model_recurrence) and `gates[variant].layers[i].alpha` ([T][width]) for the showcase sequence.

## Sections
1. **question**: "What is a Gated DeltaNet's memory, and what changes between scalar and channel-wise decay?" Visual: whiteboard analogy animation: an 8×8 grid of ink cells; scalar = one dimmer fading the whole board evenly; channel-wise = a dimmer per column fading columns at different rates. Loops gently.
2. **model**: interactive toy (computed in the browser, no API): 4×4 memory with sliders for α (one slider for scalar; 4 sliders for channel-wise via a toggle), β, and a choice of key/value. Animate the three steps (fade → read → write the difference) in sequence with motion, showing the grid after each step and the read error before/after. Maths in MathToggle.
3. **code**: `CodeWalkthrough` steps: 1 The decay gate: `recipe_model.py` `RecipeGDN.decay` (notes: `A_log`, `softplus`, `exp(-...)` gives a number between 0 and 1); 2 The memory update: `recipe_model.py` `serial_rectangular` (notes: `state=state*alpha` fade, `residual=v... - state@key` what is missing, `state=state+beta` write); 3 The one line that differs: `RecipeGDN.__init__` note on `self.gate_width=1 if variant=='scalar' else self.head_k`.
4. **results**: the dimension strip: side-by-side cards of tensor shapes and parameter counts for both decay granularities (from summary `gates`), animated count-up numbers; small caption that the memory grid is identical.
5. **inside**: real checkpoints: for the same input sequence (summary `showcase`), Heatmaps of α over time for scalar (1 row × 64 tokens) and channel-wise (128 rows × 64 tokens) side by side, and the real layer-4 memory before → faded → after for both, animating through the three stages. Show `max_abs_error_vs_model_recurrence` as "our hand calculation matches the model to 0.00000001".
6. **try**: link cards to each experiment page ("Where to go next"), plus a mini toy repeat with random key/value buttons.

Bottom of the page (small, muted, single line): "Published comparison values marked * on the TinyStories page are from Das (TMLR), 'decay design' study; all other results are from this project's own runs."
