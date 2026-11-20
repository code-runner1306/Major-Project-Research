# Content brief: Stress tests and gate interventions (`stress`)

Page id `stress`, title "Stress tests and gate interventions", status `live` (12 overwrite checkpoints run on CPU).
All numbers come from `GET /api/stress/summary` (saved result files, read at request time) or
`POST /api/stress/try` (live checkpoints). Never type a result number into the UI.

## Facts (verified against the saved files, 2026-11-20)

### Stress tests (zero-shot, fixed checkpoints, no training)
Source: `research_results/memory_stress_tests_20261117/` (`REPORT.md`, `protocol.json`, `condition_means.csv`, `models.csv`, `mqar_actual_query_distances.csv`).
- 24 checkpoints = 2 tasks x 2 initializations (default, mean-decay) x 2 decay granularities x 3 seeds. Trained 16 epochs at training length 64, 4 keys, distance 8, 1 rewrite.
- 256 examples per condition, seed offset 500000, answer tokens erased.
- Overwrite conditions (`protocol.json` `overwrite_cells`): baseline (4 keys, distance 8, 1 rewrite); more keys (8 keys); farther query (distance 24); more rewrites (3 rewrites).
- MQAR conditions: baseline (length 64, 4 keys); more keys (64, 8); longer context (128, 4); longer and more keys (128, 8).
- Overwrite "newest-value recall" (`latest_mean`) with three rewrites: 74.6% default scalar, 77.6% default channel-wise, 52.5% mean-decay scalar, 56.6% mean-decay channel-wise (mean over seeds). Matches REPORT.md.
- Obsolete-value answers (`stale_mean`, prediction equals ANY prior value): 25.0 / 22.4 / 46.1 / 41.7% at three rewrites (same order).
- Error breakdown over 12 overwrite models (REPORT): 926 of 1,065 newest-value errors return the first replacement; 59 initial; 53 second replacement; 27 other.
- MQAR 8 keys, default scalar seed 44: 1091/2048 correct (53.27%); the rest of the group is much higher (REPORT). Check `models.csv` before quoting a per-seed number.
- Mean actual query distance (`mqar_actual_query_distances.csv`): about 18 tokens at length 64 with 4 keys, about 36 with length 128 and 8 keys (max 125). Length 128 alone is well tolerated. Do not say every query is remembered across 128 tokens.
- Conflict check: none found between REPORT.md and the CSVs above.

### Gate interventions: fixed-weight, post-hoc (no retraining)
Sources: `research_results/three_seed_gate_ablation_20261117/` (`models.csv`, `REPORT.md`), `research_results/stronger_gate_tests_20261117/` (`per_model_summary.csv`, `condition_means.csv`, `REPORT.md`), `research_results/three_seed_gate_diagnostics_20261117_final/REPORT.md`.
- Probe: 256 training-side examples (indices 256-511), answer-free. Channel-wise models only (12 = 2 tasks x 2 inits x 3 seeds). Weights unchanged; the intervention is removed afterwards.
- Shared-mean decay (alpha forced to its per-token channel mean, every layer): accuracy averaged over seeds, MQAR default 82.2%, MQAR mean-decay 69.6% (from 100%); overwrite default 83.7%, overwrite mean-decay 88.3% (from 99.5-99.6%).
- Static per-channel decay (no input dependence): MQAR stays 100%; overwrite default 99.5%, overwrite mean-decay 93.5%.
- Shuffled channel assignment (3 permutations): MQAR default 98.9%, MQAR mean-decay 80.5%, overwrite default 78.2%, overwrite mean-decay 87.7%.
- Effects are seed-specific. Example: MQAR mean-decay seed 43 drops to 30.6% under shared mean (layer-one mean loses 68.95 points); overwrite default seed 43 is unaffected (100%).
- Per-layer means: some models lose accuracy under one layer's intervention, most lose nothing. Do not claim a universal mechanism.
- Interpretation (REPORT): these show reliance of already-trained networks. NOT a fair comparison with separately trained scalar models; channel models have about 4.17% more parameters.

### Write-gate patches (three rewrites, 18 models: 12 local + 6 joint results-only)
Source: `research_results/write_gate_mechanism_20261117/patch_results.csv` (`tables.md` agrees).
- Unpatched newest-value accuracy at three rewrites, mean over 18 models: 64.5%.
- Change vs unpatched, mean over models, all layers (latest / unrelated / first-replacement rate; models improved / worse by 2 or more points):
  - beta_one_latest (write strength 1 on newest write): -0.80 / -0.85 / +0.93; 0 improved, 2 worse.
  - beta_zero_rewrite1 (write strength 0 on first replacement): +4.12 / -6.86 / -10.24; 6 improved, 1 worse.
  - key_align_latest (all earlier writes use the newest write's key): +16.21 / -18.40 / -23.65; 14 improved, 0 worse.
  - beta_copy_rewrite1: -0.02 / -0.30 / +0.04; 0 improved, 1 worse (not offered on the page).
  - control_beta_one_unrelated (write strength 1 on an unrelated write, control): +0.95 / -0.39 / -0.89; 2 improved, 0 worse (not offered on the page).
- Write strength at the newest write is flat (about 0.57-0.60) for correct and wrong answers (tables.md).
- Note: this is the discovery stage; key_align_latest is the only patch with a large latest-value gain, and it costs unrelated recall.

### Live try (12 local overwrite checkpoints)
- Models: `output/calibrated_recipe_study/matched-decay-v1/*/overwrite/seed_*/{default,calibrated}/{scalar,channel-wise}/latest_checkpoint.pt`. Mean-decay = folder `calibrated`. Discovery shared with `shortcut.local_checkpoints()`.
- Interventions offered (all restore the model afterwards):
  - none: as trained.
  - beta_one_latest, beta_zero_rewrite1, key_align_latest: `analyze_write_gate.apply_patch` on all four layers.
  - alpha_shared_mean: alpha replaced by its per-token mean over channels, all four layers (a no-op for scalar models, which have one decay value per head).
- Fresh examples: seed offset 5,000,000 (not used by any saved study; grep-checked against 300000, 400000, 500000, 700000, 900000, 1300000, 1400000, 1600000, 3000000).
- Before and after are computed on the same examples in one request.
- Scalar models: alpha_shared_mean changes nothing by construction. Say so on the page.

## Sections (order) and what each shows
1. **question**: "When the test gets harder, do the models keep their memory?" Visual: a single stress-result stat pair (overwrite newest-value recall at baseline vs three rewrites, from the summary) with animated bars.
2. **model**: the 12 live models as a grid (mean-decay vs default x scalar vs channel-wise x seed), with a note that a gate intervention edits one part of the trained model. Also a small diagram: decay gate alpha (one value per head for scalar, one per key coordinate for channel-wise) and write gate beta.
3. **code**: CodeWalkthrough over real source: `analyze_write_gate.py` `apply_patch` (the patch menu), `analyze_write_gate.py` `instrumented_forward` (where alpha and beta are read), `ablate_recipe_gates.py` `shared_mean_gates` (alpha forced), `stress_recipe_checkpoints.py` `OVERWRITE_CELLS` (the harder conditions).
4. **results**: (a) stress accuracy by condition: four overwrite cells and four MQAR cells per setting, animated bars (Panels); (b) gate intervention effects: AnimatedBars of mean accuracy drop under alpha forced, static, shuffled; write-gate patch deltas as animated bars (three rewrites, 18 models); each shows n and the number of models.
5. **inside**: one real example run through the live model with and without the chosen intervention: write cards, prediction, alpha per layer (mean and channel spread), beta, and state norm over tokens. Real tensors from the checkpoint.
6. **try**: pick a local model, an intervention, rewrites 1-4, distance 8-24, examples (1-64), example seed. Compare before and after accuracy on the same examples; show one example and its gate values before and after.

## Caveats to keep on the page
- Stress tests measure robustness of 16-epoch checkpoints. They do not show what a model trained on harder data could learn.
- The gate interventions are reliance tests. They are not a fair scalar-vs-channel comparison and do not explain learning speed.
- Three seeds: descriptive patterns, not significance claims.
- Joint models (6 of 18 in the patch table) are results-only on this machine.

## Status notes for the lead (not edited by this worker)
- Suggested `experiments.ts` status for `stress`: `live`.
- Glossary terms used: decay gate (α), write gate (β), recurrent state, decay granularity, scalar decay, channel-wise decay, decay initialization. "Mean-decay" is a decay initialization name; the glossary lists "calibration" only for the fitting procedure.
