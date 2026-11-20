# Content brief: Decay initialization (`decay-init`)

Question: does the way the decay gate (α) is initialised before training change how scalar decay and
channel-wise decay compare? Experiment status: **live** (12 local overwrite checkpoints). MQAR checkpoints
(also local) appear only in the results view, labelled as a separate task.

Numbers on the page come from saved files read by the API (CSV, JSON, checkpoints), never typed in.

## Sources and what each contributes
- `output/calibrated_recipe_study/matched-decay-v1/<session>/<family>/seed_<s>/<init>/<variant>/`
  - `latest_checkpoint.pt` (12 overwrite models: default and mean-decay x scalar and channel-wise x seeds 42, 43, 44;
    folder label `calibrated` = mean-decay in the UI). Keys: `model`, `completed_epochs`, `settings`.
  - `summary.json` (`final_development.answer_free.exact_query_accuracy`, `learning.accuracy_aulc_epochs1_16`).
  - `development_curve.json` (16 epochs; per epoch `answer_free.unrelated_association_retention`).
  - `calibration.json` (mean-decay only): per layer `target_half_life` (16, 32, 64, 128), `achieved_mean_decay`
    (length 1 for scalar, 128 for channel-wise), `decay_std`, `offsets`, `maximum_error`, `tokens_sha256`.
  - `settings.json`: 16 epochs (`epochs`), `scheduler_epochs` 64, learning rate 0.0021544, `length` 64, `width` 128,
    `vocabulary` 8192, `calibration_examples` 256.
- `research_results/three_seed_exploration_20261117/models.csv`: 24 runs (2 tasks x 2 inits x 2 granularities x 3 seeds);
- `research_results/completed_joint_decay_review_20261117/training_comparison.csv`: separate joint-initialization study (3 seeds); overwrite uses `retention_aulc`, MQAR uses `accuracy_aulc`. The Results tab shows these learning curves alongside default and mean-decay. Joint has no row in the fresh-example confirmation view.
  `primary_aulc` (overwrite: AULC of unrelated-association retention over epochs 1-16, 0-1 scale), `final_accuracy`
  (development). Reproduced exactly from `development_curve.json` (checked for all 12 overwrite runs).
- `research_results/three_seed_confirmation_20261117/models.csv`: fresh-data endpoint for all 24 runs: `accuracy`
  (overall, latest + unrelated answers, 512 examples per task and seed), `retention`, `latest`, `stale`.
  Frozen protocol: generator seed 400000 + training seed. Reported in `REPORT.md`.
- `research_results/three_seed_exploration_20261117/REPORT.md`, `research_results/three_seed_confirmation_20261117/REPORT.md`
  (cross-checked against the CSVs; see below).
- `research_results/three_seed_gate_diagnostics_20261117_final/REPORT.md` and `gate_diagnostics.json`,
  `condition_layer_means.json`: trained gate means and channel spread (post-hoc descriptive).
- `research_results/three_seed_gate_ablation_20261117/REPORT.md`: shared-mean intervention (post-hoc; not used on the page except as a caveat).
- `CALIBRATED_RECIPE_STUDY.md`: protocol for the study (full study = 64 epochs; checkpoints here are the 16-epoch exploration stage).
- `recipe_model.py` `RecipeGDN.__init__` (lines 63-69) and `RecipeGDN.decay` (lines 76-78): the default initialisation and the decay formula.
- `recipe_calibration.py` `fit_offset` and `calibrate`: the mean-decay initialisation (fits `dt_bias` only).
- `analyze_write_gate.py` `load(path, variant, seed)`: loads a checkpoint into `RecipeLM` in eval mode on CPU.
- `examiner_app/server/shortcut.py`: `local_checkpoints()`, `models()`, `_load()` (cached models, shared with the shortcut page).
- `decay_design_results.md`: DeltaNet vs KDA on TinyStories from another source; NOT used on this page (different task and setup).
- `GLOSSARY.md`: "decay initialization" = default, mean-decay, or joint. Use "mean-decay" in the UI; avoid "calibration" except for the fitting procedure.
- `sweep_review_20261115/ASSESSMENT.md`: scalar-only learning-rate sweeps; not used.

## Default initialisation (verified in code)
- `dt_bias` (per head and channel): inverse softplus of dt, dt log-uniform in [0.001, 0.1]. Random per model.
- `A_log` (per head and channel): log of rate, rate uniform in (0, 16).
- Decay: alpha = exp(-exp(A_log) * softplus(a_proj(x) + dt_bias)). Scalar: one value per token (gate width 1). Channel-wise: 128 values per token.
- Start values of the default model are NOT recoverable: `three_seed_gate_diagnostics` reports that the initial replay did not match saved parameter fingerprints for any of the 24 models. The page must not show a default start curve.

## Mean-decay initialisation (verified in code and calibration.json)
- `recipe_calibration.calibrate`: for each of the 4 recurrent layers, fits only `dt_bias` so the mean alpha over the first 256 training examples equals 2^(-1/h) for half-life h = 16, 32, 64, 128 (layers 0 to 3). Tolerance 1e-6; recorded maximum error about 1e-7.
- Calibration inputs verified: `diagnostic_training_samples/overwrite_seed<s>.npz` first 256 rows have SHA-256 prefix 518ffc01..., equal to `tokens_sha256` in calibration.json.
- Recorded start values: `calibration.json` `achieved_mean_decay` per layer (per channel for channel-wise). Start mean per layer = mean over channels.
- `gate_initialization.py` (`calibrated-geometric`, per-head geometric half-lives, class `GatedDeltaNetLM` in `model.py`) is a DIFFERENT method. It is not in the source hashes of matched-decay-v1 and does not produce these checkpoints. Do not cite it for these models.

## Training (verified in settings.json)
- Default/mean-decay overwrite task: 16 epochs (3136 accepted steps), scheduler horizon 64 epochs (so the learning rate has not decayed to zero). Same data order and seeds for all four conditions within a seed (shared data hashes per seed). Joint results come from a separate 16-epoch study.
- Parameters (`model.parameters()`, tied 8192 x 128 embedding counted once): scalar 1,585,416; channel-wise 1,651,456 (+4.2 %). Matches CALIBRATED_RECIPE_STUDY.md. (The raw state_dict sums are 2,633,992 and 2,700,032 because the tied embedding appears twice; do not use those.)

## Results: overwrite (primary for this page)
Learning speed = AULC of unrelated-association retention over epochs 1-16 (0-100 scale, reported as 0-1 in CSV).
| Condition | seeds 42 / 43 / 44 | mean (SD) |
|---|---|---|
| Scalar, default | 0.7904 / 0.9706 / 0.9690 | 0.9100 (0.1036) |
| Channel-wise, default | 0.7305 / 0.9992 / 0.7843 | 0.8380 (0.1422) |
| Scalar, mean-decay | 0.8578 / 0.8335 / 0.7756 | 0.8223 (0.0422) |
| Channel-wise, mean-decay | 0.6661 / 0.8462 / 0.9632 | 0.8251 (0.1496) |
Channel-wise minus scalar: default -7.2 points mean (per seed -6.0, +2.9, -18.5); mean-decay +0.3 points (per seed -19.2, +1.3, +18.8).
Interaction (mean-decay gap minus default gap) per seed: -13.2, -1.6, +37.2 points; mean +7.5. Sign changes across seeds: no stable effect.

Fresh-data final accuracy (overall, 512 examples per seed; confirmation models.csv):
| Condition | seeds 42 / 43 / 44 | mean |
|---|---|---|
| Scalar, default | 98.24 / 100.00 / 99.71 | 99.32 |
| Channel-wise, default | 99.61 / 100.00 / 96.78 | 98.80 |
| Scalar, mean-decay | 99.51 / 95.80 / 98.63 | 97.98 |
| Channel-wise, mean-decay | 92.97 / 99.51 / 99.61 | 97.36 |
All 12 overwrite models: 100 % updated-value accuracy except 3 wrong updated-value answers among 6144 (one stale). Most residual error is unrelated-association retention (fresh means: default scalar 98.63, default channel 97.66, mean-decay scalar 95.96, mean-decay channel 94.86).

Note: the learning-speed metric and the fresh final accuracy are different measurements. Do not pool them.

## Results: MQAR (secondary, results view only)
AULC (answer-free query accuracy, 0-100): default scalar 93.71, default channel 91.16, mean-decay scalar 93.63, mean-decay channel 93.73 (mean over seeds, exploration REPORT). Interaction positive in all three seeds (mean +2.65 points). Final accuracy ceiling 99.9 % for all conditions.

## Gate values (inside section, from checkpoints)
- `server/decay_init_build.py` reads the 12 overwrite checkpoints on CPU, runs the 256 calibration rows through each model, and records per layer: trained mean alpha per channel (or the single scalar value), mean over channels, and the across-channel spread of those means. Start values (mean-decay only) come from calibration.json. Output: `server/data/decay_init_gates.json`.
- Measured (`decay_init_gates.json`): mean-decay models start with each layer's average decay at its half-life target (start channel spread 0, every channel at target). After training the layer averages move (e.g. mean-decay channel-wise seed 43: 0.9288, 0.9364, 0.9806, 0.9863 against targets 0.9576, 0.9786, 0.9892, 0.9946) and channel-wise models spread across channels (channel spread 0.02 to 0.19 across layers and seeds). Channel spread here = standard deviation across channels of per-channel means; gate_diagnostics REPORT uses a different within-token definition (0.11 trained vs 0.003 recorded start). Descriptive, post-hoc numbers.
- Independent check done: recomputed the seed-43 mean-decay channel-wise layer means and spreads by a separate forward pass; they match the JSON exactly. One fresh example's decay values match the manual formula to 5e-5 (4-dp rounding).

## Try it
- `POST /api/decay-init/try`: one of the 12 local overwrite models, a fresh example (seed offset 3,000,000 as in the shortcut and layout-fix pages), real CPU forward pass. Returns tokens, roles, the answer for each query, and alpha per token for each layer.

## Conflicts and caveats (for the report and the page)
1. `CALIBRATED_RECIPE_STUDY.md` describes a 64-epoch full study; the local checkpoints are the 16-epoch exploration (settings.json epochs 16). The page says 16 epochs.
2. `research_results/three_seed_exploration_20261117/REPORT.md` says overwrite is "strongly seed-sensitive"; the page quotes that, not a general effect.
3. `matched-decay-v1/20261116T111608181_a2524782/session_status.json` says `failed_or_incomplete` (WinError 32 during publication) although the seed-42 pair checkpoints and summaries exist. The page treats the checkpoints as complete models (summary.json reports 3136 steps). Mention in report.
4. `decay_design_results.md` calls the scalar and channel-wise rows "the same model" and "parameter-matched"; the CALIBRATED doc and the checkpoints show a 4.2 % parameter difference. That document is not used.
5. Trained gate means do not stay at the starting means (gate_diagnostics section 1): the mean-decay start is matched by construction, not during training.
6. Joint initialization results are reported from a separate saved study; its checkpoints are not among the local interactive models. Learning-curve AULCs are shown in Results; fresh-example confirmation and live try-it remain default/mean-decay only.
