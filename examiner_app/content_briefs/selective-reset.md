# Content brief: Selective reset pilot (`selective-reset`)

Can a model learn to clear only part of its memory when it is told to? A small, four-run pilot. Negative result: the models did not learn the task.
Every number on the page comes from the API, which reads the saved result files (`output/selective_reset/*.json`) or runs the one local checkpoint.

## Sources
- Task, model and scoring: `selective_reset_pilot.py` (commit `4df3392`, the only commit touching it; `c237191` added the outputs).
- Saved results: `output/selective_reset/gated_Deltanet_42.json`, `gated_deltanet_43.json`, `gated_deltanet_scalar_42.json`, `gated_deltanet_scalar_43.json`.
- Checkpoint: `output/selective_reset/gated_deltanet_42.zip` (a torch checkpoint in zip form; the pilot wrote `gated_deltanet_42.pt`). The other three runs have no checkpoint on this machine.
- Model code: `third_party/dualgoose/dualgoose/model.py` (`build_tiny_denoiser`), `third_party/dualgoose/dualgoose/mixers.py` (`DeltaNetRefLayer`, decay `"channel"` or `"scalar"`), `third_party/dualgoose/UPSTREAM.md`.
- No write-up exists: `NOTES.md`, `lessons/`, `research_notes/` do not mention the pilot. `output/pytest_two_workers` is unrelated and ignored.

## Story (verified facts)
1. **Task** (`selective_reset_pilot.py`): a sequence of 64 tokens. A "workspace" holds key-value pairs: workspace red (4) or blue (5). The model reads 16 writes, shuffled into a 48-token prefix. Then at one event position it reads either NOOP (token 1) or a reset of the target workspace (token 2 = reset red, token 3 = reset blue). Then a survivor update (a write to the other workspace) and a final query.
2. **Memory load** = number of keys in the target workspace: `LOADS = (1, 2, 4, 8)`. The other workspace holds the remaining keys.
3. **Four queries** (`QUERIES`): `target` (a key in the target workspace; after a reset the right answer is EMPTY, otherwise the old value), `survivor` (a key in the other workspace; the right answer is its value, unaffected by the reset), `rewrite` (a key written again after the event; the right answer is the new value), `absent` (a key never written; EMPTY). `EMPTY = 70` is a reserved answer token.
4. **Values**: values are tokens 38..69, i.e. 32 possible values. A random guess scores about 1/32 = 3.1 percent (derived from the code: `values = rng.integers(38, 70)`).
5. **Reset and noop branches**: each episode is run with reset and with NOOP (`branch(ep, reset, kind)`). The pair differs in one token at the event. Test size: 8 batches of 256 episodes per load at evaluation, fixed evaluation seed 908731.
6. **Labels are audited** (`audit()`): for 128 episodes per load a simple memory simulator checks each label, and the branch pair must differ in exactly one token.
7. **Model change: none.** `selective_reset_pilot.py` builds the stock vendored denoiser (`build_tiny_denoiser(..., model_type=family, d_model=32, n_layers=2, n_heads=1, short_conv=4, direction="forward", merge="static")`). The reset is only a token in the input. `dualgoose/*.py` contains no reset logic (grep: no match). The only architectural difference between the two families is the decay granularity: `gated_deltanet` = channel-wise decay (the published cell), `gated_deltanet_scalar` = scalar decay (one decay value per token).
8. **Training** (defaults, used for all four runs): 3000 steps, batch 256, AdamW lr 1e-3 with cosine decay to 1e-4, grad clip 1.0, memory load drawn per step from LOADS, reset and query kind drawn per example. Training ran on Kaggle GPUs (JSON `settings.device` cuda:0 / cuda:1; `training_seconds` about 340 to 356 s).
9. **Parameters**: channel-wise 49,732; scalar 45,640 (from JSON).
10. **Result, joint test**: a run is right on both the target and the survivor query only if both are correct. Joint accuracy is 0.0 to 0.1 percent in every run at every load (max 0.00098). No run passes.
11. **Result, per query**: reset target is 100 percent in all four runs at all loads. Noop target is 0 to 1.6 percent. Survivor (both reset and noop) is 0 to 0.1 percent everywhere, i.e. below the 3.1 percent guess rate. Rewrite is 67 to 100 percent (lowest: scalar seed 42, about 68 to 72 percent). Absent is about 100 percent.
12. **Reading** (inference from the saved files): the pattern fits a model that answers EMPTY for target queries whatever happened, and that cannot recall older writes (noop survivor also fails). Only the newest write (rewrite) is recalled. Checked on the local checkpoint (inside slide): on 256 fixed-seed episodes per cell it answers EMPTY to 100 percent of no-op target queries at every load, and to 99 to 100 percent of survivor queries (measured via /api/selective-reset/profile).

## Live status
- Checkpoints present: 1 of 4 (channel-wise, seed 42, `gated_deltanet_42.zip`). Scalar 42/43 and channel-wise 43 are absent.
- Live check (this brief's author): the checkpoint loads with `torch.load(..., weights_only=True)` into `build_tiny_denoiser(model_type="gated_deltanet", ...)` with no missing or unexpected keys (49,732 parameters), and `selective_reset_pilot.evaluate` on CPU reproduces the saved accuracies exactly (maximum absolute difference 0.0 over all loads and metrics).

## Sections (six slides; keep the text short)
1. **question**: "Can a model learn to clear part of its memory when told to?" Visual: a two-workspace whiteboard: a reset clears the red column and leaves blue.
2. **model**: "Same small model; the reset is just one more input token." Visual: input tokens -> two layers -> answer; a two-item Panels for channel-wise vs scalar decay. No new update rule.
3. **code**: CodeWalkthrough over `selective_reset_pilot.py`: 1 the task (`episodes`), 2 the reset branch (`branch`), 3 the scoring (EMPTY after reset), 4 the model (`train`), 5 the evaluation (`evaluate`).
4. **results**: Panels. (a) per query, averaged over 4 runs and 4 loads, animated bars; (b) per memory load, lines (means over 4 runs) for reset target, survivor, rewrite.
5. **inside**: one live checkpoint over fixed-seed episodes: what it answers for each query type, as bars (expected / EMPTY / stale value / survivor's value / other). Not available without the checkpoint.
6. **try**: pick load, reset or NOOP, query, seed: the token sequence and the checkpoint's answer, with the right answer. Unavailable for runs without a checkpoint.

## Vocabulary for the page
"memory" for the workspace contents; "reset" (clear the target workspace); "survivor" (the other workspace's key); "decay granularity", "scalar decay", "channel-wise decay", "arm", "run", "memory load". Never say "gating" for the scalar/channel-wise difference. Do not show `gated_deltanet_scalar`, `reset_joint`, or other keys in visible prose; use them only in More detail.
