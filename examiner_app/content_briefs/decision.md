# Content brief: Decision fine-tuning (`decision`)

Question: does a short extra round of training on a decision task change which answer the models pick?
Status: `results` (saved results only). No fine-tuned checkpoint was saved, so nothing can be rerun here.

## What the saved results are (verified)
Two saved feasibility runs, both on a Tesla T4 (torch 2.11.0+cu128), both from `decision_feasibility.py`:
- `result.json` (repo root): run "short": 5 steps, batch 2, max length 256, 48 training and 16 validation cases.
- `result (1).json` (repo root): run "longer": 50 steps, batch 4, max length 512, 48 training and 16 validation cases.
- Both: workflow `customer_service`, question `action`, labels `answer_directly, close_no_action, escalate_to_human, execute_refund, request_information`, dataset fingerprint `204dc064fcdddd1f`, split = training split only, 80/20 internal split (240 train / 60 held-out available), benchmark test split never loaded.
- Per variant (scalar, channel-wise): first and last training loss, mean validation soft loss, steady step median seconds, peak allocated and reserved GiB.
  - Short run: scalar 1.958 -> 1.203, val 1.141, 3.12 s/step, 0.98 GiB; channel-wise 1.796 -> 1.163, val 1.146, 2.85 s/step, 1.00 GiB.
  - Longer run: scalar 1.940 -> 1.217, val 1.153, 5.77 s/step, 2.82 GiB; channel-wise 1.651 -> 1.175, val 1.178, 5.61 s/step, 2.86 GiB.
- These numbers are read by the API from the files at request time. They are not typed into the UI.

## What the saved results do NOT show
- No pick-before vs pick-after comparison: no accuracy, no argmax answers, no per-case predictions were saved. The question is not answered.
- No pre-training validation loss was saved, so "validation soft loss went down" cannot be claimed. Validation loss is flat-to-up (1.141 to 1.178 across runs) and has no baseline.
- The two runs used different settings (steps, batch, length), so they are not a controlled comparison.
- The only variant difference is scalar vs channel-wise decay, and both are near-identical on the numbers above.

## The decision task (plain words, from the code)
- A customer-service case (a free-text "state") and one question with five fixed options. The model must choose one option.
- Options (labels from the saved file): answer directly, close with no action, escalate to a human, execute a refund, request more information.
- Training target: a soft probability over the five options from the dataset (gold probabilities, normalised to sum to 1). Loss = soft-label cross-entropy.
- Input text (from `decision_feasibility.py` `prepare_examples`): `State: <state>` then `Question: <instructions>`, `Options:` with one line per label, then `Decision:`. Long states are cut so the whole input fits the max length; the question and options are always kept.
- Model: the 30M GatedDeltaNet checkpoint (`PRODUCTION_MODELS["30M"]`, d_model 320, 10 layers), with its output head removed and a new linear 5-way head on the last token's hidden state.

## Source discrepancies (report these)
- `DECISION_FEASIBILITY.md` says the report is written to `output/decision_feasibility/result.json`. No such file exists in the repo; the saved reports are at the repo root as `result.json` and `result (1).json`.
- `DECISION_FEASIBILITY.md` says the pilot uses scalar seed 43 and channel-wise seed 42. Both are present locally. The saved reports only record `/tmp` checkpoint paths, so the exact file used is not recorded.
- `decision_research_config.py` and `decision_research.py` define a full study (candidate learning rates 1e-5 and 2e-5, 3 epochs, option-scoring head). No outputs for it exist under `output/` or `research_results/`. It was not run.
- The fixed-size head in the pilot differs from the option-scoring head in `decision_model.py`. The pilot is not the final model.

## Inside (what could and could not be run)
- `decision_data.py` is the real generator, but it loads `LocalLLaMA/typed-decisions` from Hugging Face. The dataset is not in the local HF cache, so the generator cannot run here without downloading it. Not downloaded (needs the user's go-ahead).
- So the inside slide is a schematic of the pipeline as coded (text -> GPT-2 tokens -> 30M backbone -> 5-way head -> soft target), with the real option labels. No example data is shown.

## Checkpoints
- Local 30M base checkpoints exist: scalar seeds 42, 43, 44 and channel-wise seeds 42, 43 (listed by the API as `base_checkpoints_local`). They are pre-extra-round models. The fine-tuned heads were never saved, so nothing fine-tuned can be run.
- Therefore `try` = `<Unavailable>` plus a browsable view of the saved numbers.

## Sections (six slides)
1. **question**: Takeaway "Does a short extra round of training change which answer a model picks?" Visual: the case -> five options diagram (labels from API).
2. **model**: Takeaway "The same 30M model, with a new five-way choice head." Visual: diagram of base checkpoint -> extra round -> head, scalar vs channel-wise.
3. **code**: CodeWalkthrough over `decision_feasibility.py`: `prepare_examples` (prompt and soft target), `DecisionPilot.forward` (head on last token), `load_model` (the one line where decay granularity is set), `run_variant` (training step loss).
4. **results**: Panels: training loss (start vs end, both runs), held-out soft loss (both runs, no baseline), cost (step time, peak GiB). Takeaway: "The extra round trains, but no saved result shows the answers changed."
5. **inside**: schematic pipeline with real labels. Takeaway: "The model reads the case as text, then a new head scores each action."
6. **try**: `<Unavailable>` ("no fine-tuned models saved on this machine") plus a table of the saved numbers. Takeaway: "This machine cannot run the fine-tuned models, so the saved numbers are all there is."

## Hero headline
Do not add a headline stat (the only numbers are loss values, not a pick change).
