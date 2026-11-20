# Content brief: The hidden shortcut (`shortcut`)

How the block-first shortcut was discovered and then confirmed on fresh data (the fix is the Layout-fix page).
Numbers on the page must come from the API (which reads saved result files or runs local checkpoints), never typed in.
Do NOT use numbers from research_notes prose: four of its figures disagree with the result files; the files win.

## Story (verified facts)
1. **First sign** (memory stress test, 17 Nov; `research_results/memory_stress_tests_20261117/`): with three rewrites, newest-value recall fell to 52–78% depending on initialization and decay granularity, although training-layout accuracy was ~100%. Most wrong answers were the *first replacement* (926 of 1,065 errors over 12 models).
2. **Hypotheses tested** (`analyze_write_gate.py`, `research_results/write_gate_mechanism_20261117/`):
   - *Write-gate collapse?* No: write strength β at each write is flat (≈0.57–0.60) for right and wrong answers; forcing β=1 on the latest write did not help (−0.8 points).
   - *Key confusion?* Partly: making all earlier writes use exactly the latest write's key raised latest-value accuracy by +16.2 points (14 models up, 0 down), but hurt unrelated recall (−18.4).
   - Splitting examples by layout revealed the failures concentrate in the **block-first** layout (all of the queried key's writes come before any other key's write).
3. **Pre-registered confirmation** (`confirm_block_layout.py`, `research_results/block_layout_confirmation_20261117/`; hypotheses written to protocol.json before evaluation; fresh data at seed offset 1,300,000; 256 examples per layout; 18 models incl. 6 joint not on this machine): H1 shortcut in 16 of 18 models; H2 97.1% of block-first errors are the first replacement; H3 all 18 models ≥95% on the training layout (so ordinary validation cannot see it); H4 near-miss layouts (one unrelated write first; run broken before the latest; initial then a run) behave normally in 13 of 16; H5 the deficit persists when 4 or 8 filler tokens separate memory and query (14/16, 15/16).
4. **Which rule do they follow?** Six candidate rules were scored (`confirm_block_layout.rule_answers`): latest, initial, second_occurrence, memory_slot_1, opening_run_second ("if the key's writes form an unbroken run from the start, answer the second write"), final_run_second. Shortcut models agree with opening_run_second 89.6–100%. In training (one rewrite) that rule always equals the latest value, which is why it was learned.
5. **Position evidence:** with no gap, errors return memory slot 1; with a filler gap, errors return the second-to-last memory slot (an unrelated key's value), i.e. position-based retrieval.
6. **Granularity does not matter:** shortcut in 8 of 9 scalar and 8 of 9 channel-wise models.
7. **Negative controls:** standard Zoology MQAR (1 of 14 models fragile; not confirmed as position-anchored: 34% of its shift-2/4 errors, below the 50% bar) and the KATA replication (0 of 3) show no comparable shortcut → it comes from a fixed regularity in our generator, not from Gated DeltaNets in general.

Local checkpoints: 12 overwrite models (default and mean-decay ["calibrated" in folder names] × scalar/channel-wise × seeds 42–44) under `output/calibrated_recipe_study/matched-decay-v1/*/overwrite/seed_*/{default,calibrated}/{scalar,channel-wise}/latest_checkpoint.pt`. Joint models: results only ("Models unavailable").

## Sections
1. **question**: "Can a model that looks perfect be using the wrong rule?" Visual: a two-column "what validation sees vs what a block-first test sees" for one live model (training-layout accuracy vs block-first 3-rewrite accuracy), big animated numbers.
2. **model**: same ~1.6M GDN; which 18 models were tested (grid of initialization × granularity × seed with local/unavailable badges).
3. **code**: CodeWalkthrough: `confirm_block_layout.py` `special_patterns` (the named layouts); `sample_layouts` (rejection sampling of fresh layouts); `rule_answers` (the six rules); `analyze_write_gate.py` `apply_patch` (the β and key patches).
4. **results**: (a) hypothesis scoreboard H1–H5 with the pre-registered threshold and the observed value (from results.json), animated pass/fail; (b) per-model block-first vs general accuracy at 3 rewrites (from cell_results.csv) as a dumbbell chart; (c) the discovery-stage patch results as bars.
5. **inside** (decided 2026-11-19, the error map): three steps. (1) One example: memory as numbered cards in write order, the question, a "correct answer" badge and an arrow to the model's pick, with previous/next over six examples. (2) All 64 examples: bars above the seven cards showing how many answers came from each card. (3) Move the question further away: 0 / 4 / 8 extra blank tokens, same bars. A compare toggle switches between the shortcut model (seed 44) and a healthy model (seed 43). The rule-detective and model-surgery variants were removed from the page (not archived on any branch yet).
6. **try**: choose model (12 local), layout, rewrites 1–4, distance 8–24, example seed; show the example, the answer, which write it came from, which rules the answer agrees with.
