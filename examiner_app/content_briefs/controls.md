# Content brief: Negative controls (`controls`)

Question: can two external benchmarks (standard Zoology MQAR, and the KATA overwrite replication) tell the decay
designs apart, and do they show the block-first shortcut? They are negative controls: if they show no shortcut, the
shortcut comes from our generator, not from Gated DeltaNets in general.

Every number on the page is read from the saved files below (or from the live checkpoints) at request time.

## Sources (verified against the files; the files win)

- `research_results/zoology_layout_test_20261118/` (`results.json`, `cell_results.csv`, `protocol.json`, `REPORT.md`), made by `zoology_layout_test.py`.
- `research_results/kata_replication_eval_20261118_103946/` (`results.json`, `cell_results.csv`, `protocol.json`, `REPORT.md`), made by `eval_kata_replication.py`.
- `output/kata_replication/20261118T101702039_cbe1357b/` (`lr_*/summary.json`, `settings.json`, curves, logs). No `*.pt` file.
- `KATA_REPLICATION.md` (prose; agrees with the result files on parameters and the protocol).
- `kata_model.py` (`GDNNoGate`, `KataLM`), `kata_overwrite_data.py` (`generate`).

## Checkpoints on this machine

| Benchmark | Local checkpoints | Status on page |
|---|---|---|
| Zoology MQAR | 14 `latest_checkpoint.pt` files, all present (checked against `protocol.json`): 12 MQAR-trained runs in `output/calibrated_recipe_study/matched-decay-v1/*/mqar/seed_*/{default,calibrated}/{scalar,channel-wise}/` plus 2 published 64-token models in `published_mqar/{scalar,channel-wise}/` | **live** try-it |
| KATA | none (0 `.pt` files under `output/kata_replication`; its checkpoints are git-ignored and were not copied) | **results only**, models `<Unavailable>` |

Note for the lead: `experiments.ts` currently gives `controls` status `results`. With live Zoology checkpoints it should read `live`. Not edited here (file is off-limits to this worker).

Labels: the protocol's `calibrated` folder = `mean-decay` on the page (same mapping as `server/shortcut.py`). The published models are labelled `published`.

## Zoology MQAR layout test (verified)

- 14 models, 512 examples per cell, seed offset 1,600,000 (`zoology_layout_test.py` `SEED_OFFSET`). Wall 1265 s.
- Cells (`CELLS`): `zoology_baseline`, `shift_0`, `shift_1`, `shift_2`, `shift_4`, `shift_8`, `shift_16`, `near`, `far`, `memory_order`, `reverse_order`, `shift_8_far`.
- Verdicts (`results.json`):
  - Z1 standard layout at least 95%: 14 of 14.
  - Z2 position shortcut (shift 1, 2 or 4 at least 20 points below shift 0): **1 of 14** (`default/scalar/44`, shift_1 75.3%). Majority: no.
  - Z3 anchored errors (errors in shift_2 and shift_4 of the shortcut models that equal the position-anchored token): share **34.2%** (134 of 392). Threshold 50%. Not confirmed.
  - Z4 query layout (near, far, memory order, reverse order within 10 points of shift 0): 14 of 14 for each.
- Brief item 7 check: "1 of 14 fragile" is correct. "Not position-anchored" is a slight overstatement: 34% of that model's shift-2/4 errors are position-anchored (below the 50% bar), so "not confirmed as position-anchored" is the accurate wording.
- Design means (computed from `cell_results.csv`, 3 seeds each; published = 1 seed each): see the API `designs`.
  - shift_1 mean: default/scalar 91.7%, calibrated/scalar 98.8%, default/channel-wise 99.6%, calibrated/channel-wise 98.6%.
  - Every design is at least 96% on the standard layout and near/far. The separation on shifted layouts is driven mostly by one model.
- Reproduction: `make_cell` seeds are `example_seed + 1,600,000 + 1000 * cell_index`. With `example_seed` equal to a model's seed and 512 examples, the page reproduces the saved CSV exactly (verified by `controls_export.py`).
- Zoology has **no rewrites**: each key is written once. The block-first shortcut (which needs rewrites) cannot occur in this benchmark; the test is about positional shifts only.
- Error attribution (all models, `REPORT.md`): 3665 errors across all cells; most are another memory pair's value; shift_2 is the only cell with a sizeable position-anchored count (136).

## KATA overwrite replication (verified)

- One architecture: Gated DeltaNet, d_model 128, 2 heads, use_gate=False, **scalar decay only** (`kata_model.py` `GDNNoGate.__init__` passes `variant='scalar'`). No channel-wise arm exists, so KATA cannot tell the decay designs apart.
- Parameters: ours 1,167,876 (`output/.../lr_0.003/summary.json`); paper 1,167,878 (Table 2). Difference of 2 is recorded in `KATA_REPLICATION.md`.
- Learning rates 0.001, 0.003, 0.01; selected 0.003 (best average).
- Replication (R1), selected lr: ours K=64 99.5%, K=128 76.5%, K=256 28.4%, K=384 11.9%, average 54.1%. Paper: 99.8%, 85.3%, 38.4%, 21.7%, average 61.3%. Verdict: adequate = true.
- Shortcut test (R2): one-, three-, four-pass and partial-third-pass cells at K=4, 8, 16, 32 are 99.9% to 100% in all three runs. Two-pass cells run K=4..384 (the replication). **Shortcut in 0 of 3 runs.**
- R3 error identity: computed over all three runs' three-pass cells at K=4..32 (the script's `errors` counter, not only the selected run): 14 wrong answers, 13 from another key's value, 1 other; second-write share 0.0; not confirmed.
- R4 partial third pass: rewritten keys 100% for all lr. R5 single pass: 100% for all lr.
- Note: the replication cells are K=64..384 with T=6K (up to 2304 tokens). The shortcut cells in `cell_results.csv` are K=4..32 plus the K=64..384 rows; use the CSV rows as they are.

## Sections

1. **question**: "Can two outside benchmarks tell the decay designs apart, and do they show the shortcut?" Visual: two-column card per benchmark with stat callouts ("1 of 14 Zoology models show a position shortcut", "0 of 3 KATA runs show the rewrite shortcut").
2. **model**: the Zoology models (12 MQAR-trained decay designs + 2 published) and the KATA model (one scalar-decay design, 3 learning rates). Grid with local / unavailable badges.
3. **code**: `CodeWalkthrough`: `zoology_layout_test.py` `make_cell` (where the shift and query placement happen); `recipe_data.py` `mqar` (standard generator); `kata_overwrite_data.py` `generate` (two passes then queries); `kata_model.py` `GDNNoGate` (scalar decay).
4. **results**: Panels. (a) Zoology accuracy vs shift for the four designs (AnimatedLines, mean over seeds). (b) Zoology per-model accuracy on shift_1 (AnimatedBars). (c) KATA replication vs paper (AnimatedLines over K). (d) KATA shortcut test (bars by pass count at selected lr).
5. **inside**: where answers come from (memory slot histogram). Zoology fragile model (`default/scalar/44`) on shift_1 and shift_0 vs the block-first shortcut model (`default/scalar/44`, 3 rewrites). Data: `server/data/controls_inside.json` generated by `server/controls_export.py` from real checkpoints.
6. **try**: live Zoology try-it (layout, model, example count, example seed) using the same shortcut-style summary. KATA: `<Unavailable>` for its models plus a browsable table of the saved results.

## Writing constraints
- Protocol identifiers (Z1..Z4, R1..R5, `shift_1`, `calibrated`, `lr_0.003`) stay out of visible prose; use plain names ("one blank token before the table", "mean-decay", "learning rate 0.003"). They may appear in More detail.
- Design names on the page: default, mean-decay; scalar, channel-wise.
