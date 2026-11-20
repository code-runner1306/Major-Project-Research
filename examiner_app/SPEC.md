# examiner_app: build spec (source of truth for every worker)

A localhost web app that lets a non-specialist examiner explore the Gated DeltaNet decay-granularity research.
It runs the repository's **real trained checkpoints** through a Python API. Read `../GLOSSARY.md` for vocabulary
and use its terms exactly (say "decay granularity", "scalar decay", "channel-wise decay", "decay gate (α)", "write gate (β)",
"recurrent state", "arm", "run", "experiment"; never say "gating" for the scalar/channel-wise difference).

## Ground rules
- Never copy or modify research code. The server **imports** repo modules (`recipe_model`, `confirm_block_layout`, `analyze_write_gate`, `layout_fix_data`, `retrieval`, `config`) by putting the repo root on `sys.path`.
- Never retrain anything, never write into `output/` or `research_results/`.
- Results shown in the UI come from saved artifacts (listed below). The "try it" endpoints run checkpoints live on CPU.
- Python: the repo venv, `../.venv/Scripts/python.exe` (Python 3.11, torch 2.11 CPU). Web: Vite + React 19 + TypeScript, `motion` (import from `motion/react`) and `d3` are installed in `web/`.
- Audience: a technical person who knows nothing about this field. Plain English first; equations only inside a "Show the maths" toggle.

## Layout
```
examiner_app/
  SPEC.md            this file
  start.ps1          starts server (port 8765) and web (port 5180), opens browser
  server/
    app.py           FastAPI app (run: python -m uvicorn app:app --port 8765 from server/)
    checkpoints.py   checkpoint discovery + cached loading
    internals.py     forward pass with captured tensors (gates, states, predictions)
    codeview.py      safe source extraction for the code walkthrough
    data/layout_fix_export.json   cached fresh evaluation (see below); do not regenerate
    legacy_prototype/export_data.py   reference implementation of tracing/eval; read it, reuse its logic
    tests/test_api.py
  web/               Vite app (port 5180, proxies /api -> http://127.0.0.1:8765)
```

## Saved artifacts the server reads
| Artifact | Path (repo-relative) |
|---|---|
| Layout-fix checkpoints (30) | `output/layout_fix_study/layout-fix-v1/*/<arm>/seed_<seed>/<variant>/latest_checkpoint.pt` |
| Per-run training curve | same folder, `development_curve.json` (list per epoch: `epoch`, `train_loss`, `answer_free.exact_query_accuracy`, `arm_answer_free.exact_query_accuracy`) |
| Per-run summary | same folder, `summary.json` (`final_development.answer_free.exact_query_accuracy`) |
| Saved A4 confirmation | `research_results/layout_fix_a4_confirmation_20261118/cell_results.csv` |
| Cached fresh evaluation of all 30 runs | `examiner_app/server/data/layout_fix_export.json` (keys: `meta`, `fresh_results`, `overlap_audit`, `showcase`, `delta_rule`, `gates`, `curves`, `saved_a4_report`, `paper`) |

Arms: `A0_control, A1_varied_distance, A2_no_block_first, A3_mixed_rewrites, A4_distance_and_rewrites`. Variants: `scalar, channel-wise`. Seeds: `42, 43, 44`.
Load a checkpoint with `analyze_write_gate.load(path, variant, seed)` (returns eval-mode `RecipeLM` on CPU).

## HTTP API (all JSON)
`GET /api/health` -> `{"ok": true, "torch": "<version>", "checkpoints": <int count found>}`

`GET /api/layout-fix/summary` -> the cached export, unchanged except `showcase`/`delta_rule`/`gates` may be omitted. Add `"provenance": {"file": "...", "generated": meta.generated}`.

`GET /api/runs` -> `[{"arm","variant","seed","path","final_dev_accuracy"}]` for all 30 layout-fix runs.

`POST /api/layout-fix/try` body `{"arm": str, "variant": "scalar"|"channel-wise", "seed": 42|43|44, "layout": "block_first"|"general_nonblock"|"block_after_one_unrelated"|"run_broken_before_latest"|"initial_then_run", "rewrites": 1..4, "distance": 8..24, "example_seed": int >= 0}`
- Generate ONE fresh overwrite example: set `confirm_block_layout.SEED_OFFSET = 3_000_000` once at import, then
  `chosen, info = confirm_block_layout.sample_layouts(example_seed, 0, 4, distance, rewrites, [layout], 1)`; record = `chosen[layout][0]`.
  `layout` must be `general_nonblock` or a key of `confirm_block_layout.special_patterns(rewrites)`; otherwise 422.
- Build the answer-free input exactly like `confirm_block_layout.to_cell` (gold answers after each query replaced by token 1).
- Run the model with internals (see `internals.py` below). Response:
```json
{"example": {"tokens": [int], "roles": [str], "pattern": "TTUUU", "write_values": [int], "queries": [
   {"position": int, "kind": "latest"|"unrelated", "gold": int, "prediction": int,
    "which_write": "latest"|"initial"|"first_rewrite"|"second_rewrite"|"third_rewrite"|"other",
    "top5": [{"token": int, "prob": float}]}]},
 "internals": {"layers": [{"layer": 1..4, "alpha": [[float]] /* [T][1 or 128], 4 dp */, "beta": [float] /* [T] */,
                            "state_norm": [float] /* Frobenius norm of state after each token */}],
               "state_crops": [[[float]]] /* layer 4, 32x32 top-left crop of state after EVERY token, 3 dp: [T][32][32] */,
               "write_positions": [int], "latest_write_position": int},
 "model": {"arm","variant","seed","parameters": int, "checkpoint": str}}
```
- roles per position: `filler, key, value, target_key, target_value, marker, query_latest, query_unrelated` (same rule as `describe_example` in the legacy export).
- Must respond in under 3 s on CPU. Cache loaded models (`functools.lru_cache`, max 12).

`GET /api/code?path=<repo-relative .py>&symbol=<Name or Class.method>` -> `{"path","symbol","start_line","end_line","source"}`.
Use `ast` to find the def/class; allow only `.py` files that exist under the repo root, reject `..` and absolute paths (400). Optional `&context=n` adds n lines before/after.

`GET /api/datasets` -> `[{"id","name","source","origin": "port"|"ours"|"upstream"|"external","live": bool,"params": {name: {"type":"int"|"choice","min"?,"max"?,"choices"?,"default"}}}]` for the 11 generators in `content_briefs/dataset.md`.

`POST /api/datasets/generate` body `{"generator": id, "seed": int >= 0, "params": {...}}` -> `{"generator","tokens": [int],"roles": [str],"scored": [{"position": int,"answer": int,"kind": str}],"notes": [str] /* plain-English reading, one sentence per fact */,"meta": {...generator-specific, e.g. pattern, write_values, layout},"reserved_tokens": {"0": "query marker", ...}}`.
Optional response fields (only some generators): `token_text`: [str] (one decoded string per token, same length as `tokens`), `text`: str (story text, first 800 characters).
Roles vocabulary: `filler, key, value, target_key, target_value, marker, query_latest, query_unrelated, query_key, answer_slot, noise, decoy_key, decoy_value, mask, text, eos`. 422 for invalid params, 409 with a message naming the missing file for `tinystories` when the local parquet or tokenizer cache is absent.
`tinystories` (offline, validation split of roneneldan/TinyStories, GPT-2 tiktoken): story = `seed % 21990` (parquet row); tokens = the story's GPT-2 ids followed by EOT (50256, role `eos`), cut to `window` (choice 64, 128, 256, 512; default 128). `scored` is always `[]`. `meta`: `story_index` (0-based row), `num_story_tokens` (excludes EOT), `window`, `split`, `source`, `file`, `tokenizer`.
Must be index-stable where the generator allows it (overwrite, arms, layouts, kata, zoology cells: example i depends only on seed and i); for Zoology MQAR generate `mqar(256, …, seed)` and return row 0, and say so in `meta.note`. Each call under 1 s after warm-up; import generator modules lazily and cache.

`GET /api/shortcut/summary` -> `{"hypotheses": {H1..H5: {"statement","threshold" (prose), "threshold_value" (number in [0,1]: H1 0.5, H2 0.80, H3 (of-1)/of, H4 0.5, H5 0.5), "observed" (raw object from results.json, kept as is), "observed_text" (one plain-English sentence computed from observed), "observed_value" (number in [0,1]: H1 models_showing/of, H2 first-replacement share of errors, H3 models_at_least_95/of, H4 min share over layouts, H5 min share over distances), "confirmed"}}, "per_model": [{"init","variant","seed","local": bool,"block_first_r3","general_r3","block_first_r1","shortcut": bool}], "rule_agreement": [{"init","variant","seed", rule: share...}], "patches": [{"patch","latest_delta","unrelated_delta","first_rewrite_delta","models_up","models_down"}], "stress_first_sign": {...}, "provenance": {files}}` built from `research_results/block_layout_confirmation_20261117/{results.json,cell_results.csv,protocol.json,REPORT.md}`, `research_results/write_gate_mechanism_20261117/` and `research_results/memory_stress_tests_20261117/`. Read files; never hard-code numbers.

`GET /api/shortcut/models` -> the 12 local overwrite models `[{"init": "default"|"mean-decay","variant","seed","checkpoint"}]` (folder label `calibrated` = "mean-decay").

`POST /api/shortcut/evaluate` body `{"init","variant","seed","layout","rewrites": 1..4,"distance": 8..24,"examples": 1..64,"example_seed": int,"patch": "none"|"beta_one_latest"|"key_align_latest"|"beta_zero_rewrite1", "show_examples": 1..8 (optional, default 3)}` -> `{"accuracy_latest","accuracy_unrelated","first_rewrite_rate","answer_sources": {latest, initial, first_rewrite, ...},"rule_agreement": {rule: share},"slot_histogram": [{"slot": int,"is_target": bool,"count": int,"tokens_before_query": int}],"attribution": [{"layer","target_share_correct","target_share_wrong"}] | null,"pick_histogram": [{"slot": 1-based, "count", "is_target", "is_latest_target"} for each memory slot, then {"slot": null, "count": n} for answers matching no slot],
"examples": [first show_examples examples with tokens, roles, prediction, which_write, rules_matched, "writes": [{"slot": 1-based, "key", "value", "is_target" (write of the queried key), "is_latest_target"}] in write order, "question_key", "gold", "prediction", "picked_slot" (1-based slot holding the prediction, most recent; null if none), "gold_slot", "tokens_between_last_write_and_question" (blank tokens between the queried key's latest write and the question marker)]}`.
`pick_histogram` counts all `examples` latest-value answers (correct and wrong) by the slot they came from. Its slot flags are majority votes over examples (identical for fixed layouts).
Uses `confirm_block_layout.sample_layouts` (SEED_OFFSET 3,000,000 as elsewhere), `to_cell`, `evaluate`, `rule_answers`, and `analyze_write_gate` (`load`, `instrument`, `predict`, `apply_patch`, `restore`, `attribution`) for patches/attribution. Patched runs must restore the model afterwards (shared cache). Must finish < 4 s for 64 examples on CPU.

`GET /api/code/lines?path=<...>&start=<int>&end=<int>` -> same shape, for non-symbol snippets (max 80 lines).

## internals.py
Port `mixer_inputs`, `mixer_tensors` and the serial recurrence loop from `legacy_prototype/export_data.py`.
For each of the 4 layers compute q,k,v,alpha,beta from the captured mixer input, then iterate the recurrence token by
token (`S = S * alpha[t] (per key column); S += beta[t] * (v[t] - S@k[t]) k[t]^T`) to record state norms (all layers)
and layer-4 crops. Assert the final state equals `recipe_model.serial_rectangular` output within 1e-5.
Predictions come from `model.scored_logits(tokens, mask)` softmax.

## Writing rules (decided 2026-11-19 after user feedback: "way too much text, people get bored")
Show, don't tell. The visual carries the story; text only labels it.
- Each section shows: one bold **takeaway** (≤ 15 words) and at most two short sentences (≤ 35 words total). Everything else goes in a collapsed `<MoreDetail>` ("More detail") disclosure.
- Labels live inside the visual (on cards, axes, arrows), not in paragraphs above it.
- `WhatThisShows` captions: one sentence, ≤ 20 words.
- Hero: one-liner ≤ 20 words; analogy ≤ 20 words.
- Big numbers are stat callouts (`<Stat value label/>`), not buried in sentences.
- Never show protocol identifiers (`main_r3`, `general_nonblock`, `block_first`, `calibrated`, H1…H5 jargon) in visible text; translate ("3 rewrites", "mixed layout", "block-first layout", "mean-decay"). They may appear inside More detail.
- Prefer one concrete example (real tokens) before any aggregate chart.

## Slides, not scrolling (decided 2026-11-19 after user feedback)
Each experiment page is a deck of six full-height slides (question, model, code, results, inside, try), one visible at a time.
- The chapter bar switches slides; number keys 1–6 jump straight to a slide (ignored while typing in a text box, select or editable field); Previous / Next buttons and a "3 / 6" indicator sit at the bottom of every slide; keys PageDown/PageUp (and `]`/`[`) move between slides and ← / → also step between slides unless focus is on an in-slide tab row, slider, radio group or text field (those keep their own arrow behaviour). URL `#/<experiment>/<section>` deep-links a slide.
- Every slide must fit within the viewport at 1280×720 (sidebar visible) without page scrolling. If a section has more than one visual, use `<Panels>` (small internal tabs, one visual at a time) instead of stacking them. `MoreDetail` opens as an overlay/drawer inside the slide, scrolling internally if long.
- The hero (title, one-liner, analogy, headline stat) is a compact strip on the first slide only.
- Slide changes animate (short horizontal slide/fade, reduced-motion aware).

## Web app

### Routing and shell
Hash routing `#/<experimentId>`. Persistent left sidebar listing experiments in this order, grouped:
- Start: `intro` Introduction, `dataset` The datasets
- Experiments: `layout-fix` Layout fix A0–A4, `shortcut` The hidden shortcut, `decay-init` Decay initialization,
  `stress` Stress tests and gate interventions, `length` Longer sequences and filler decay, `anatomy` Anatomy harness,
  `controls` Negative controls (Zoology, KATA), `tinystories` TinyStories language models, `cost` Cost and runtime
- Appendix (collapsed by default): `selective-reset` Selective reset pilot, `attention-control` Attention control, `decision` Decision fine-tuning

Each sidebar item shows a small status chip: `live` (checkpoints run here), `results` (saved results only), `unavailable`.
Pages not yet written render a placeholder with title + one-liner + "Page in progress".

### Page model (`web/src/types.ts`, already written: do not change without updating this spec)
An experiment page is `ExperimentPage({ def, sections })` where `sections` is an ordered list of
`{ id: SectionId, title, explain: ReactNode, visual: ReactNode }`, SectionId in
`question | model | code | results | inside | try`. Layout components decide where `explain` and `visual` go.

### Layout (decided 2026-11-19)
Chapters layout is the only page layout: `web/src/layouts/LayoutChapters.tsx`, rendered by `ExperimentPage` (props unchanged).
Full-width hero (title, one-liner, analogy, a headline stat slot), then a sticky horizontal chapter bar
(Question · Model · Code · Results · Inside · Try it), then each section as a full-width band with alternating
background; explain above, visual full width below. Scroll-spy and smooth scrolling live in `web/src/layouts/scroll.ts`.
Layouts A (Document) and B (Scrollytelling) are archived on branch `prototype/examiner-layouts`; they are not in `main`.

### Responsive breakpoints
Enforced by `@media` rules in `web/src/styles.css`; the drawer reads the same `(max-width: 899px)` query in `web/src/shell/useMediaQuery.ts`.
- **1200px and up:** the hero is three columns: one-sentence 1.2fr, analogy 1fr, headline stat 0.8fr (the stat never goes below 240px).
- **760 to 1199px:** the one-sentence and analogy boxes sit side by side (two columns, each at least 240px); the headline stat spans full width below.
- **Below 760px:** the hero stacks to one column.
- **Below 900px:** the sidebar becomes a drawer. A slim top bar (`web/src/shell/TopBar.tsx`) holds a hamburger button. The drawer slides in with `motion`, and it closes on a backdrop click, on Escape, or when an experiment is chosen (route change). The chapter bar sits under the top bar.
- Breakpoints apply to the viewport width, not the content width; the sidebar (260px) is part of the 900px+ layout.

### Palette (decided 2026-11-19)
Swiss is the only colour scheme. Space Grotesk Variable (headings and body) and JetBrains Mono Variable (mono), black, white and the signal red `#e10600`. Square corners, no shadows, 2px rules. The tokens sit on `:root` in `web/src/theme.css`; components use `var(--token)` only.
Token roles (`--tok-*`, and arms `--c-arm-a0..a4`) keep CIE76 dE >= 15 between every pair. The other schemes (lab, paper, blueprint, sage) are archived on branch `prototype/examiner-themes`.
`web/src/shell/PrototypeSwitcher.tsx` is a generic dev-only bar (`PrototypeSwitcher({ param, options })`) kept for future prototypes. It is not mounted anywhere now. It reads and writes one URL search param, cycles with ← →, dispatches a `prototype-param` event and is hidden in production builds. Canvas drawing reads tokens with `getComputedStyle` (`web/src/theme.ts`).

### Shared components (web/src/components/)
- `Term({ term, children })`: dotted underline; hover/focus shows plain-English definition from `web/src/glossary.ts`.
- `MathToggle({ children })`: collapsed "Show the maths" disclosure.
- `WhatThisShows({ children })`: one-line caption under every chart, prefixed "What this shows:".
- `Takeaway({ children })`: the bold one-liner a section or step leads with (≤ 15 words).
- `MoreDetail({ children, label? })`: collapsed "More detail" disclosure with an animated open; all prose beyond the takeaway goes here.
- `Stat({ value, label, format?, tone? })`: big number with a small label, counts up on change (writing rules).
- `AnimatedBars({ data: {label, value, color?, note?}[], max?, format? })`: horizontal bars that grow from 0 when scrolled into view (`useInView`) and smoothly tween when data changes.
- `AnimatedLines({ series: {name, ys, color?, dashed?}[], yLabel, xLabel, yMax?, log? })`: SVG lines that draw in (`pathLength` 0→1) on view; hover shows values.
- `Heatmap({ matrix, cellSize?, mode: 'diverging'|'sequential', max?, title? })`: canvas; tween between successive matrices over 250 ms.
- `TokenStrip({ tokens, roles, selected?, onSelect?, compactFiller? })`: role-coloured token chips, filler runs collapsible.
- `CodeWalkthrough({ steps })`: step = `{ title, plain, path, symbol? | {start,end}, notes: {match: string, note: string}[] }`.
  Fetches `/api/code`, shows numbered lines, highlights lines containing each `match` string, and clicking/hovering a
  highlighted line shows its `note` in a side panel. Step tabs across the top (1 Dataset → 2 Model → 3 The one line that differs → 4 Training → 5 Evaluation).
- `StatusChip`, `Unavailable({ reason })`.
Every animated component must respect `prefers-reduced-motion`.

### Decisions
- Datasets page: Assembly line visual chosen 2026-11-19; Annotated specimen and Pattern grammar archived on branch `prototype/examiner-dataset-visuals`.
