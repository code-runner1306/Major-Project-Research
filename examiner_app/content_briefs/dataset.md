# Content brief: The datasets (`dataset`)

Audience: a technical person new to the field. Plain English first. Every example shown is produced live by the repo's own generator through the API (never hand-written).

## Generators (verified facts; see each file for details)
| id | Name on page | Source | Ours? | One example |
|---|---|---|---|---|
| `mqar` | Multi-query associative recall (Zoology) | `recipe_data.mqar` | Byte-exact port of HazyResearch Zoology (commit de4e258), verified equal on 1000 rows | 64 tokens: 4 key–value pairs at positions 0–7, then filler token 0 with each key asked once at random places (power-law spacing). 4 scored answers, at the asked keys. Keys 1–4095, values 4096–8191. |
| `overwrite` | Keyed overwrite (our task) | `retrieval.RetrievalDataset` generator v4 via `recipe_data.build_data('overwrite', …)` | Ours; motivated by KATA's repeated-key test | 64 tokens: filler token 1, then 5 interleaved writes (4 keys, the queried key written twice), the latest write placed at a random interior position with ≥1 unrelated write after it; query marker 0 then key. 2 scored answers: queried key's latest value and an unrelated key's value. |
| `arm` | Layout-fix training arms A0–A4 | `layout_fix_data.build_arm_data` | Ours (adaptation of overwrite v4) | Same as overwrite; A1 varies query distance 8–16, A2 rejects block-first layouts, A3 varies rewrites 1–3, A4 both. |
| `layout` | Named test layouts | `confirm_block_layout.sample_layouts` + `special_patterns` | Ours | Overwrite example drawn until its write pattern matches a named layout: block_first `T…TUUU`, block_after_one_unrelated `UT…TUU`, run_broken_before_latest, initial_then_run (≥2 rewrites), general_nonblock. T = write of the queried key, U = other write. |
| `length` | Longer sequences (filler inserted) | `run_paper_mqar_length_extension.py` logic: Zoology MQAR + `L−64` zeros inserted after the 8 memory tokens | Ours | 64 / 256 / 1024 tokens; memory unchanged, queries pushed later. |
| `kata` | KATA two-pass overwrite (reconstruction) | `kata_overwrite_data.generate` | Our reconstruction of Ghriss & Chakraborty (arXiv 2607.17419) Appendix F; their data code is unpublished | Pass 1: all K keys with values; pass 2: same keys, new order, new values; then each key once as `[key, 0]`. T = 6K. Scored: K answers (latest values). No filler. |
| `anatomy_e1` | Anatomy E1: masked recall | `third_party/dualgoose/dualgoose/toy_data.make_mqar_batch(…, query_layout='after_table')` | Upstream code (JIBSIL/dualgoose 8ca624a, Boesch & Wee arXiv 2609.16183) | Keys are fixed tokens 5+i at even positions; values random; one query key then a MASK slot (the answer); remaining positions random noise tokens. Vocab 64. Bidirectional denoiser task. |
| `anatomy_e4` | Anatomy E4: distractor haystack | `make_long_retrieval_batch(…, collision=False)` | Upstream | 64 tokens: 4-pair table, 48-token random haystack, 4 queries `[key, MASK]`. |
| `anatomy_e5` | Anatomy E5: colliding keys | `make_long_retrieval_batch(…, collision=True)` | Upstream | Like E4 but the haystack is full of decoy pairs reusing the table keys with wrong values; the true answer is the first (table) binding. |
| `zoology_cell` | Zoology layout test cells | `zoology_layout_test.make_cell` | Ours (built on the Zoology port) | MQAR with the memory shifted by 1–16 tokens, or queries near/far/in memory order/reversed. |
| `tinystories` | TinyStories (language) | HF `roneneldan/TinyStories`, GPT-2 tokenizer, packed into 512-token windows | External data, our packing | **Not available offline on this machine** (no cached dataset or tokenizer). Show description + a static note; no live generation. |

Token conventions differ: our overwrite uses 0 = query marker, 1 = filler; Zoology MQAR uses 0 = filler/answer slot; KATA uses 0 = answer slot; Anatomy uses keys 5+i and MASK = vocab−1. The page must say this plainly ("different generators use different reserved tokens").

## Page sections
1. **question**: "What data did the models learn from and get tested on?" Visual: the family tree of generators (Zoology → our MQAR port → length extension, zoology cells; KATA → our overwrite v4 → arms A0–A4, named layouts; Anatomy E1/E4/E5 upstream), each node labelled ported / ours / upstream / external.
2. **model** → retitle "Where each dataset comes from": provenance cards with citation, commit and "ported / ours / upstream / external" badge.
3. **code**: CodeWalkthrough steps: Zoology port `recipe_data.py` `mqar`; overwrite v4 `retrieval.py` `RetrievalDataset._overwrite_v4`; arms `layout_fix_data.py` lines 20–27; KATA `kata_overwrite_data.py` `generate`; Anatomy `third_party/dualgoose/dualgoose/toy_data.py` `make_long_retrieval_batch`.
4. **results** → "Example gallery": one live example per generator side by side, aligned TokenStrips with role legend and the scored positions highlighted.
5. **inside** → "How one example is built": THE PROTOTYPED VISUAL (3 variants, see below).
6. **try** → "Generate your own": generator picker + its parameters + seed dice; shows the example, a plain-English reading ("key 1850 is written with 4720, then rewritten with 4571; the model is asked for 1850 → correct answer 4571"), and the raw JSON toggle.

## Prototype variants for section 5 (switch with `?visual=assembly|anatomy|grammar` via PrototypeSwitcher)
- **assembly**: an assembly line. Step-by-step animation building one live overwrite example: draw 4 keys → draw values → choose which key is rewritten and its new value → order the writes (latest placed with ≥1 unrelated write after) → pad with filler → append queries → mark scored answers. Tokens fly into the strip (motion layout animations). Play / step / restart.
- **anatomy**: an annotated specimen. One example per generator stacked and aligned, with brackets and callouts (memory, gap/haystack, queries, answer slots) drawn as SVG over the TokenStrips; hovering a bracket highlights the same region in all generators, so differences between MQAR, overwrite, KATA and Anatomy are visible at a glance.
- **grammar**: a pattern grammar. Abstract blocks (T / U writes, gap, query) instead of numbers; sliders for rewrites (1–4), distance (8–24) and layout name morph the block layout live (motion `layout`), with the real token strip below re-generated from the API on each change.
