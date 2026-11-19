# Systematic review: gate granularity, decay initialization, and associative-memory evaluation

Search date: **17 November 2026** (Asia/Calcutta). Scope: papers and primary code/results on gated linear attention, GDN/DeltaNet/KDA, scalar versus channel-wise decay, static/dynamic decay, activation-based or timescale initialization, MQAR/associative recall, repeated-key overwrite, and causal gate interventions. The target claim is the project's current one: whether matching the initial realized decay of scalar and channel-wise GDN gates changes retrieval versus update/retention behavior, plus the planned DAMI-style adaptation.

## What this review can and cannot establish

This is a documented systematic search and citation chase, not a mathematical guarantee that every paper ever written has been found. I used arXiv, PMLR/NeurIPS, TMLR/OpenReview, and primary author or official repository pages. Search-engine results were used for discovery, then claims were checked against a primary landing page, paper, or official code/results where possible. Search coverage has no lower publication-year cutoff, but indexing, language, access, and terminology limit recall. “No exact match identified” means no exact match in the documented corpus; it does not prove priority.

## Search log

All searches were run on 2026-11-17 unless stated otherwise. Queries were varied rather than treated as one exact-string test.

| Query family | Sources searched | Result / follow-up |
|---|---|---|
| `Gated DeltaNet channel-wise gating initialization` | arXiv search | GDN, GDN-2, FG²-GDN, KDA candidates; opened primary abstracts. |
| `Kimi Linear channel-wise decay gate` | arXiv | Opened Kimi Linear v2 and technical report links. |
| `StaticChannelDelta scalar channel-wise decay controlled factorial` | TMLR page, OpenReview PDF, official GitHub | Located Das study and code/results; read abstract and relevant reported findings. |
| `PoST geometric decay spectra linear recurrence Gated DeltaNet` | arXiv | Opened Optimal Decay Spectra abstract; verified GDN/MQAR instantiation. |
| `FG²-GDN doubly fine-grained control` | arXiv | Opened primary abstract; verified channel-wise learning/erase extensions. |
| `linear attention associative recall repeated-key overwrite` | arXiv | Located KATA and related memory papers; opened KATA Sections 5.1–5.2. |
| `linear attention causal memory gate intervention` | arXiv | Located How Linear Attention Remembers and Muir; opened primary abstracts/full HTML. |
| `decay initialization timescale recurrent gate` | arXiv | Located DAMI, chrono initialization, LSUV, PoST. |
| `fast weight associative memory delta rule` | PMLR/NeurIPS/arXiv | Located Schmidhuber/Ba/Schlag foundations and DeltaNet parallelization. |
| backward references from GDN/KDA/DAMI/Das/KATA | each paper's references where accessible | Confirmed older fast-weight, gated-linear, delta-rule, and initialization antecedents. |
| forward/citation discovery by title and arXiv ID | arXiv search and official pages | Added 2026 papers GDN-2, Muir, KATA, How Linear Attention Remembers, Memory by Design, QED, DASC. |
| exact-phrase searches: `"mean decay" GDN`, `"matched decay" gating`, `"initial realized decay" channel-wise`, `MQAR overwrite calibration` | arXiv/web index | No exact source found in indexed results; this is a provisional negative finding, not proof of absence. |

## Candidate ledger

Evidence levels: **A** = primary full text/official results inspected; **B** = primary abstract/landing page inspected; **C** = discovery only or inaccessible primary text. Included means it directly informs overlap or the historical boundary. Excluded means discovered but not sufficiently relevant or not used for a substantive claim.

| Year | Primary source | Evidence | Include? | Verified overlap or role |
|---:|---|:---:|:---:|---|
| 1991–1992 | [Schmidhuber, fast-weight history and original references](https://people.idsia.ch/~juergen/neural-attention-1990-1993.html) | B | Yes | Fast-weight memories and context-dependent weight changes predate current linear-attention terminology. Historical source; not used for detailed numerical claims. |
| 2016 | [Ba et al., Using Fast Weights to Attend to the Recent Past](https://proceedings.neurips.cc/paper/2016/hash/9f44e956e3a2b7b5598c625fcc802c36-Abstract.html) | B | Yes | Temporary fast-weight memory and associative sequence processing. Establishes the broad memory setting. |
| 2016 | [Mishkin & Matas, LSUV](https://arxiv.org/abs/1511.06422) | B | Yes | Layer-sequential activation-statistic initialization. Establishes the general calibration pattern, not GDN decay. |
| 2018 | [Tallec & Ollivier, chrono initialization](https://arxiv.org/abs/1804.11188) | B | Yes | Gate-bias initialization from expected temporal horizons. Establishes timescale-aware initialization in recurrent gates. |
| 2021 | [Schlag et al., Linear Transformers Are Secretly Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html) | A | Yes | Delta-like fast-weight programming, finite-memory limitations, and dynamic learning rates. |
| 2023/2024 | [Arora et al., Zoology](https://arxiv.org/abs/2312.04927) | B | Yes | Introduces MQAR and studies associative recall in efficient models. MQAR is not new in this project. |
| 2024 | [Yang et al., Gated Linear Attention Transformers](https://proceedings.mlr.press/v235/yang24ab.html) | A | Yes | Input-dependent gated linear attention and efficient recurrent state updates. |
| 2024 | [Yang et al., Parallelizing Linear Transformers with the Delta Rule](https://arxiv.org/abs/2406.06484) | B | Yes | Efficient parallel delta-rule training; informs implementation but not our scientific claim. |
| 2024/2025 | [Yang et al., Gated Delta Networks](https://arxiv.org/abs/2412.06464) | B | Yes | Base GDN: adaptive forgetting combined with targeted delta updates. Blocks any novelty claim about the base recurrence. |
| 2025 | [Kimi Linear / KDA](https://arxiv.org/abs/2510.26692) | B | Yes | Channel-wise decay extends GDN's coarser gate and improves finite-state control. Blocks “first channel-wise gate” claims. |
| 2026 | [Cao, Optimal Decay Spectra for Linear Recurrences (PoST)](https://arxiv.org/abs/2604.07658) | B | Yes | Geometric decay spectra and position-adaptive scaling instantiated in GDN and evaluated on MQAR. Blocks generic multi-timescale novelty claims. |
| 2026 | [Sun et al., FG²-GDN](https://arxiv.org/abs/2604.19021) | B | Yes | Channel-wise update learning rates and decoupled key/value control. Shows the design space has moved beyond decay alone. |
| 2026 | [Das, Delta-Rule Writes in Linear Attention](https://tmlr.infinite-conf.org/paper_pages/FdXB4u27LD.html) ([paper](https://openreview.net/pdf?id=FdXB4u27LD), [code/results](https://github.com/emailtovamos/delta-rule-linear-attention)) | A | Yes | Closest direct overlap: factorial scalar/channel-wise × static/data-dependent × delta/non-delta, three seeds, scale studies, synthetic recall, and initialization ablations including StaticChannelDelta. |
| 2026 | [Feng et al., DAMI](https://arxiv.org/html/2610.41994v2) | A | Yes | Jointly fits `dt_bias` sensitivity and `A_log` timescales from activations. The planned run is an attributed one-head/four-layer adaptation, not a new method. |
| 2026 | [Hatamizadeh et al., Gated DeltaNet-2](https://arxiv.org/abs/2605.22791) | B | Yes | Channel-wise erase and write gates; repeated-memory editing is central. |
| 2026 | [Ghriss & Chakraborty, KATA](https://arxiv.org/html/2607.17419) | A | Yes | Explicit MQAR and repeated-key overwrite sections, including GDN comparisons and OOD key/context sweeps. Blocks “first paired task” claim. |
| 2026 | [Dowling et al., Memory by Design](https://arxiv.org/abs/2605.31163) | B | Yes | Covariance-aware memory, collision robustness, associative recall, MQAR, and GDN distillation. Adjacent mechanism/retention work. |
| 2026 | [Lee et al., How Linear Attention Remembers](https://arxiv.org/html/2609.33093) | A | Yes | Controlled causal interventions on GLA/GDN writes, retention, readout, load, and editability. Closely overlaps our mechanistic framing. |
| 2026 | [Gangavarapu et al., Muir](https://arxiv.org/html/2611.08376) | A | Yes | Gate-freezing/static-mean interventions, channel sensitivity, overwrite and multi-query recall in GDN-2/Muir. Gate object differs, but generic intervention novelty is blocked. |
| 2026 | [Gupta et al., Query-Derived Erase Direction](https://arxiv.org/abs/2608.13668) | B | Yes | Targets interference in fixed-state linear attention by adding a query-derived erase direction. Relevant to obsolete-value and editability failures. |
| 2026 | [Yu et al., Decay-Aware State Compression](https://arxiv.org/abs/2608.30386) | B | Yes | Analyzes head/channel retention horizons in GDN/KDA and uses them for state compression. Relevant to heterogeneous decay interpretation. |
| 2026 | [KATA / related source-code records not matching target](https://github.com/ayghri/kata) | C | No | Implementation corroborates KATA paper but was not counted as an additional paper. |

The ledger contains 20 candidates to reduce false negatives. Several are adjacent rather than exact matches; all are retained because they constrain claims about gates, timescales, memory load, overwrite, or causal interventions.

## Exact-overlap analysis

### Claim: scalar versus channel-wise decay is novel

Rejected. KDA already makes decay channel-wise; Das gives the closest controlled factorial comparison and StaticChannelDelta baseline. FG²-GDN and GDN-2 further refine channel-wise control. Our models are a small, one-head GDN adaptation and should be described as a replication/extension.

### Claim: fixed per-channel gates or heterogeneous timescales are novel

Rejected. StaticChannelDelta explicitly uses data-independent per-channel decay and an exponential spectrum. PoST studies geometric spectra in multiple recurrences. Das also reports initialization ablations. Our intervention preserves trained channel means while removing input variation; that is a narrower diagnostic, not invention of the mechanism.

### Claim: activation-sample calibration of decay is novel

Rejected as a broad method claim. DAMI jointly calibrates the same native GDN memory parameters from shallow-layer activations and gives equations for sensitivity and log-decay matching. LSUV and chrono initialization provide older general precedents. Our earlier mean-alpha offset fit is a simpler diagnostic; the new joint run is an attributed adaptation.

### Claim: MQAR plus overwrite is a novel evaluation combination

Rejected. Zoology establishes MQAR; KATA explicitly has both MQAR and repeated-key overwrite. Das and recent memory papers provide related associative/overwrite diagnostics. Our three-rewrite, unrelated-retention, stale-answer taxonomy can still be a protocol extension.

### Claim: gate-freezing or intervention analysis is novel

Rejected as a general methodological claim. How Linear Attention Remembers performs causal interventions on GLA/GDN memory pathways; Muir freezes/replaces gate pathways and reports effects. Our alpha-decay interventions are a smaller, more controlled object and should be framed as replication diagnostics.

### Claim: repeated overwrites returning obsolete values is a new phenomenon

Not supported as a priority claim. KATA and GDN-2 explicitly motivate repeated memory editing and latest-write behavior; How Linear Attention Remembers studies load/editability degradation. Our observed first-replacement error count (926/1065 newest-value errors) may be a new quantitative signature for this recipe, but it requires comparison with varied rewrite training, fresh seeds, and prior baselines before asserting generality.

### Claim: exact initial mean-decay matching × scalar/channel × MQAR/overwrite is already published

Not identified in the inspected corpus. Searches included `mean decay GDN`, `matched decay gating`, `initial realized decay channel-wise`, `MQAR overwrite calibration`, and title/ID searches around the closest papers. This is a provisional negative finding. DAMI is the nearest methodological overlap, and Das is the nearest granularity/controlled-design overlap, so the claim must be stated narrowly and with direct citations.

## What remains potentially defensible

The strongest candidate is a boundary/replication result:

> In a small one-head GDN adaptation, matched initial realized decay does not eliminate seed-dependent differences between scalar and channel-wise gates across clean multi-association recall and repeated-write update stress. Clean MQAR can be near-perfect while repeated-write stress reveals obsolete-value errors; interventions separate channel heterogeneity from token dependence on the tested probes.

This is an empirical extension, not a new architecture or calibration algorithm. The paper needs a direct prior-art table, per-seed results, confidence intervals or uncertainty summaries, and a parameter-matched control (current channel models have ~4.17% more parameters). The planned DAMI-style run should be labeled as an adapted comparison and should not be used to claim DAMI was independently invented.

## Required controls before making a strong conclusion

1. Include default/random, old mean-alpha calibration, and attributed DAMI-style joint initialization. Keep the targets fixed before observing new stress results.
2. Train and test varied rewrite counts; otherwise first-replacement errors may be a training-distribution shortcut.
3. Use fresh held-out data for final claims and report newest-value recall separately from unrelated retention.
4. Compare an independently trained static channel baseline, not only post-hoc frozen-gate interventions.
5. Match parameter count or report the 4.17% channel overhead explicitly.
6. Add a simple attention or published scalar GDN reference where compute permits; otherwise call the results a controlled small-model study.

## Limits and reproducibility

The review does not include every publisher index, thesis repository, non-English database, or unpublished manuscript. Some 2026 records are preprints and may change. OpenReview was intermittently protected by a browser-verification page; the Das paper's TMLR landing page and official code/results were used alongside the indexed paper PDF. Search dates, queries, source URLs, evidence levels, and the deduplication ledger are recorded above. Future updates should repeat the exact query families and append a dated delta rather than silently changing the conclusion.

**Verdict:** broad novelty is occupied. The precise calibrated small-model boundary study remains plausible but unproven; publication depends on showing a result that distinguishes it from Das, DAMI, KATA, and the causal memory-intervention literature.

## Supplementary catalog screen (17 November 2026)

The repository's OpenAlex catalog discovery produced four additional candidates. I verified each against its primary arXiv landing page rather than treating catalog metadata as evidence. All four are real arXiv records with titles, authors, abstracts, dates, and stable IDs. They are added below to avoid silently excluding newly indexed work.

| Record | Primary verification | Evidence level | Relevance to this project | Novelty consequence |
|---|---|:---:|---|---|
| [Unlocking Feature Learning in Gated Delta Networks at Scale](https://arxiv.org/abs/2606.04048), Liu & Gu, 2026 | arXiv abstract verified; submitted 2 Jun 2026; arXiv:2606.04048 | B | Derives GDN scaling rules through gates and recurrent state dynamics and validates learning-rate transfer across widths. | Not a direct gate-granularity or MQAR/overwrite comparison. It is relevant to the interpretation of our fixed-width/LR recipe: any claim about architecture behavior should not be conflated with width-transfer or parametrization effects. |
| [Complex KDA: Understanding and Enhancing the Expressivity of Kimi Delta Attention](https://arxiv.org/abs/2609.24797), Siems et al., 2026 | arXiv abstract verified; submitted 21 Sep 2026; arXiv:2609.24797 | B | Shows that KDA's channel-wise gate can supply a reflection enabling 2-D rotations and stronger state tracking under extended gate ranges. | Not an initialization-calibration paper, but it reinforces that channel-wise gates change transition expressivity, not merely memory half-life. Our post-hoc decay-only interventions cannot support broad claims about all channel-wise advantages. |
| [Taylor-Calibrate: Principled Initialization for Hybrid Linear Attention Distillation](https://arxiv.org/abs/2606.16429), Zhou et al., 2026 | arXiv abstract verified; submitted 15 Jun 2026; arXiv:2606.16429 | B | Uses teacher attention statistics to initialize GDN value projection, memory timescale, write gates, and output gate, followed by per-layer alignment. | Directly adds another data-driven GDN initialization precedent. Its teacher-distillation setting differs from our scratch training, but “activation/statistics-based initialization improves early dynamics” is not a unique contribution. Cite it when describing the calibration space. |
| [State Rank Dynamics in Linear Attention LLMs](https://arxiv.org/abs/2602.02195), Sun et al., 2026 | arXiv abstract verified; submitted 2 Feb 2026; arXiv:2602.02195 | B | Finds stable low-rank/high-rank head stratification in pretrained linear-attention LLM states and proposes rank-norm pruning. | Adjacent rather than direct: it studies state rank and head roles, not decay initialization. It cautions against interpreting channel-gate interventions as complete explanations of memory behavior without state-rank diagnostics. |

### Verification notes for supplementary records

The catalog rows were initially marked “discovery only; not full-text screened.” The arXiv pages confirm that the titles are not metadata hallucinations or duplicate records. At this pass I inspected the primary abstracts and metadata, but not all full HTML/PDF details or released code. Therefore the evidence level remains B. No numerical claims from these four records are used beyond their abstracts. They should be upgraded to A only after full-text sections and code are checked for a specific overlap.

These additions do not overturn the main decision. The nearest direct overlaps remain Das (factorial granularity/conditioning), DAMI (joint decay initialization), KATA (MQAR plus repeated-key overwrite), and the causal intervention papers. Taylor-Calibrate expands the initialization prior art; Complex KDA expands the expressivity prior art; Liu & Gu constrains scaling interpretation; State Rank Dynamics constrains mechanism interpretation.

## Supplementary exact-match / gating / training-dynamics screen

The citation-extension catalog surfaced three additional records. I verified each title, author list, date, abstract, and arXiv identifier on its primary arXiv page. These are real primary records, not metadata-only candidates.

| Record | Primary abstract finding | Relation to this project |
|---|---|---|
| [Latest Exact Match Attention](https://arxiv.org/abs/2609.25802), Brösamle, 2026 | A hard exact-match attention model stores many associations in a growing state and outperforms fixed-state GDN on a synthetic associative-recall task; it also evaluates repeated rare phrases and needle retrieval. | Not a gate-initialization or scalar/channel ablation. It is a strong reminder that near-perfect associative recall can come from a growing memory rather than a better fixed-state gate. Do not frame our MQAR results as a general solution to exact retrieval. |
| [Near-Zero Token Embeddings Silently Stall Training in Gated Sequence Models](https://arxiv.org/abs/2611.04449), Liu et al., 2026 | Reports a training failure in gated models caused by near-zero token embeddings, with historical-state readout suppression and gradient/clipping failures; evaluates Mamba, GDN, and xLSTM and proposes GroupClip/GateFloor. | This is not a decay-gate comparison, but it is directly relevant to debugging training traces. Before interpreting seed differences as memory mechanisms, verify embedding norms, state-readout magnitudes, non-finite gradients, and clipping events in our runs. |
| [On the Importance of Gating: Memorization vs. In-Context Learning in State Space Models](https://arxiv.org/abs/2609.16540), Tong et al., 2026 | Theory and experiments argue that gating can induce an in-weights memorization solution and delay or prevent a correct in-context-learning solution, while often helping long-sequence generalization. | Highly relevant to our interpretation: clean MQAR performance and obsolete-value stress can reflect training dynamics or memorization shortcuts, not only finite-state capacity. It strengthens the case for varied rewrite training, fresh OOD tests, and diagnostics that distinguish weights from recurrent-state use. |

### Three-line implication

These records do not duplicate the exact mean-decay-matching factorial protocol. They do narrow our claims further: exact-match attention already provides a non-gated associative-memory contrast; near-zero embeddings are a concrete alternative explanation for seed-dependent gated training failures; and gating can change whether a model learns in-weights memorization versus in-context behavior. Any final paper must audit those possibilities before calling the obsolete-value pattern a decay-mechanism limitation.
