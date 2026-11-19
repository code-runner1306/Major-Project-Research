# Systematic historical review: associative memory, gated decay, and overwrite

**Search date:** 17 November 2026 (Asia/Calcutta)  
**Question:** Has prior work already studied the exact combination of (i) scalar versus channel-wise decay in a delta-rule or GDN-like recurrent memory, (ii) matching or calibrating initial realized decay from activations, and (iii) MQAR plus repeated-write/overwrite and unrelated-retention evaluation?  
**Review type:** Targeted systematic search across primary papers and author/proceedings pages, with backward tracing from the closest papers. It is not a registered systematic review and cannot guarantee discovery of every paper, thesis, preprint, code-only result, unpublished manuscript, non-English source, or inaccessible citation.

## Search protocol

The searches were designed around both terminology and mechanism. “Scalar” and “channel-wise” are not universal labels, so searches also used diagonal decay, data-independent/static gates, per-channel forgetting, fast weights, delta rule, associative recall, repeated-key overwrite, latest-write, chrono initialization, and sensitivity/timescale initialization.

| Date | Query family | Primary-source follow-up | Outcome |
|---|---|---|---|
| 2026-11-17 | `fast weight programmer associative memory delta rule attention 1992 2016` | PMLR NeurIPS proceedings; arXiv records | Fast-weight and delta-memory precedents verified. |
| 2026-11-17 | `chrono initialization recurrent neural network long term dependencies` | arXiv paper record | Chrono gate-bias initialization verified. |
| 2026-11-17 | `gated linear attention data dependent decay 2023` | arXiv paper records and released-paper pages | GLA and GDN decay precedents verified. |
| 2026-11-17 | `S4 structured state space long range dependencies initialization` | arXiv paper record | Structured state-space long-range precedent verified. |
| 2026-11-17 | `scalar channel-wise overwrite linear attention` | TMLR page and paper PDF | Closest direct factorial study found. |
| 2026-11-17 | `MQAR repeated-key overwrite Gated DeltaNet` | KATA HTML paper and Zoology arXiv record | Direct task overlap found. |
| 2026-11-17 | `GDN causal memory intervention gate freeze` | How Linear Attention Remembers and Muir HTML papers | Mechanistic/intervention overlap found. |
| 2026-11-17 | `joint initialization A_log dt_bias sensitivity` | DAMI HTML/arXiv record | Direct initialization overlap found. |
| 2026-11-17 | Backward references from Zoology, GDN, Das, DAMI, KATA | Linked primary references and paper sections | Historical candidates below verified or marked as limited. |
| 2026-11-17 | Forward/current search for `linear attention memory retrieval overwrite` | arXiv records through the search date | Recent KATA, Muir, Bayesian Layers, and memory-mechanism papers found. |

Search-engine results were used for discovery; claims below rely on a primary paper, proceedings record, or author-maintained source. Third-party summaries are excluded from the evidence ledger.

## Candidate ledger

Evidence levels: **A** = primary paper/proceedings/full text inspected; **B** = primary abstract/record inspected; **C** = discovery only and excluded from substantive claims.

| Year | Primary work | Evidence / relevance to our question | Classification |
|---:|---|---|---|
| 1982 | [Hopfield, Neural networks and physical systems with emergent collective computational abilities](https://www.pnas.org/doi/10.1073/pnas.79.8.2554) | A/B historical foundation for content-addressable associative memory. It is not a modern gated delta recurrence. | A — background, not direct overlap |
| 1992 | [Schmidhuber, Learning to Control Fast-Weight Memories](https://gwern.net/doc/ai/nn/rnn/1992-schmidhuber-2.pdf) | Primary paper describes a controller that produces context-dependent changes to fast weights. | A — fast-memory precedent |
| 2016 | [Ba et al., Using Fast Weights to Attend to the Recent Past](https://proceedings.neurips.cc/paper/2016/hash/9f44e956e3a2b7b5598c625fcc802c36-Abstract.html) | Fast weights store temporary recent memories and act as an associative mechanism. | A — associative-memory precedent |
| 2016 | [Mishkin and Matas, All You Need Is a Good Init (LSUV)](https://arxiv.org/abs/1511.06422) | Sequentially calibrates layer outputs from data to unit variance. | A — calibration procedure precedent; different architecture/target |
| 2018 | [Tallec and Ollivier, Can recurrent neural networks warp time?](https://arxiv.org/abs/1804.11188) | Derives chrono initialization for gate biases from temporal horizons. | A — timescale-initialization precedent |
| 2020 | [Katharopoulos et al., Transformers are RNNs](https://arxiv.org/abs/2006.16236) | Establishes linear-attention recurrent form and fixed-state computation. | B — recurrence foundation |
| 2021 | [Gu, Goel, Ré, Efficiently Modeling Long Sequences with Structured State Spaces (S4)](https://arxiv.org/abs/2111.00396) | Structured state dynamics and stable long-range sequence modeling; not associative key-value overwrite. | A — state-space context |
| 2021 | [Schlag, Irie, Schmidhuber, Linear Transformers Are Secretly Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html) | Formal fast-weight view of linear attention; delta-like programming corrects current key-value mappings and addresses finite memory. | A — delta/overwrite foundation |
| 2023 | [Yang et al., Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635) | Input-dependent gates in linear-attention Transformers; shows gated linear attention as an efficient language-model layer. | B — dynamic-gating precedent |
| 2023/24 | [Arora et al., Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927) | Introduces MQAR and studies associative recall in gated/convolutional and attention models. | A — direct benchmark precedent |
| 2024 | [Yang et al., Parallelizing Linear Transformers with the Delta Rule](https://arxiv.org/abs/2406.06484) | Delta-rule recurrence and parallel sequence computation. | B — implementation/training precedent |
| 2024/25 | [Yang et al., Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) | GDN combines input-dependent forgetting with delta-rule writes. | A/B — base architecture precedent |
| 2025 | [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) | KDA uses channel-wise decay in a delta-attention architecture. | B — channel-wise architecture precedent |
| 2026 | [Das, Delta-Rule Writes in Linear Attention: A Controlled Factorial Study of Decay Design](https://tmlr.infinite-conf.org/paper_pages/FdXB4u27LD.html) ([paper](https://openreview.net/pdf?id=FdXB4u27LD)) | Directly crosses scalar/channel-wise, static/data-dependent gates, and delta/no-delta with three seeds and scale studies; includes synthetic recall and initialization ablations. | A — very high direct overlap |
| 2026 | [Hatamizadeh et al., Gated DeltaNet-2](https://arxiv.org/abs/2605.22791) | Channel-wise erase and write gates; separates update roles. | B — close architecture/interference precedent |
| 2026 | [Feng et al., DAMI](https://arxiv.org/html/2610.41994v2) | Jointly initializes `A_log` and `dt_bias` from activations, targeting head timescales and layer sensitivity. | A — very high initialization overlap |
| 2026 | [Ghriss and Chakraborty, Kernelized Linear Attention (KATA)](https://arxiv.org/html/2607.17419) | Dedicated MQAR and repeated-key-overwrite sections; evaluates GDN and OOD load/context. | A — very high task/protocol overlap |
| 2026 | [Lee et al., How Linear Attention Remembers](https://arxiv.org/html/2609.33093) | Causal interventions trace writes, retention, readout, cross-fact coupling, load, and editability in GLA/GDN. | A — high mechanistic overlap |
| 2026 | [Dowling et al., Memory by Design: Bayesian Layers for Sequence Modeling](https://arxiv.org/abs/2605.31163) | Covariance-aware memory and collision/associative-recall robustness; includes GDN distillation and MQAR context. | B — adjacent memory-robustness precedent |
| 2026 | [Gangavarapu et al., Muir](https://arxiv.org/html/2611.08376) | Channel-wise gated delta memory, multi-query recall, write/erase sensitivity, and counterfactual freezing/static interventions. | A — high intervention overlap |

The ledger intentionally includes older foundations even when they are not close competitors. It does not count every citation in each paper; it records the primary works most directly connected to the proposed claims.

## What is already established

1. Associative/content-addressable memory and fast-weight memories predate modern linear attention.
2. Delta-rule updates are an established way to correct a key-value mapping and implement replacement.
3. MQAR is an established synthetic recall benchmark.
4. Scalar and channel-wise/diagonal gating are already compared in modern delta-rule studies. Static channel decay is already a named control (`StaticChannelDelta`).
5. Gate/timescale initialization is established in recurrent networks; DAMI directly targets the GDN `A_log`/`dt_bias` pair with data-dependent calibration.
6. MQAR and repeated-key/latest-write overwrite have already been evaluated together in modern linear-attention work.
7. Causal gate freezing and memory interventions are established analysis tools.

## What remains potentially distinct

The inspected primary sources did not reveal an exact duplicate of this complete protocol: a small one-head GDN adaptation, scalar/channel-wise token-dependent models, empirical mean-decay matching, answer-free MQAR and overwrite with unrelated-key retention, three seeds, and a post-hoc breakdown of obsolete returned values. This is a **provisional gap**, not a priority guarantee.

Our current data can therefore be framed as a boundary/replication study:

> Under a 1.6M one-head GDN recipe, does matching initial realized decay remove scalar/channel differences, and does that conclusion survive repeated-write generalization?

The answer should be compared explicitly with Das, DAMI, KATA, and the mechanistic/intervention literature. The result is not a new gate architecture or a new initialization method. If using the joint launcher, call it a DAMI-style adaptation and identify which target equations are reproduced versus modified.

## Required novelty safeguards

Before claiming a contribution, include:

- A claim-to-prior-art table naming the closest paper for every claimed novelty.
- Parameter-matched scalar/channel controls, because the current channel model has approximately 4.17% more trainable parameters.
- Prespecified targets and fresh held-out examples; do not tune half-lives or stress settings after inspecting results.
- A comparison among random/default, mean-decay matching, and DAMI-style joint calibration.
- Rewrite-count training variation and unseen rewrite-count evaluation to distinguish a training-distribution shortcut from a fundamental failure.
- Per-seed results and uncertainty; three seeds are descriptive, not a guarantee of generality.

## Limits of the “exhaustive” request

This review is exhaustive only in the procedural sense that the search terms, source families, cutoff date, and inclusion decisions are recorded. It is not an exhaustive census of all literature. No responsible literature search can guarantee every paper because coverage depends on indexing, terminology, access, language, publication status, and citation lag. The defensible conclusion is: **within the documented search scope, no exact duplicate was identified, while broad novelty is clearly pre-empted by the direct prior work above.**
