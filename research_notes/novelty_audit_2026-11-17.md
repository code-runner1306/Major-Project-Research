# Novelty audit: historical foundations and closest overlapping work

Reviewed 17 November 2026. This is a targeted, historically broad literature search, not an exhaustive review of every paper or a guarantee of priority. Primary papers, proceedings, and author material support the findings below. Search snippets and third-party summaries were used only for discovery. Preprint findings are attributed claims, not independently reproduced results.

## Decision

**Substantial prior-art overlap; narrow the contribution.** Scalar versus channel decay, static channel gates, timescale initialization, MQAR, repeated-key overwrite, and causal memory interventions all have prior work. The available evidence does not justify presenting any of these as our invention. I did not identify an exact duplication of our complete gate-granularity × empirical mean-decay matching × retrieval/update-retention protocol in the inspected sources. That narrower absence is provisional, not proof of novelty or sufficient evidence of publication quality.

Our existing data can support a small-model replication/extension and potentially a focused failure analysis. A strong original-method paper is not established. Additional seeds or epochs alone do not solve this positioning issue.

## Historically relevant foundations

| Work | Established contribution relevant here | Implication |
|---|---|---|
| [Schmidhuber, fast-weight history and original references (1991–1992)](https://people.idsia.ch/~juergen/neural-attention-1990-1993.html) | Learned context-dependent changes to fast-weight memory predate modern attention. This is an author historical account, not a full rereading of every original paper. | Do not claim learned associative memory is new. |
| [Ba et al., Using Fast Weights to Attend to the Recent Past, NeurIPS 2016](https://proceedings.neurips.cc/paper/2016/hash/9f44e956e3a2b7b5598c625fcc802c36-Abstract.html) | Fast weights store temporary information about recent inputs. | Fixed-memory recall has a long experimental history. |
| [Mishkin and Matas, LSUV, ICLR 2016](https://arxiv.org/abs/1511.06422) | Data-based initialization proceeds sequentially through layers to match activation statistics. | Layer-sequential fitting itself is established. |
| [Tallec and Ollivier, Can recurrent neural networks warp time?, ICLR 2018](https://arxiv.org/abs/1804.11188) | Chrono initialization configures recurrent gate biases for temporal dependencies. | Timescale-aware gate initialization is established, though its LSTM/GRU setting differs. |
| [Schlag, Irie, Schmidhuber, Linear Transformers Are Secretly Fast Weight Programmers, ICML 2021](https://proceedings.mlr.press/v139/schlag21a.html) | Connects linear attention and fast weights; delta-like writes correct key–value mappings and address finite-memory limitations. | Selective replacement is an existing motivation for delta updates. |
| [Arora et al., Zoology, 2023 / ICLR 2024](https://arxiv.org/abs/2312.04927) | Introduces MQAR and studies recall differences between efficient architectures. | MQAR is a borrowed benchmark, not our contribution. |
| [Yang et al., GLA, 2023 / ICML 2024](https://arxiv.org/abs/2312.06635) | Input-dependent forgetting in efficient linear attention. | Dynamic gated memory is established. |
| [Yang et al., Parallelizing Linear Transformers with the Delta Rule, 2024](https://arxiv.org/abs/2406.06484) | Efficient sequence-parallel delta-rule training. | Our portable executor is not the first efficient delta-rule implementation. |
| [Yang et al., Gated Delta Networks, 2024–2025](https://arxiv.org/abs/2412.06464) | Combines forgetting and delta-rule memory updates. | The base recurrence and motivation are prior work. |
| [Kimi Linear, 2025](https://arxiv.org/abs/2510.26692) | KDA introduces finer-grained gating in a delta-attention architecture. | Channel-wise gating is not a new architecture from this project. |

The older sources establish antecedents, not an exact duplicate of our experiment. They must not be treated as evidence that the precise current factorial interaction was already measured.

## Closest direct and adjacent overlaps

### Controlled decay-design factorial study — very high overlap

[Das, Delta-Rule Writes in Linear Attention: A Controlled Factorial Study of Decay Design](https://tmlr.infinite-conf.org/paper_pages/FdXB4u27LD.html), [paper](https://openreview.net/pdf?id=FdXB4u27LD).

The official abstract describes scalar/channel-wise comparisons, static learned gates, delta/non-delta factorial controls, three seeds, and 18M/125M models on TinyStories and WikiText-103. It reports StaticChannelDelta competitiveness and scale/dataset dependence. The paper also contains a synthetic recall probe and initialization ablations. This blocks claims of the first controlled granularity comparison or first evidence that static channel diversity can matter. Our empirical mean-matching intervention and update-retention tasks differ; that difference must carry explanatory value rather than merely alter the benchmark. Independent inspection also identified the paper's scalar/channel recall disagreement as a useful replication target, not evidence that our different MQAR setup refutes it. See [the independent audit](novelty_independent_review_2026-11-17.md) for protocol details.

### DAMI — very high initialization overlap

[Feng et al., DAMI, 2026, v2](https://arxiv.org/html/2610.41994v2).

Jointly configures `A_log` and `dt_bias` using activation samples, target timescales, and layer sensitivities. Equations 10–11 fit mean sigmoid sensitivity and mean log decay. Our older method changes only offsets to match mean alpha; our new launcher is explicitly a one-head, four-layer adaptation, not an independent invention of DAMI. A negative result for mean-only matching cannot refute DAMI. The new run can be an attributed comparison baseline, not the paper's novel method.

### KATA — very high task-combination overlap

[Ghriss and Chakraborty, Kernelized Linear Attention: Breaking the Capacity Wall with Symmetric Cones, 2026](https://arxiv.org/html/2607.17419), sections 5.1–5.2 and Table 2.

Already evaluates both MQAR and repeated-key overwrite against GDN in small Zoology-based models, with parameter/state reporting and out-of-distribution tests. It reports a recurrence-dependent retrieval/overwrite tradeoff. Its overwrite task writes each key twice; our three-rewrite shift and unrelated-key retention scoring differ. These differences need controlled evidence: the broad pairing of tasks or existence of a recall/update tradeoff is not new.

### How Linear Attention Remembers — high mechanistic overlap

[Lee et al., How Linear Attention Remembers, 2026](https://arxiv.org/html/2609.33093).

Uses causal interventions in pretrained GLA/GDN to examine writes, retention, readout, multi-fact coupling, and editability. It reports that increased load matters more than elapsed context in its regime. Our similar load-versus-length pattern is therefore corroboration, not a first discovery. Our interventions address decay-gate structure in small controlled training runs, which is a narrower distinct setting.

### Muir — adjacent gate-intervention overlap

[Gangavarapu et al., Muir, 2026](https://arxiv.org/html/2611.08376), especially Appendix I.3/Table 21.

Includes static-mean write/erase interventions and gate-rank reductions in GDN-2/Muir. This is relevant precedent for separating channel structure from token conditioning. Its erase/write gates are not the same object as our decay alpha: do not equate the results. It nevertheless blocks presenting generic gate freezing or the importance of static channel baselines as an unprecedented analysis technique.

### Additional architecture and interference context

[Gated DeltaNet-2, 2026](https://arxiv.org/abs/2605.22791) separates channel-wise erasure and writing. [Memory by Design: Bayesian Layers for Sequence Modeling, v5 2026](https://arxiv.org/abs/2605.31163) studies covariance-aware preservation, collision effects, and retrieval robustness. Neither establishes exact duplication of our calibration experiment, but selective preservation and interference are active, well-established research topics.

## What our evidence currently supports

Local measured evidence is in `research_results/stronger_gate_tests_20261117`, `memory_stress_tests_20261117`, and the three-seed confirmation/diagnostics folders.

1. Calibration has different observed effects on MQAR learning and overwrite retention; no universal benefit was demonstrated. This is a candidate interaction result, limited to our recipe and three seeds.
2. Fixed per-channel decay preserves all six MQAR channel-checkpoint training-side probes at 100%, while channel averaging harms some. This distinguishes channel heterogeneity from token dependence on those probes. It does not establish a general static-gating theorem, OOD sufficiency, or a fair independently trained static baseline.
3. With three rewrites, 926 of 1065 newest-value errors select the first replacement. This is a specific descriptive failure signature. Its cause and persistence across varied training distributions remain untested.
4. Seed dependence, one head, small capacity, same-backbone rather than parameter-matched models, saturated baseline accuracy, and observed stress data limit scope. An arbitrary unique configuration is not itself substantive novelty.

## Recommended framing and next decision

Candidate question: **When does initial decay matching change the apparent value of channel-wise gating, and does that change survive repeated-update generalization?** Treat this as a replication and diagnostic extension of prior gating and initialization studies.

Before committing substantially more compute, make a claim-to-evidence table and specify what result would distinguish this question from Das/DAMI/KATA. The new joint run should be labeled an adapted published baseline. Its current mean-log-decay match also differs from the old mean-alpha match; a mean-log-decay-only control is needed to isolate sensitivity as the cause.

The most informative follow-up to the obsolete-answer pattern would compare fixed versus varied rewrite counts under equal training exposure, then test unseen rewrite counts, unrelated retention, and randomized positions. Use new untouched evaluation examples and causal controls. This is a recommendation, not work performed in this review. If the signature disappears under ordinary varied training, describe it as a training-distribution shortcut rather than a fundamental architecture failure.

## Search scope and limits

Search families included fast weights and delta-rule associative memory; chrono/data-dependent initialization; scalar/channel/static/dynamic decay; MQAR; repeated-key overwrite; load/context stress; and gate-freezing interventions. Starting papers' cited antecedents were followed to older primary sources. The review covers relevant historical foundations through 17 November 2026, without a recent-only date filter. It does not cover every unpublished manuscript, inaccessible thesis, non-English paper, or source-code-only result. Some older papers were checked at abstract/proceedings level, while closest overlaps were checked at section/full-text level. Search-engine dates were not used as reliable publication dates.

**Conclusion: broad novelty is already occupied. The precise diagnostic extension remains plausible but unproven, and publication success cannot be promised.**
