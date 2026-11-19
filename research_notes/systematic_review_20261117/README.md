# Systematic scoping and novelty review

**Completed search cutoff: 17 November 2026.** No lower year cutoff; retained historical foundations extend to 1982. The search protocol and access limits are recorded in [PROTOCOL.md](PROTOCOL.md). This is a bounded systematic scoping review, not an exhaustive review of every paper.

## Decision for this project

**The broad contribution is already occupied.** There is direct prior work on controlled scalar/channel decay comparisons, static channel gates, activation-based decay initialization, MQAR plus overwrite, and recurrent-memory interventions. The current results may support a focused replication and boundary study; a new architecture or initialization-method claim is not defensible.

No exact duplication of our complete empirical mean-alpha matching × scalar/channel GDN × retrieval/update-retention experiment was identified in the **selected primary evidence corpus**. This is a provisional gap, not a claim of priority. Close-looking inaccessible manuscripts and unread catalog candidates prevent a stronger negative conclusion. A distinctive combination of settings alone is not a scientific contribution.

## What was actually covered

| Audit quantity | Result |
|---|---:|
| Native keyword/bibliographic searches | 12 OpenAlex + 4 Crossref |
| Initial distinct normalized-title records | 405 |
| Final discovery catalog after citation expansion | 954 title keys |
| Primary records in the synthesis register | 46 |
| Relevant full-text/section-level evidence records | 13 |
| Primary record/abstract-level evidence records | 33 |
| Indexed forward citations retrieved for GDN / Kimi / Zoology | 343 / 311 / 167 |

**These counts do not mean 954 full papers were read.** The title ledger marks 31 catalog entries with selected-primary follow-up, 20 clear title-scope exclusions, and 903 catalog/background candidates not selected for primary reading. Other selected sources came through web searches and backward tracing. Versions with changed titles may remain separate in the discovery catalog. The 46-record evidence register uses stable source identifiers where available.

OpenAlex/Crossref were queried natively. arXiv, OpenReview/TMLR, PMLR, NeurIPS, ACL Anthology, author pages, and official repositories supplied primary evidence. Semantic Scholar was rate-limited; DBLP returned a server error. No native Scopus, Web of Science, Google Scholar, ACM, or IEEE export was obtained. Three particularly relevant catalog-only candidates remain unresolved below. This review improves the evidence base without certifying universal coverage.

## Claim-by-claim comparison

| Potential claim | Closest primary precedent | Assessment |
|---|---|---|
| First controlled scalar versus channel-wise decay comparison | [Das: controlled decay-design factorial study](https://tmlr.infinite-conf.org/paper_pages/FdXB4u27LD.html) | Already studied; includes static controls, multiple seeds, recall and initialization probes. |
| Channel-wise decay is a new architecture | [Kimi Linear / KDA](https://arxiv.org/abs/2510.26692), [GDN-2](https://arxiv.org/abs/2605.22791) | Already established. Our model is an adaptation. |
| Static per-channel decay or heterogeneous timescales are new | [Das](https://openreview.net/pdf?id=FdXB4u27LD), [PoST](https://arxiv.org/abs/2604.07658) | Already established. Post-hoc freezing does not substitute for an independently trained static baseline. |
| Data-driven decay initialization is new | [DAMI](https://arxiv.org/html/2610.41994v2), [Taylor-Calibrate](https://arxiv.org/abs/2606.16429) | Already established in related settings. Our joint launcher is an attributed adaptation; Taylor-Calibrate uses teacher distillation. |
| Evaluating MQAR and latest-write overwrite together is new | [KATA, sections 5.1–5.2](https://arxiv.org/html/2607.17419) | Already done, including GDN and OOD comparisons. |
| Load matters more than filler length | [How Linear Attention Remembers](https://arxiv.org/html/2609.33093) | Existing evidence in another regime; our pattern is corroboration. |
| Gate freezing is a new analysis method | [Muir, Appendix I.3](https://arxiv.org/html/2611.08376) | Existing intervention precedent. Its write/erase gates differ from our decay alpha. |
| Recall failures necessarily reveal architectural capacity limits | [Anatomy of Associative Recall](https://arxiv.org/abs/2609.16183), [Revisiting AR](https://arxiv.org/abs/2508.19029) | Not justified: training coverage, seed outcomes, convolution, and optimization are competing explanations. |
| Gate strength affects memorization versus retrieval | [On the Importance of Gating](https://arxiv.org/abs/2609.16540) | Already studied theoretically and experimentally in SSMs; distinct from our exact GDN factorial. |
| Exact matched-decay × granularity × repeated-update-retention interaction | Closest combination: Das + DAMI + KATA + gating-dynamics papers | No exact duplicate identified among selected sources; novelty and significance remain unproven. |

The older chain is also documented: Hopfield/content-addressable memory, Schmidhuber/fast weights, Ba/temporary memory, LSUV/data-based initialization, chrono gate biases, Schlag/delta-rule correction, and Zoology/MQAR. These are antecedents, not evidence that the complete present experiment has already been measured.

## What the expanded search changed

The initial review missed several relevant competitors. The broader catalog/citation pass identified:

- **Anatomy of Associative Recall:** a matched-state decomposition with seed-dependent training outcomes and curricula. This makes an optimization/training-coverage control especially important for our failures.
- **On the Importance of Gating:** gate strength can alter the competition between memorization and contextual retrieval. A calibration effect alone does not establish a new mechanism.
- **Revisiting Associative Recall:** recurrent-model performance is sensitive to learning-rate choice. A shared LR is a controlled recipe choice, not proof that every architecture received an optimal recipe.
- **Taylor-Calibrate:** another GDN initialization precedent, in a distillation setting.
- **Constant-Memory Recall** and **Variational Linear Attention:** further small fixed-state recall and selective-retention evidence.

Primary links and reading-depth labels for these records appear in the table above and [the evidence register](evidence_register.csv). These papers' findings are reported claims, not reproduced results from this review.

## What our measurements can still add

Candidate question: **When does matching initial decay change the apparent effect of channel-wise gating, and does that conclusion survive repeated updates outside the training distribution?**

The existing 24-model results, confirmation, and interventions provide material for this question. The distinctive observations are task-dependent calibration effects, preservation of MQAR training-side probe accuracy under static per-channel decay, and the first-replacement error signature under three rewrites. The measured count is 926 of 1065 newest-value errors across twelve overwrite models. These are observations specific to our recipe, not established general laws.

The static-gate result separates two interventions in the tested trained models: removing channel differences can hurt while removing decay's token dependence can preserve accuracy. Other operations remain dynamic. It does not show that a newly trained static architecture would behave identically or that the result survives harder held-out inputs.

## Recommended next experiment and decision criteria

1. **Treat the joint run as a published-baseline adaptation.** Retain its fixed targets and distinguish mean-alpha matching from mean-log-decay matching. To attribute an effect to sensitivity, add a mean-log-decay-only control; the current joint comparison changes both.
2. **Test training-distribution coverage.** Compare fixed versus varied rewrite counts at equal data/step exposure. Evaluate unseen rewrite counts and randomized positions on new examples, reporting latest-value recall, each obsolete-value category, and unrelated retention separately.
3. **Test a causal explanation.** Apply the same static/shuffled/averaged decay interventions on the new held-out stress cells. Add an independently trained static-channel control if the claim concerns architecture rather than checkpoint reliance.
4. **Report confounds.** Include per-seed learning curves and escape/success timing; separate load and distance; retain a small LR-stability check before attributing an architecture ranking to gates. Report the channel model's 4.17% parameter overhead or add a parameter-matched control. Prior work also motivates embedding/readout and numerical diagnostics, but it does not establish that those bugs exist in our runs.

**Go criterion:** a repeatable interaction or failure mechanism that survives these controls and explains something not resolved by the closest studies. **Narrow/negative outcome:** varied training removes the obsolete-answer signature and rankings remain seed-dependent; report a training-coverage boundary or replication result. **No-go for a strong novelty claim:** only baseline saturation, isolated seed failures, or a copied initialization benefit. More epochs alone cannot establish novelty.

These are recommendations, not additional training performed during this review. Publication outcome or venue suitability cannot be guaranteed.

## Unresolved close-looking sources

| Catalog title | Identifier | Status |
|---|---|---|
| Latent Recall Fragility in Kimi Delta Attention: A Conditional Dynamical Analysis | [Zenodo 10.5281/zenodo.23270559](https://doi.org/10.5281/zenodo.23270559) | Primary access unsuccessful; catalog-only, no result used. |
| Similarity Finds the Fact, Not the Version: A Co-Trained Order Stamp in the Key Resolves Version Conflict in Persistent Model-Internal Memory | [Research Square rs-10896018/v1](https://doi.org/10.21203/rs.3.rs-10896018/v1) | Primary access unsuccessful; potential overlap must remain unresolved. |
| The Query Cannot See the Question: A Short Convolution's Reach Decides Which Part of a Query Conditions Retrieval | [Research Square rs-10934426/v1](https://doi.org/10.21203/rs.3.rs-10934426/v1) | Primary access unsuccessful; catalog-only, no result used. |

## Files and audit trail

- [Protocol and access limits](PROTOCOL.md)
- [Primary evidence register](evidence_register.csv), [counts](review_counts.json)
- [Title screening ledger](screening_ledger.csv), [combined discovery catalog](extended_catalog_candidates.csv)
- [Native query log](catalog_query_log.json), [citation-page log](citation_extension_log.json), [web query/access log](web_search_log.json)
- [Primary arXiv metadata checks](primary_metadata_verification.json), [catalog citation edges](citation_edges.csv)
- [Historical review](history.md), [historical supplement](ownhistory.md), [gates and initialization](gates_initialization.md), [tasks and mechanisms](tasks_memory.md)

Raw catalog responses and rerunnable discovery/register scripts are retained in this folder. The protocol records a corrected fuzzy-seed selection error and title/ID bibliographic corrections. This transparency is part of the review, not evidence of complete retrieval.

Suggested manuscript wording: **“Within a documented search of historical and contemporary primary literature through 17 November 2026, we did not identify a study combining these particular controls. We position the work as a controlled replication and extension of prior gating, initialization, and associative-memory studies.”** Update that sentence after resolving outstanding sources and rerunning searches before submission.
