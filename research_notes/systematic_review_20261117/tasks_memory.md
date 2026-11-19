# Systematic review: MQAR, overwrite, load, and recurrent-memory diagnostics

Date of search: **17 November 2026** (Asia/Calcutta). Scope: historical and recent primary research relevant to the project’s task and memory claims. The project studies a small one-head Gated DeltaNet adaptation with scalar/channel-wise decay, activation-sample decay matching, MQAR, one- and multi-rewrite overwrite, unrelated-association retention, and gate interventions.

## What this review can and cannot establish

This is a transparent, broad search of primary papers and official proceedings pages. It is not a guarantee that every paper ever written was found: indexing differs across services, terminology changes, some papers are inaccessible, and unpublished or non-indexed work can exist. “No exact match found” below means no exact match was found in the documented search and citation-chasing scope; it is not a priority claim.

The review is designed to answer two questions:

1. Are MQAR, repeated-key overwrite, load/context stress, and causal recurrent-memory analysis already established methods?
2. Does our observed obsolete-value error pattern look like a new phenomenon or a useful boundary/diagnostic extension?

## Search log

All web searches were performed on 2026-11-17. Search results were used for discovery; each retained claim was checked against an arXiv page, conference/proceedings page, paper HTML/PDF, or official project page. Search engines may return later papers or duplicate versions; duplicate records are consolidated below.

| Query | Purpose | Result / action |
|---|---|---|
| `site:arxiv.org MQAR multi-query associative recall linear attention overwrite` | Find MQAR and close variants | Zoology, KATA, later associative-recall work; retained primary sources. |
| `site:arxiv.org repeated-key overwrite linear attention Gated DeltaNet` | Find latest-write and repeated-update tests | KATA, GDN-2, Bayesian Layers, related capacity work. |
| `site:arxiv.org causal interventions recurrent memory GDN linear attention` | Find memory-mechanism interventions | How Linear Attention Remembers, Muir. |
| `site:arxiv.org RULER benchmark long context retrieval linear attention` | Check established long-context retrieval methodology | RULER and papers using NIAH/RULER; retained RULER primary paper. |
| `RULER benchmark long context retrieval arxiv Hsieh 2024 needle haystack` | Verify RULER scope and tasks | Official arXiv record. |
| `MAD benchmark memory associative dynamic language model retention overwrite arxiv` | Search for memory/associative benchmarks under alternate names | No clearly closer primary source was retained from this query; terminology is broad. |
| `associative recall benchmark recurrent linear attention synthetic memory arxiv` | Search older and newer synthetic-memory tasks | Zoology, fast-weight, factual-recall and recurrent-model records. |
| `Gated Linear Attention Transformers hardware efficient training arxiv 2312.06635` | Verify GLA as the dynamic-gate foundation | Official GLA arXiv/proceedings records. |
| `fast weight programmers delta rule associative memory overwrite Schlag 2021` | Trace historical fast-weight and delta-rule roots | Fast Weight Programmer and associative-inference records. |
| `Zoology recall associative multi query 2023 2024 paper` | Trace MQAR source and citation family | Zoology official arXiv and code links. |
| `chrono initialization recurrent networks Tallec Ollivier 2018 timescales` | Check older initialization precedent | Official arXiv record. |
| `linear attention associative memory overwrite repeated updates benchmark gating scalar channel` | Search exact project concepts with alternate wording | KATA, GDN-2, How Linear Attention Remembers, Muir. |
| `site:arxiv.org factual recall linear associative memories capacity` | Search capacity and collision explanations | Factual-recall theory record. |
| `site:arxiv.org load context memory recall recurrent state interference` | Search load versus distance/context analyses | How Linear Attention Remembers, KATA, Bayesian Layers. |
| `site:arxiv.org Gated DeltaNet repeated writes latest value` | Search direct GDN overwrite evidence | KATA, GDN-2, Bayesian Layers. |
| `site:arxiv.org multi-query associative recall recurrent model learning rate` | Search learnability and recipe sensitivity | Revisiting associative recall in modern recurrent models. |
| `site:arxiv.org higher order recurrent states MQAR` | Check newer memory-capacity alternatives | RunningTensor record; treated as adjacent, not direct overlap. |

Backward chasing started from Zoology, Fast Weight Programmers, GLA, GDN, KATA, How Linear Attention Remembers, and Bayesian Layers. Forward chasing was limited to records and official links discoverable through arXiv related-paper pages and the above searches on the cutoff date; it is not a complete citation-database export.

## Candidate ledger

Evidence levels: **A** = primary full text/official proceedings inspected; **B** = primary abstract plus official paper page inspected; **C** = discovery-only or adjacent and not used for a strong claim. “Include” means it materially informs overlap or framing; “exclude” means it was not used as evidence for the project’s exact question.

| Record | Main evidence relevant to this project | Overlap with our study | Evidence / decision |
|---|---|---|---|
| Arora et al., *Zoology: Measuring and Improving Recall in Efficient Language Models* (2023/ICLR 2024), [arXiv](https://arxiv.org/abs/2312.04927) | Defines MQAR, multiple key-value associations and query positions; analyzes fixed-state recall and scaling. | Direct task overlap. | A — include; MQAR is established, not ours. |
| Hsieh et al., *RULER: What’s the Real Context Size of Your Long-Context Language Models?* (2024/COLM), [arXiv](https://arxiv.org/abs/2404.06654) | Flexible synthetic long-context benchmark extending NIAH with multiple retrieval and non-retrieval tasks. | Evaluation-family overlap, not exact MQAR/overwrite protocol. | A — include as context; do not claim long-context stress testing is new. |
| Yang et al., *Gated Linear Attention Transformers with Hardware-Efficient Training* (2023/ICML 2024), [arXiv](https://arxiv.org/abs/2312.06635) | Data-dependent gated linear attention; length generalization and efficient recurrent state. | Architecture/gate foundation. | A — include; establishes dynamic gating context. |
| Schlag et al., *Linear Transformers Are Secretly Fast Weight Programmers* (ICML 2021), [PMLR](https://proceedings.mlr.press/v139/schlag21a.html) | Fast-weight view, delta-like correction and dynamic learning rates; synthetic retrieval. | Memory/update rationale. | A — include; selective associative correction is prior. |
| Schlag et al., *Learning Associative Inference Using Fast Weight Memory* (2020/2021), [arXiv](https://arxiv.org/abs/2011.07831) | Fast-weight state representations compose facts for associative inference. | Broader associative-memory precedent. | B — include as historical context; not an exact GDN comparison. |
| Ba et al., *Using Fast Weights to Attend to the Recent Past* (NeurIPS 2016), [proceedings](https://proceedings.neurips.cc/paper/2016/hash/9f44e956e3a2b7b5598c625fcc802c36-Abstract.html) | Temporary memories implemented with rapidly changing fast weights. | Historical memory precedent. | A — include as background; no exact overlap. |
| Yang et al., *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* (NeurIPS 2024), [arXiv](https://arxiv.org/abs/2406.06484) | Efficient parallel delta-rule training and retrieval motivation. | Training implementation and delta recurrence. | B — include; our executor is not a first parallel delta implementation. |
| Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule* (2024/2025), [arXiv](https://arxiv.org/abs/2412.06464) | Gated decay combined with delta-rule writes. | Direct base architecture. | B — include; recurrence is prior. |
| Hatamizadeh et al., *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention* (2026), [arXiv](https://arxiv.org/abs/2605.22791) | Channel-wise erase and write gates; evaluates retrieval and overwrite-like memory behavior at scale. | Direct architecture family and update/editing concern. | B — include; channel-specific control and update robustness are active topics. |
| Feng et al., *DAMI: Dual-Axis Memory Initialization for Enhanced Long-Context Retrieval in Gated Linear Attention* (2026), [arXiv HTML](https://arxiv.org/html/2610.41994v2) | Activation-sample joint initialization of timescales and sensitivities; GDN-family retrieval. | Direct overlap with our planned calibration experiment. | A — include; our launcher is an attributed one-head adaptation, not a new method. |
| Das, *Delta-Rule Writes in Linear Attention: A Controlled Factorial Study of Decay Design* (2026), [TMLR page](https://tmlr.infinite-conf.org/paper_pages/FdXB4u27LD.html), [paper](https://openreview.net/pdf?id=FdXB4u27LD) | Scalar/channel-wise, static/data-dependent, delta/non-delta factorial cells; three seeds; synthetic recall; initialization ablations. | **Very high direct overlap** with granularity and static/dynamic decay. | A/B — include; blocks “first scalar-vs-channel study” and “first static-channel evidence.” |
| Ghriss and Chakraborty, *Kernelized Linear Attention: Breaking the Capacity Wall with Symmetric Cones* (2026), [arXiv HTML](https://arxiv.org/html/2607.17419) | Explicit MQAR length extrapolation and repeated-key overwrite; compares GDN and fixed-state alternatives. | **Very high task overlap**; latest-write evaluation and load/context stress. | A — include; broad task pairing and overwrite benchmark are prior. |
| Lee et al., *How Linear Attention Remembers* (2026), [arXiv HTML](https://arxiv.org/html/2609.33093) | Controlled causal interventions on writes, retention, reads, multi-fact coupling and editability in GLA/GDN; load harms recall more than elapsed context in tested regime. | **High mechanistic overlap**; our load-vs-length pattern is corroborative. | A — include; our decay-specific small-model intervention is narrower. |
| Dowling et al., *Memory by Design: Bayesian Layers for Sequence Modeling* (2026 v5), [arXiv](https://arxiv.org/abs/2605.31163) | Covariance-aware memory, collision dynamics, learned associative recall, MQAR and controlled robustness beyond training regime. | Memory-interference and robustness context. | B — include; alternative memory mechanism, not exact gating experiment. |
| Gangavarapu et al., *Muir: Linear Attention with Sensitivity-Allocated Gating and Matched-Filter Readout* (2026), [arXiv HTML](https://arxiv.org/html/2611.08376) | Channel-wise gates, sensitivity allocation, multi-query recall/overwriting analysis, gate rank and counterfactual freezing/static-mean interventions. | High adjacent intervention overlap; its gates are not exactly our decay alpha. | A — include; generic gate-freezing is not novel. |
| Cerruti et al., *Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing* (2026), [arXiv](https://arxiv.org/abs/2607.07953) | Matched comparisons of DeltaNet, GDN, KDA, GDN-2 and routing, including loss and throughput. | Comparative recipe context. | B — include as recent synthesis; not a synthetic task duplicate. |
| Okpekpe and Orvieto, *Revisiting Associative Recall in Modern Recurrent Models* (2025), [arXiv](https://arxiv.org/abs/2508.19029) | Reports strong sensitivity of recurrent models to narrow learning-rate windows in MQAR-style evaluation. | Training-recipe/learnability confound. | B — include; supports interpreting seed/optimization effects, not mechanism novelty. |
| Herranz-Celotti and Guigue, *RunningTensor: Generalizing Linear Attention to Higher-Order Recurrent States* (2026), [arXiv](https://arxiv.org/abs/2609.12814) | Higher-order recurrent state tested on synthetic multi-query associative recall against linear-attention/SSM baselines; capacity alternative. | Task/capacity context only. | B — include adjacent; not a gate experiment. |
| Giorlandino, Goldt, and Maillard, *Factual Recall in Linear Associative Memories: Sharp Asymptotics and Mechanistic Insights* (2026), [arXiv](https://arxiv.org/abs/2605.10795) | Theoretical capacity and interference for linear associative memories. | Explains why load can cause collisions; no GDN gate comparison established here. | B — include cautiously; abstract and primary metadata inspected, but do not rely on detailed numerical claims without full-text review. |
| Katharopoulos et al., *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention* (ICML 2020), [arXiv](https://arxiv.org/abs/2006.16236) | Kernelized linear attention as recurrent state with constant-memory inference. | Foundational recurrent-memory formulation. | A/B — include background; no overwrite-specific claim. |
| *Finetuning Pretrained Transformers into RNNs* (2021), [arXiv](https://arxiv.org/abs/2103.13076) | Recurrent reformulation of attention and associative state for inference. | General recurrent-state context. | B — include background; not a direct task/gate match. |
| *Revisiting Associative Recall in Modern Recurrent Models* search-adjacent papers and non-primary summaries | Several records use “associative recall” differently. | Terminology can create false matches. | C — exclude from strong claims unless full primary text is inspected. |

## Findings for our exact claims

### MQAR

MQAR is established by Zoology and has been reused in subsequent architecture and capacity work. KATA explicitly combines MQAR with length/load extrapolation and GDN comparisons. A paper cannot claim to introduce MQAR or be first to use it for GDN memory analysis. Our implementation can still be valuable as a carefully audited replication with a different data recipe, model size, seeds, and answer-free evaluation.

### Repeated-key overwrite

Latest-write-wins evaluation is established. KATA writes each key twice and asks for the latest value, with larger key/context extrapolation. GDN-2 and related delta-rule work explicitly motivate separating erase and write to protect unrelated associations. Our three-rewrite condition and explicit separation of newest-value accuracy, obsolete-value errors, and unrelated retention are a distinct diagnostic decomposition, but “we test overwrite” is not new.

### Load versus elapsed context

Our stress result—more associations hurt more than length alone—is consistent with How Linear Attention Remembers and KATA. It should be reported as replication/cross-protocol evidence. The stronger contribution would be to show whether the effect changes under matched decay initialization or channel intervention, with fresh test data and prespecified controls.

### Obsolete-value errors

Our post-hoc result finds 926/1065 latest-value errors returning the first replacement across the twelve models evaluated in the three-rewrite case. The reviewed papers establish that fixed-state memory has interference and latest-write challenges, but I did not find this exact four-way error breakdown (initial, first rewrite, second rewrite, other) in the inspected sources. This is a plausible diagnostic extension, not a guaranteed novel phenomenon. To strengthen it, train with varied rewrite counts and test an unseen rewrite count; otherwise it may be a training-distribution shortcut.

### Gate interventions

How Linear Attention Remembers and Muir establish causal/interventional memory analyses and static-mean/frozen-gate controls. Our intervention targets the decay alpha in a small matched-recipe checkpoint and contrasts shared mean, per-channel static values, coordinate shuffles, and layerwise replacement. This is a narrower implementation and protocol extension. It does not establish that static decay is sufficient in general, nor that token dependence is unnecessary.

### Initialization and calibration

DAMI directly occupies activation-sample joint timescale/sensitivity initialization. The older mean-decay calibration and new joint launcher must be framed as reproduction/adaptation and controls. A negative result cannot be presented as refuting DAMI because our model is one-head, four-layer, one small synthetic recipe and differs from DAMI’s multihead/scale/training setup.

## Comparison to our results

| Our result | What prior art already explains | What remains potentially useful |
|---|---|---|
| Clean MQAR near-perfect across confirmed seeds | MQAR learnability and fixed-state recall are established; scalar/channel outcomes are protocol-sensitive in Das. | A reproducible cross-protocol comparison under matched realized decay. |
| Default scalar seed 44 collapses at 8 associations | Fixed-state collision/load limits are established by Zoology, KATA, Bayesian-memory work. | Per-memory ordinal failure signature and calibration-dependent recovery, if replicated on fresh tests. |
| Longer context alone is tolerated; load hurts | Consistent with KATA and How Linear Attention Remembers. | Whether decay calibration/gate heterogeneity shifts the load boundary. |
| Three-rewrite latest-value recall is 52–78% by condition; errors mostly first rewrite | Latest-write and interference are established; delta-rule correction is prior. | Quantified obsolete-value taxonomy and training-distribution/generalization test. |
| Fixed per-channel decay preserves MQAR probes; shared mean harms some | Static channel structure and channel-gate interventions already exist in Das/Muir. | Small-model causal comparison isolating channel heterogeneity versus input dependence, with fair parameter matching. |

## Recommended evidence needed before a novelty claim

1. Cite and reproduce or explicitly scope against Das’s factorial cells. Include a parameter-matched channel control; current channel-wise models have about 4.17% more parameters.
2. Treat DAMI as the published calibration baseline. Separate default, mean-alpha matching, mean-log-decay matching, and joint sensitivity/timescale calibration if claiming sensitivity causes an effect.
3. Use untouched test generators and prespecified load, distance, and rewrite-count cells. Report per-seed values and uncertainty rather than only means.
4. Train at least one condition with varied rewrite counts. Test unseen rewrite counts to distinguish a learned single-rewrite shortcut from a recurrence limitation.
5. Preserve the answer-free protocol and report unrelated-key retention separately from latest-value recall; a single aggregate score can hide obsolete-value behavior.
6. Avoid claims such as “first channel-wise GDN study,” “first MQAR/overwrite comparison,” “first gate-freezing analysis,” or “new joint memory initialization.” Those are contradicted by the records above.

## Limits and audit trail

The search included broad historical fast-weight and linear-attention terms, MQAR, repeated-key/latest-write, long-context/RULER, load/interference, causal interventions, and gate-initialization terms. It used web-indexed primary sources and citation chasing from the closest papers on 2026-11-17. It did not query subscription databases, every national index, every thesis/dissertation repository, or unpublished manuscripts. Search results can omit papers with different terminology. Several records were checked at abstract level only; the closest overlap records (Das, DAMI, KATA, How Linear Attention Remembers, Muir) were checked at full-text/section level where accessible.

**Conclusion:** the broad task and mechanism ideas are occupied. The defensible opportunity is a small-model replication/boundary study explaining when matched decay, channel heterogeneity, and activation dependence alter the difference between clean retrieval and repeated-update generalization. No exhaustive search can turn that into a guaranteed new method; it can only make the scope and novelty claim auditable.

## Supplementary catalog-native screening (17 November 2026)

The following records were added after an extended catalog search. I opened the primary record where possible and recorded inaccessible records explicitly rather than treating catalog metadata as evidence.

### Anatomy of Associative Recall in Fixed-State Recurrences

Primary: Boesch and Wee, *Anatomy of Associative Recall in Fixed-State Recurrences: A Matched-State Decomposition, an Interference Wall, and a Curriculum That Breaks It* ([arXiv abstract](https://arxiv.org/abs/2609.16183), [full HTML](https://arxiv.org/html/2609.16183), arXiv:2609.16183, submitted 14 September 2026). Evidence level A.

The paper is very close to our failure-analysis question. It decomposes fixed-state recall into short causal convolution, transition structure (rank-1 delta versus diagonal), and decay. In its matched-state experiments, the convolution accounts for roughly a half-point recall gain, while transition rank contributes less once the convolution is present. Its masked multi-query recall/distractor-haystack tests show graceful load degradation in the basic load setting but a flat, chance-level distractor-haystack regime. Under sparse supervision, the authors observe seed-dependent lock-in: in their reported experiment, only 1/10 uniform trials lock in versus 7/10 under another condition (reported as p=.02); a distance curriculum succeeds in 4/5 trials at length 256 where uniform training succeeds in 0/9, and a success-gated curriculum succeeds in 6/6 at length 512 (reported as p=.001). They also report that collision-key retrieval needs two layers and that causal versus bidirectional variants do not materially change the relevant result in that setup.

Overlap is direct: MQAR-like masked multi-query recall, load/interference, fixed-state recurrence, transition/decay decomposition, and training-coverage interventions are all now documented in a primary source. This paper is especially important for interpreting our three-seed means: a low score can be a seed/training-coverage lock-in event rather than evidence of an architectural capacity limit. Our obsolete-value error taxonomy and calibrated scalar-versus-channel comparison may still be a useful protocol extension, but it must include per-seed lock-in diagnostics, varied rewrite counts, and a curriculum or training-coverage control before it is described as a mechanism discovery.

### Variational Linear Attention: Stable Associative Memory for Long-Context Transformers

Primary: Pandey and Singh, *Variational Linear Attention: Stable Associative Memory for Long-Context Transformers* ([arXiv abstract](https://arxiv.org/abs/2605.11196), [full HTML](https://arxiv.org/html/2605.11196), arXiv:2605.11196, submitted 11 May 2026). Evidence level A.

The paper proposes online regularized least-squares updates with an adaptive penalty and a fixed matrix state. On synthetic MQAR, it reports near-perfect recall for eight pairs across lengths 64--512 and substantially better load behavior than DeltaNet; the reported overload setting reaches about 62% at capacity. The authors attribute the contrast to scalar DeltaNet decay suppressing older associations uniformly, while the matrix state preserves directions selectively. The experimental scope is narrow (synthetic MQAR, not natural-language modeling), and the paper itself notes that capacity remains finite.

Overlap is high for the load axis and the scalar-decay explanation, but the mechanism is different: this is a new matrix-state update, not a scalar/channel GDN intervention. It prevents a claim that our scalar-versus-channel load/retention question is the first investigation of direction-selective associative retention. It supports framing our work as a controlled small-model boundary/diagnostic study of when channel heterogeneity helps, not as the first demonstration that uniform scalar decay can erase old associations.

### Constant-Memory Recall: Learned Associations in a Fixed Matrix State

Primary: Larson, *Constant-Memory Recall: Learned Associations in a Fixed Matrix State* ([arXiv abstract](https://arxiv.org/abs/2610.00232), [full HTML](https://arxiv.org/html/2610.00232), arXiv:2610.00232, submitted 23 September 2026). Evidence level A.

This is a close small-model recall study. It reports a DeltaNet variant with token-specific key biases and a 32 KiB matrix state, trained to remember 32 key-value pairings, with 99.95% mean accuracy over three seeds. Filler-only contexts preserve recall through 454, 902, and 1,798 tokens. A 48-pair exploratory setting is near chance, but it used only one quarter of the primary training budget, so the authors explicitly leave capacity versus under-training unresolved. The paper also uses state ablations and ridge probes, and limits its claim to a small synthetic setup with three seeds.

Overlap is direct for fixed-state associative recall, three-seed reporting, filler-distance retention, and a capacity/under-training ambiguity. It does not use our exact scalar/channel GDN, activation-sample decay calibration, or three-rewrite obsolete-value breakdown. The appropriate novelty claim is therefore a focused comparison of calibration and channel heterogeneity under clean MQAR versus repeated-update generalization, not “first small fixed-state recall study.”

### Latent Recall Fragility in Kimi Delta Attention

Catalog record: *Latent Recall Fragility in Kimi Delta Attention: A Conditional Dynamical Analysis*, DOI [10.5281/zenodo.23270559](https://doi.org/10.5281/zenodo.23270559), OpenAlex W7221107996. Evidence level C / unresolved. The catalog supplied a title and DOI, but the DOI and direct Zenodo record ([record URL](https://zenodo.org/records/23270559)) were inaccessible in the primary-source browser, and exact-title searches did not surface a readable primary paper. I therefore did not use any purported result from this record. It remains a candidate for manual follow-up, not evidence for or against our novelty claim.

### Extended-catalog rows 526, 528, and 530

The extended catalog returned three additional close-looking records. The first two were not verifiable as primary full text in this pass:

| Catalog record | Primary URL | Screening result |
|---|---|---|
| *Similarity Finds the Fact, Not the Version: A Co-Trained Order Stamp in the Key Resolves Version Conflict in Persistent Model-Internal Memory* (2026) | [Research Square DOI](https://doi.org/10.21203/rs.3.rs-10896018/v1) | DOI/landing page inaccessible; exact-title search did not yield a readable primary. C, unresolved; no claims used. |
| *The Query Cannot See the Question: A Short Convolution's Reach Decides Which Part of a Query Conditions Retrieval, and What Falls Outside Becomes a Confident Wrong Answer* (2026) | [Research Square DOI](https://doi.org/10.21203/rs.3.rs-10934426/v1) | DOI/landing page inaccessible; exact-title search did not yield a readable primary. C, unresolved; no claims used. |
| Tong et al., *On the Importance of Gating: Memorization vs. In-Context Learning in State Space Models* (2026) | [arXiv abstract](https://arxiv.org/abs/2609.16540), [full HTML](https://arxiv.org/html/2609.16540) | A, full primary HTML inspected. Included below. |

The Tong et al. paper is highly relevant to our training interpretation. It proves and tests a competition between in-weights memorization and in-context retrieval in gated SSMs. Its synthetic multi-token retrieval uses a fixed dictionary, perturbed value tokens, key-value lists, and an additive initialization offset to the decay parameter. Stronger gating can leave models on a memorization plateau for the full reported training window, while weaker gating escapes faster; the same stronger gating can improve length generalization by suppressing accumulated distractors. The paper reports an explicit escape-time scaling with initial retrieval factor and notes that initial gate ordering remains informative after training because gate drift is often modest. Its multi-token task uses five seeds, a 512-entry dictionary limit, and a default Mamba setup; its reported full-scale recipe is 24-layer Mamba-2 at 768 hidden size, AdamW, peak learning rate 3e-4, 5% warmup, cosine decay, and up to 100k steps.

This is direct overlap with our decay-offset/calibration question and with the possibility that our retrieval failures are optimization shortcuts rather than a memory-capacity failure. It means we must not present decay initialization, gate offsets, or “training versus architecture” separation as new in isolation. It also gives a concrete control for our paper: randomize or perturb values/keys so memorization cannot solve the task, report an escape-time or first-success metric, and separate in-distribution clean recall from unseen-key/unseen-rewrite retrieval. The paper’s trade-off supports our planned load/distance grid: weak decay helps discover retrieval, but too little decay accumulates distractor variance; stronger decay improves long-context generalization while harming distant retrieval.

### Supplementary access log

| Date | Query/access | Result |
|---|---|---|
| 2026-11-17 | Opened arXiv 2609.16183 abstract and HTML | Accessible; full sections on decomposition, interference, and curriculum inspected. |
| 2026-11-17 | Opened arXiv 2605.11196 abstract and HTML | Accessible; MQAR load and DeltaNet comparison inspected. |
| 2026-11-17 | Opened arXiv 2610.00232 abstract and HTML | Accessible; three-seed recall, filler, 48-pair caveat, and limitations inspected. |
| 2026-11-17 | Opened DOI/Zenodo 10.5281/zenodo.23270559 | Inaccessible in the browser; catalog-only. |
| 2026-11-17 | Opened Research Square 10.21203/rs.3.rs-10896018/v1 and 10.21203/rs.3.rs-10934426/v1 | Both inaccessible in the browser; catalog-only. |
| 2026-11-17 | Opened arXiv 2609.16540 abstract and HTML | Accessible; theory, multi-token retrieval, offset experiments, and gate-drift appendix inspected. |

### Supplementary decision

The new closest evidence strengthens—not weakens—the cautious framing already above. The strongest competing explanation for the project’s retrieval failures is now testable training dynamics: gate initialization can create a memorization shortcut or lock-in, and short/fixed training can make a capacity-limited model look broken. Before a full 3-seed, eight-model run is treated as a paper result, include (i) unseen key/value or perturbed-value evaluation, (ii) varied and unseen rewrite counts, (iii) per-seed escape/lock-in reporting, and (iv) a load-versus-distance grid. If those controls show a reproducible scalar/channel difference after the model is forced to use context, the contribution can be framed as a calibrated, mechanistic boundary study. Without them, the current result is an informative replication/diagnostic, not a novelty guarantee.

## Bibliographic identifier audit (17 November 2026)

The ledger was checked against each arXiv landing page rather than trusting inherited identifiers. One mismatch was corrected: arXiv **2009.06732** is *Efficient Transformers: A Survey* by Tay et al., not Katharopoulos et al.; the correct primary record for *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention* is arXiv **2006.16236** ([verified primary](https://arxiv.org/abs/2006.16236)). The other checked identifiers resolve as follows: arXiv **2605.10795** is *Factual recall in linear associative memories: sharp asymptotics and mechanistic insights* by Alessio Giorlandino, Sebastian Goldt, and Antoine Maillard ([verified primary](https://arxiv.org/abs/2605.10795)); arXiv **2609.12814** is *RunningTensor: Generalizing Linear Attention to Higher-Order Recurrent States* by Luca Herranz-Celotti and Vincent Guigue ([verified primary](https://arxiv.org/abs/2609.12814)); arXiv **2607.07953** is *Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing* by Tommaso Cerruti, Tim Rieder, George Rowlands, Lingfeng Jin, and Imanol Schlag ([verified primary](https://arxiv.org/abs/2607.07953)); and arXiv **2103.13076** is *Finetuning Pretrained Transformers into RNNs* by Jungo Kasai et al. ([verified primary](https://arxiv.org/abs/2103.13076)). No unsupported Ghriss/Chakraborty attribution remains for RunningTensor.
