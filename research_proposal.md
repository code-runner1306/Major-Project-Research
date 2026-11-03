# Does Fine-Grained Gating Generalize Below Scale?
### A Controlled, Multi-Scale Study of Scalar vs. Channel-Wise Gating in Gated DeltaNet Language Models

**Final Year Engineering Major Project — Research Proposal**
Department of Computer Engineering

**Team of 3**
- Mayank Mehta (10199)
- Michelle Aloz (10165)
- Amrita Pati (10210)

**Guide:** [Faculty Guide Name]
**Date:** July 2026

---

## Abstract

Recent efficient-attention language model architectures — most notably Qwen3-Next and Kimi Linear — extend the Gated DeltaNet mechanism with a data-dependent forgetting gate, but disagree on how fine-grained that gate should be: Qwen3-Next uses one scalar decay value per attention head, while Kimi Linear (via Kimi Delta Attention) and the more recent Gated DeltaNet-2 use a separate decay value per feature channel, reporting improved long-context and retrieval performance as a result. These comparisons, however, are made between full production-scale models (multi-billion parameters) that differ in many respects beyond the gate, and no published work has isolated gating granularity as the sole variable in a controlled experiment, nor examined whether its benefit holds at the much smaller scales (10M–100M parameters) accessible to compute-constrained researchers.

This project trains matched pairs of small Gated DeltaNet language models — identical in every respect except gate granularity — across four parameter scales, on the TinyStories dataset, and measures validation perplexity, latency, and (as a secondary probe) simple long-context retrieval accuracy. The project produces one of two clean, publishable outcomes: either the accuracy advantage of channel-wise gating holds even at small scale (supporting broader adoption of Kimi/GDN-2-style gating in resource-constrained settings), or it is scale-emergent and not worth its latency cost below a certain size (a useful negative result for practitioners deciding where to spend their compute).

---

## 1. Introduction

Transformer-based language models rely on self-attention, whose cost grows quadratically with sequence length. Over the last two years, a family of "linear" or "sub-quadratic" attention mechanisms has emerged to address this — replacing the growing key-value cache with a small, fixed-size recurrent state that is updated as the model reads. Gated DeltaNet [1] is one of the most influential of these mechanisms: it combines a data-dependent forgetting gate (borrowed from gated linear recurrences such as Mamba-2) with the delta rule (a corrective, error-driven memory update), and has since been adopted, in hybrid form alongside a small number of full-attention layers, by production language models including Qwen3-Next [3][4].

A specific design question inside this mechanism has become a live point of disagreement between two of the labs building these models. The original Gated DeltaNet and Qwen3-Next use a single scalar gate per attention head — one "forget dial" shared across an entire head's feature vector. Kimi Linear's Kimi Delta Attention (KDA) [2] instead uses a channel-wise gate — a separate forget dial for every individual feature — and reports that this finer control is part of why it is, by the authors' account, the first linear-attention hybrid to match or exceed full attention under a fair comparison. A subsequent NVIDIA paper, Gated DeltaNet-2 [5], goes further, explicitly building on "Kimi Delta Attention (KDA), [which] sharpens forgetting with channel-wise decay," and proposing an even more decoupled gating scheme.

What none of these papers do — because it is not their purpose — is isolate gate granularity as the only variable, under matched data, model size, training budget, and seeds, and ask how the effect changes as model scale shrinks. This is a real, practical, and currently unanswered question for two reasons. First, scaling behaviour is not always monotonic or scale-invariant in efficient-attention research generally — a recent 2026 study of hybrid architectures found that the interaction between attention window size and full-attention layers changes qualitatively with configuration, a phenomenon the authors term "Large-Window Laziness" [6], underscoring that findings established at one configuration or scale do not automatically transfer to another. Second, a large population of researchers, students, and practitioners without frontier-lab compute budgets need to know whether adopting the more complex, more expensive channel-wise gate is worth it at the scales they can actually afford to train and deploy.

This project answers that question directly: we implement both gating variants inside an otherwise identical small Gated DeltaNet architecture, train matched pairs of models at four parameter scales (approximately 10M, 30M, 55M, and 100M parameters) on the TinyStories dataset [7] — a dataset purpose-built for evaluating small language models coherently — and report perplexity, inference latency, and a simple retrieval probe across scale, with multiple random seeds per configuration for statistical reliability.

---

## 2. Literature Review

### 2.1 The Efficiency Motivation: From Quadratic Attention to Linear Recurrence

Standard self-attention scores every token against every other token, giving O(n²) time and an ever-growing key-value cache at inference time. Linear attention formulations reframe this as a recurrent state update with a fixed-size state matrix, giving O(n) time and O(1) memory per step during decoding. Early linear-attention formulations, however, struggled on retrieval-heavy and long-context tasks relative to full attention, motivating a line of work aimed at recovering full-attention-level quality while keeping linear-time computation.

### 2.2 The Delta Rule and Gated DeltaNet

DeltaNet reframes the recurrent state update as an online learning problem: rather than additively writing new key-value associations into the state, it corrects the state based on the error between what is currently stored and the new target value, which was shown to improve targeted memory updates. Yang, Kautz, and Hatamizadeh [1] observed that the delta rule and gating are complementary — gating enables rapid, wholesale memory erasure, while the delta rule enables precise, targeted updates — and combined them into Gated DeltaNet, which "consistently surpasses existing models like Mamba2 and DeltaNet across multiple benchmarks, including language modeling, common-sense reasoning, in-context retrieval, length extrapolation, and long-context understanding" [1]. Critically for this proposal, Gated DeltaNet's gate is a single scalar per head: one decay value αₑ ∈ (0,1) applied uniformly across the whole feature vector at that step.

This mechanism has since been adopted in hybrid form — interleaved with a smaller number of full self-attention layers — by production open-weight models. Qwen3-Next stacks three Gated DeltaNet layers for every one full-attention layer [4][9], and uses the same head-wise scalar gating scheme as the original formulation, additionally incorporating a gated-attention variant on its full-attention layers that has been associated with reduced attention-sink behaviour and improved training stability [3][8].

### 2.3 Channel-Wise Gating: Kimi Linear and Gated DeltaNet-2

Kimi Linear [2], released by the Kimi Team in October 2025, introduces Kimi Delta Attention (KDA), which replaces Gated DeltaNet's scalar-per-head gate with a channel-wise gate — a separate decay value for every feature dimension within a head, rather than one shared value for the whole head. The Kimi Team frames this as giving the model much finer control over what it forgets, and report state-of-the-art results for a linear-attention hybrid relative to full attention under matched training conditions.

The most directly relevant related work is Gated DeltaNet-2 [5] (Hatamizadeh, Choi, and Kautz, 2026), which explicitly starts from the observation that "Kimi Delta Attention (KDA) sharpens forgetting with channel-wise decay" but notes that KDA still ties erasing and writing together through a single gate value, and proposes decoupling them into a separate channel-wise erase gate and channel-wise write gate, reporting that the advantage is "most pronounced on long-context RULER needle-in-a-haystack benchmarks." This paper is important to acknowledge directly: it already establishes, empirically, that channel-wise gating outperforms scalar gating, and that a further-decoupled channel-wise scheme outperforms both. This substantially narrows the novelty available to this project — the qualitative claim "channel-wise beats scalar" is no longer open. What remains genuinely untested, and is the specific gap this proposal targets, is whether that ordering holds at small scale (10M–100M parameters) under a controlled, single-variable comparison, since Gated DeltaNet-2, Kimi Linear, and the original Gated DeltaNet paper all report their central results at much larger scale and as part of broader architectural packages rather than as a clean, isolated ablation across a size sweep.

### 2.4 Scale-Dependent Effects in Hybrid Efficient-Attention Architectures

There is direct precedent for architectural findings in this space not transferring cleanly across configurations or scale. Qiao et al. [6] (June 2026) show that in hybrid models mixing full attention with sliding-window attention, larger window sizes delay the formation of "retrieval heads" in the full-attention layers — a phenomenon they term "Large-Window Laziness" — and that a fix effective in one configuration (applying no positional encoding only to the full-attention layers) does not behave identically across all window sizes and scales. This is used in the present proposal as direct motivation: if window-size effects in a structurally similar hybrid-attention family are already known to be configuration- and scale-sensitive, gating-granularity effects reported only at large scale should not be assumed to hold unchanged at small scale without being tested.

### 2.5 Small-Scale Language Modeling as a Controlled Testbed

Eldan and Li's TinyStories [7] demonstrated that models far smaller than contemporary state-of-the-art (below 10 million parameters in some configurations) can produce fluent, coherent English when trained on a vocabulary-constrained synthetic dataset of short children's stories, and that this setting is sufficient to distinguish meaningfully between architectural choices without requiring internet-scale data or compute. This makes TinyStories a natural, well-precedented testbed for isolating a single architectural variable — such as gate granularity — under a training budget realistic for a student team, while still producing a signal (perplexity, coherence) that is comparable in spirit to what large-scale papers report.

### 2.6 Summary of the Literature and the Specific Gap

Table 1 summarizes how the closest prior work relates to this proposal.

| Work | What it establishes | What it leaves open |
|---|---|---|
| Gated DeltaNet [1] | Introduces scalar, head-wise gating combined with the delta rule; strong results vs. Mamba2/DeltaNet. | Does not test channel-wise gating. |
| Qwen3-Next / Qwen3 [3][4][9] | Deploys scalar gating at production (multi-billion parameter) scale in a hybrid architecture. | No small-scale or ablation comparison of gate granularity. |
| Kimi Linear (KDA) [2] | Introduces channel-wise gating; reports it matches/exceeds full attention at large scale. | Compared as part of a full new architecture, not as an isolated variable; only large scale tested. |
| Gated DeltaNet-2 [5] | Confirms channel-wise gating beats scalar gating; proposes further decoupling erase/write gates; strongest gains on long-context retrieval. | Again evaluated as part of a broader architectural change, at production scale; no systematic small-scale sweep. |
| Large-Window Laziness [6] | Shows hybrid-attention findings can be configuration- and scale-sensitive in a structurally related architecture family. | Does not study gating granularity directly; motivates why scale-transfer should not be assumed. |
| TinyStories [7] | Establishes that small models trained on constrained data give a reliable, coherent signal for architectural comparison. | Not previously used to test gating granularity in Gated DeltaNet variants. |

Taken together, the literature establishes the direction of the effect (channel-wise beats scalar gating) but not its behaviour across scale, nor its cost-benefit trade-off in the compute-constrained regime most students, researchers, and edge/on-device deployments actually operate in. That is the gap this project fills.

---

## 3. Research Questions and Objectives

**Primary research question:**

- **RQ1:** Does the accuracy advantage of channel-wise gating over scalar gating in Gated DeltaNet hold at small parameter scales (10M–100M), or is it scale-emergent?

**Secondary research questions:**

- **RQ2:** How does the size of any gating-granularity advantage change as model scale increases from 10M to 100M parameters (i.e., is the gap growing, shrinking, or flat)?
- **RQ3:** What is the latency/throughput cost of channel-wise gating relative to scalar gating at each scale, and at what point (if any) does the accuracy benefit justify that cost?
- **RQ4 (stretch):** Does the accuracy gap on a simple synthetic long-context retrieval probe mirror the perplexity gap, or do the two metrics diverge?
- **RQ5 (post-training diagnostic):** How sensitive is a trained channel-wise model to sharing its decay values across increasingly large channel groups at evaluation time?

**Objectives:**

- Implement a small, from-scratch Gated DeltaNet language model with a swappable gating module (scalar vs. channel-wise), building on an existing open-source reference implementation rather than writing the mechanism from a blank page.
- Train matched pairs of models (same data, optimizer, schedule, and seeds) at four scales, varying only the gate.
- Evaluate all models on validation perplexity, wall-clock inference latency/throughput, and a simple synthetic retrieval benchmark.
- For each existing channel-wise checkpoint, evaluate post-hoc gate sharing at several group counts without retraining, using the same validation and held-out retrieval examples.
- Report results with multiple seeds per configuration and honest statistical treatment (mean ± standard deviation, not single-run numbers).
- Write up the result — whichever direction it points — as a clear, honestly-scoped empirical finding suitable for a workshop-tier submission.

---

## 4. Methodology

### 4.1 Base Architecture

We build on the from-scratch, educational Gated DeltaNet implementation in the open-source repository rasbt/LLMs-from-scratch [10], which provides a working, dependency-light PyTorch implementation without requiring custom CUDA/Triton kernels. This is a deliberate choice for feasibility: it lets the team focus engineering effort on the gating variable itself rather than on reproducing a hardware-efficient chunkwise training algorithm from the original paper.

### 4.2 Independent Variable: Gating Granularity

The single manipulated variable across both model variants is the shape of the forget-gate parameter:

- **Variant A (Scalar / Qwen3-Next-style):** one gate value αₑ per attention head per timestep, broadcast across all feature channels in that head, following the original Gated DeltaNet / Qwen3-Next formulation [1][4].
- **Variant B (Channel-wise / Kimi-style):** one independent gate value per feature channel per timestep within each head, following the Kimi Delta Attention formulation [2].

Everything else — layer count, hidden dimension, number of heads, hybrid ratio of Gated DeltaNet to full-attention layers, optimizer, learning-rate schedule, tokenizer, and random seeds — is held identical between the two variants at each scale, so that any measured difference is attributable to gating granularity alone.

### 4.3 Model Scales

Four parameter scales will be trained, chosen to span a range large enough to reveal a trend while remaining trainable on free-tier or single-GPU compute:

- ~10M parameters (comparable to TinyStories' own smallest reported configurations [7])
- ~30M parameters
- ~55M parameters
- ~100M parameters

### 4.4 Dataset

Training and validation use TinyStories [7], a synthetic dataset of short stories restricted to vocabulary understood by a 3–4-year-old child, generated by GPT-3.5/GPT-4 and specifically designed and validated for training and evaluating small language models. This choice is deliberate: it lets small models reach a coherent, measurable signal within a realistic training budget, isolating the architectural comparison from confounds caused by simply under-training a small model on data that is too difficult for its capacity.

### 4.5 Training Protocol

- Matched optimizer (AdamW), learning-rate schedule, batch size, and total training tokens across both variants at a given scale.
- 3 random seeds per (scale × variant) combination → 24 total training runs (4 scales × 2 variants × 3 seeds), with a documented fallback to 3 scales × 2 variants × 3 seeds (18 runs) if compute or time is constrained — prioritizing seed count over scale count, since a consistent, well-replicated trend at fewer scales is more defensible than a noisy trend at more scales.
- Fixed evaluation checkpoints (e.g., matched total tokens seen) rather than matched wall-clock training time, so that any latency difference is measured separately and not conflated with training-time effects.

### 4.6 Implementation and Compute

- **Language/framework:** Python, PyTorch.
- **Compute:** free-tier or low-cost single-GPU resources (e.g., Google Colab, Kaggle Notebooks), split across three student accounts to parallelize the run queue.
- **Team roles:** one member owns the data/training pipeline, one owns implementing and validating the two gating variants, one owns evaluation, benchmarking methodology, and write-up coordination — with all three contributing to final analysis.

---

## 5. Experiments and Evaluation

### 5.1 Experiment 1 — Perplexity Across Scale (addresses RQ1, RQ2)

Train both gating variants at all four scales and report validation perplexity (mean ± std. dev. across 3 seeds) in a single comparison table and an accompanying line plot of perplexity gap vs. parameter count. This is the primary experiment and directly answers whether channel-wise gating's advantage, established at large scale in [2] and [5], is present, absent, or changing in magnitude at small scale.

### 5.2 Experiment 2 — Latency and Throughput Benchmarking (addresses RQ3)

For each trained model, measure inference latency (ms/token, decode phase) and throughput (tokens/second) under a fixed batch size and sequence length, with proper warmup iterations and repeated trials to reduce measurement noise. Report the latency cost of channel-wise gating relative to scalar gating at each scale, and combine with Experiment 1 to produce an explicit accuracy-per-millisecond comparison.

### 5.3 Experiment 3 (stretch goal) — Synthetic Long-Context Retrieval Probe (addresses RQ4)

Construct a simple needle-in-a-haystack style synthetic evaluation (insert a fact at a random position in a long, otherwise-irrelevant context and query the model to retrieve it) scaled appropriately to the small models' context length. This is included as a stretch goal, not a committed deliverable, since it adds evaluation-design complexity; if time allows, it lets us check whether perplexity and retrieval accuracy tell the same story or diverge, which would itself be an interesting finding.

### 5.4 Experiment 4 — Post-Training Gate-Grouping Sensitivity

This evaluation-only diagnostic reuses each trained channel-wise checkpoint; it adds no training runs and does not replace a dataset. After the channel-wise gate has been computed for each token, average its values within contiguous groups of channels and broadcast each group average back across those channels. Test one group (scalarization), a few intermediate group counts (for example 2, 4, and 8 where they divide the head dimension), and the full channel count. Use the same validation batches and the same fixed held-out MQAR and overwrite examples for every setting.

Report TinyStories validation loss/perplexity and retrieval accuracy for each group count, alongside the unmodified channel-wise result and the existing gate-diversity diagnostics. Interpret a drop as sensitivity of the trained model to removing channel differences. Because the model was trained with fully channel-wise gates, this intervention does not measure how a model trained from scratch with fewer gate groups would perform; it is a mechanistic diagnostic, not a new trained-model comparison. This check is secondary to the scalar-versus-channel-wise results.

### 5.5 Statistical Treatment

All headline results are reported as mean ± standard deviation across seeds, not single-run numbers. Where compute allows, we additionally report a simple significance test (e.g., a t-test across seeds) for the perplexity difference between variants at each scale, so that any claimed trend is not an artifact of run-to-run noise.

---

## 6. Expected Outcomes

The project is designed so that either direction of result is a complete, publishable finding:

- **Outcome A:** channel-wise gating's advantage holds even at 10M–100M scale. This extends the generality of the Kimi Linear / Gated DeltaNet-2 finding [2][5] down to a regime those papers did not test, and gives compute-constrained practitioners a validated reason to adopt the more complex gate.
- **Outcome B:** the advantage is scale-emergent — negligible or inconsistent below some threshold, and not worth its latency cost at small scale. This is an equally valid, and arguably more actionable, negative result: it tells small-model builders they can safely use the simpler, cheaper scalar gate.

In both cases, the deliverable is a clean table and plot of perplexity, latency, and (if completed) retrieval accuracy across scale and gate type, with honest, replicated statistics — exactly the kind of contribution workshop venues are designed to accept.

---

## 7. Target Venues

Based on the scope of this project — a controlled empirical ablation rather than a new method or large-scale validation — realistic targets are workshop and second-tier venues rather than flagship main tracks. Recommended targets, roughly in order of preference:

### 7.1 Primary Targets — Workshops at Major ML Conferences

- An efficient-ML / long-context / architecture-focused workshop co-located with NeurIPS, ICML, or ICLR (e.g., workshops on efficient deep learning, sequence modeling, or foundation model architecture). Workshops at these venues explicitly do not require prior publications and are described as selecting on "the quality of the research, rigorous experiments, and clear writing" rather than novelty of method — a good match for this project's strengths [11].
- Note that popular workshops can still be competitive — for example, the ICML 2026 Mechanistic Interpretability Workshop reported 2.6× the prior year's submissions [12] — so a specific, well-matched workshop with a clear efficient-architectures or small-models focus should be identified once results are in hand.

### 7.2 Secondary Targets — Journals and Smaller Venues

- Smaller AI/ML journals with a track record of accepting rigorous empirical studies, where a controlled ablation with proper statistical treatment is often valued even without a new method.
- Student or national-level CS research conferences, and your institution's own research symposium — realistic, safe outcomes for a first publication.

### 7.3 Stretch Target — Main Conference Tracks

Main tracks of NeurIPS, ICML, and ICLR accept roughly 20–33% of all submissions in recent years [13][14][15][16], but that pool is dominated by papers proposing new methods validated at scale — a category this project does not fall into. Main-track submission should be treated as a long shot, not a primary plan, regardless of the pool-level acceptance rate.

### 7.4 Recommended Path

Post a first version to arXiv as soon as results are stable (essentially guaranteed availability, and establishes priority/timestamp), then target one specific, well-matched conference workshop as the primary submission, with a smaller journal or student conference as a fallback if the workshop submission is not accepted.

---

## References

[1] Yang, S., Kautz, J., & Hatamizadeh, A. (2024/2025). *Gated Delta Networks: Improving Mamba2 with Delta Rule*. Published as a conference paper at ICLR 2025. https://arxiv.org/abs/2412.06464

[2] Kimi Team (2025). *Kimi Linear: An Expressive, Efficient Attention Architecture (Kimi Delta Attention)*. https://arxiv.org/abs/2510.26692

[3] Qiu, Z., Wang, Z., Zheng, B., et al. (2025). *Gated Attention for Large Language Models: Nonlinearity, Sparsity, and Attention-Sink-Free*. https://arxiv.org/abs/2505.06708

[4] Qwen Team (2025). *Qwen3 Technical Report*. https://arxiv.org/abs/2505.09388

[5] Hatamizadeh, A., Choi, Y., & Kautz, J. (2026). *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention*. https://arxiv.org/abs/2605.22791

[6] Qiao, et al. (2026). *Rethinking the Role of Efficient Attention in Hybrid Architectures*. https://arxiv.org/abs/2606.15378

[7] Eldan, R., & Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* https://arxiv.org/abs/2305.07759

[8] Raschka, S. (2026). *Gated Attention — The Big LLM Architecture Comparison* (blog / architecture gallery). https://sebastianraschka.com/llm-architecture-gallery/gated-attention/

[9] *A Persistent-State Dataflow Accelerator for Memory-Bound Linear Attention Decode on FPGA* (2026) — documents Qwen3-Next's 3:1 Gated DeltaNet-to-full-attention layer ratio. https://arxiv.org/pdf/2603.05931

[10] Raschka, S. rasbt/LLMs-from-scratch — open-source, from-scratch reference implementations including Gated DeltaNet (GitHub repository). https://github.com/rasbt/LLMs-from-scratch

[11] Algoverse AI Research (2026). *Beyond NeurIPS: A Student's Guide to ICML, ICLR, AAAI, and Other AI Conferences*. https://algoverseairesearch.org/blog/icml-iclr-aaai-student-guide

[12] Call for Papers, Mechanistic Interpretability Workshop at ICML 2026. https://mechinterpworkshop.com/cfp/

[13] International Conference on Machine Learning — Wikipedia (submission/acceptance statistics 2015–2026). https://en.wikipedia.org/wiki/International_Conference_on_Machine_Learning

[14] IntuitionLabs (2025). *NeurIPS 2025: A Guide to Key Papers, Trends & Stats*. https://intuitionlabs.ai/articles/neurips-2025-conference-summary-trends

[15] lixin4ever. Conference-Acceptance-Rate — Acceptance rates for major AI conferences (GitHub repository). https://github.com/lixin4ever/Conference-Acceptance-Rate

[16] *What Drives Paper Acceptance? A Process-Centric Analysis of Modern Peer Review* (2025) — ICLR 2017–2025 submission/acceptance statistics. https://arxiv.org/pdf/2509.25701
