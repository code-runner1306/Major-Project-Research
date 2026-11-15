# Learning-rate review for the corrected 1,000-step pilot

Date: 2026-11-13

## Decision

Use **0.001 as the peak AdamW learning rate** for the current corrected 10M
pilot, including the Kaggle 50-step timing smoke. In this repository the
training loop applies `OneCycleLR` with `div_factor=10`, `final_div_factor=10`,
and a 5% warmup. Therefore a 0.001 peak means an initial rate of about
0.0001, a peak near step 50 in the 1,000-step run, and a final rate near
0.00001. Keep this as a fixed protocol choice for the timing smoke and the
planned 1,000-step exploratory comparison; do not describe it as a universal
optimum or as a convergence guarantee.

## Evidence from primary papers

### Gated Delta Networks (Yang, Kautz, Hatamizadeh, 2024/2025)

The original Gated DeltaNet paper trains 1.3B-parameter models on 100B
FineWeb-Edu tokens with AdamW, a **4e-4 peak learning rate**, cosine decay,
1B-token warmup, weight decay 0.1, gradient clipping 1.0, and 0.5M-token
batches. This is the closest architecture-specific baseline, but its model,
token budget, warmup, and schedule are very different from a 10M model trained
for 1,000 optimizer steps. It supports a conservative sub-1e-3 scale for a
large production pretraining recipe; it does not determine the short-budget
small-model rate.

Source: https://arxiv.org/html/2412.06464 (see §4 Setup, lines 200–204).

### Zoology: Measuring and Improving Recall in Efficient Language (Akyürek et al., 2023)

This is directly relevant to the synthetic MQAR setting. For all synthetic
runs, the authors use AdamW with weight decay 0.1, 10% linear warmup, and
sweep four learning rates across **1e-4 to 1e-2**. Their gated-recurrence and
other sequence-mixer experiments therefore establish that the useful search
range for synthetic associative recall spans the same order of magnitude as
0.003. The paper does not identify one rate that transfers to this model or to
the overwrite task; it explicitly tunes rates per architecture.

Source: https://arxiv.org/pdf/2312.04927 (Appendix E.2, lines 2008–2019).

### Unlocking Feature Learning in Gated Delta Networks at Scale (Liu & Gu, 2026)

This is the most informative recent GDN learning-rate study. It evaluates
roughly 21M–342M-parameter GDNs with AdamW, 0.1 weight decay, gradient clipping
1.0, cosine scheduling, and 2,000 warmup steps. It searches AdamW rates from
**1e-3 to 2e-2** and reports an optimal rate of **8e-3** for its proposed
μP parameterization across widths. The same paper says standard
parameterization does not reliably transfer the optimal rate across widths.

This result places 0.003 inside a current GDN-specific empirical range, but it
does not justify copying 8e-3: the repository uses standard parameterization,
different gate initialization, a 10M model, and a 1,000-step OneCycle schedule.

Sources:

- https://arxiv.org/abs/2606.04048
- https://arxiv.org/html/2606.04048 (experiment details and Appendix D; the
  paper reports the 8e-3 AdamW dynamic-analysis setting).

### Kimi Linear (Kimi Team, 2025)

Kimi Linear pretrains a 3B-activated/48B-total hybrid model with MuonClip and
sets the rate to **1.1e-3** under a WSD schedule, 32M tokens per step, and
1.4T–5.7T tokens. It is useful evidence that modern KDA-style training can use
an order-1e-3 rate, but the optimizer, model scale, batch, schedule, and token
budget differ too much to select the rate for this pilot.

Source: https://arxiv.org/html/2510.26692 (§5.4.1, lines 328–332).

### Does Step Law Transfer to Small-Scale Language Models? (Romanyukov et al., 2026)

This recent TinyStories study is useful as a scale check. It runs 935
standard-parameterization nanoGPT/AdamW experiments, with model sizes from
255K to 2.03M parameters (plus 5M-parameter anchors), warmup-cosine decay,
5% warmup, Adam betas (0.9, 0.95), weight decay 0.1, gradient clipping 1.0,
and a 1e-5 terminal learning rate. The repository's corrected synthetic
models have about 3.3M–3.5M trainable parameters, so this is much closer in
scale than the billion-parameter GDN papers, although it is a different task
and architecture.

The paper finds that a large-model Step-Law estimate over-predicts the
small-model optimum by a median factor of about **4x** (roughly 2.4x–6.6x
across its cells), and proposes a separate small-scale fit. It does not give
a universal rate for this repository, but it is a strong warning against
blindly importing a higher rate from large-model scaling or μP experiments.
Its 5% warmup and 1e-5 floor are also close to the effective endpoints of the
repository's 0.001 OneCycle schedule.

Source: https://arxiv.org/html/2609.27581 (Methods and Results, lines 189–203,
215–223, 239–289).

### Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing (Cerruti et al., 2026)

This is the most directly matched short-run comparison currently available.
Its 350M-parameter Gated DeltaNet learning-rate ablation trains hybrid models
for 2,000 steps (about 1.05B tokens) and reports that AdamW Gated DeltaNet
prefers a rate near **1e-3**. The shared recipe uses Adam betas (0.9, 0.95),
weight decay 0.1, gradient clipping 1.0, and a WSD schedule; the recorded
350M AdamW GDN baseline uses a peak rate of 3e-4 outside the ablation. The
paper explicitly says the preferred rate depends on optimizer and mixer, and
that its ablation optimum is architecture-specific.

This result is not a direct proof for the repository's 10M standard-parametrized
model: it uses 350M parameters, a different WSD schedule, and a hybrid stack.
It is nevertheless stronger evidence for a short fixed-budget GDN run than the
large-scale 4e-4 recipe or the μP-specific 8e-3 result. It supports choosing
1e-3 as the robust fixed point and treating 3e-3 as an optional sensitivity
condition rather than the default.

Source: https://arxiv.org/html/2607.07953 (§5.2 and §6.2, lines 259–270,
315–323).

## Why 0.001 is the best fixed choice here

The old 0.0006 setting came from the project plan before the short fixed-budget
experiment was finalized. The primary literature gives three useful anchors:

- 4e-4 for large-scale original GDN pretraining;
- roughly 1e-3–1e-2 for synthetic recall sweeps and recent small-to-medium GDN
  rate searches;
- 1.1e-3 for KDA under a different optimizer and massive pretraining recipe.
- the small-TinyStories sweep shows that large-scale rate laws can be about 4x
  too high below a few million parameters, so scale transfer should be
  conservative.

The current run is a standard-parameterization 10M model, uses AdamW and
gradient clipping, and needs meaningful movement in only 1,000 optimizer steps.
The new 350M/2,000-step GDN ablation directly favors 1e-3, while the MQAR
literature shows that 1e-3 is inside a broad useful sweep. A 0.001 peak is
therefore the most defensible fixed point: it is high enough to move a small
model within a short budget, but closer to the directly matched GDN evidence
than 0.003. It is a protocol choice informed by literature, not a claim that
the literature proves 0.001 is universally optimal.

## Required interpretation

The 1,000-step run should report fixed-budget performance and validation curves.
It should not claim convergence, architecture-wide optimality, or transfer to
other scales. If the timing smoke is the only new run, its timing is useful for
planning only and says nothing about which rate gives the best final accuracy.
If the project later wants to optimize the rate scientifically, use a small,
validation-only comparison of 0.0006, 0.001, and 0.003 (with the same V4 data,
seed, and schedule), then lock one rate before the multi-seed study.
