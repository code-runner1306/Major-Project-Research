# Final synthesis: research topic, results and conclusions

Date: 18 November 2026. This note consolidates every experiment in the project. The supporting notes and result
folders are listed in the evidence map at the end.

## Paper topic

**Working title:** *Data Structure, Not Gate Granularity: What Determines How Small Gated DeltaNets Remember*

**Question.** In small Gated DeltaNet (GDN) memories, does channel-wise (per-key-channel) decay help over scalar
decay, and if not, what does determine whether the learned memory is reliable?

**One-sentence answer.** Channel-wise decay gives no reliable benefit at small scale, in our own tasks and in an
exactly reproduced published harness. What decides whether a model learns real latest-write recall or a hidden
position-based shortcut is the structure of its training data, and targeted data diversity removes the shortcut at
no cost.

## Part 1: gate granularity (the original question). Result: no reliable benefit

| Evidence | Setting | Result |
|---|---|---|
| Our synthetic study | 1.6M GDN, MQAR and overwrite; 3 seeds × 3 initializations; fresh held-out stress tests | Final accuracy equal (about 99-100% for both). MQAR: no difference (channel-wise better in 4 of 9 model pairs). Overwrite stress: small exploratory channel-wise edge (8 of 9 pairs, about +3 points; uncorrected, not pre-registered) |
| **Published harness** (Anatomy of Associative Recall, arXiv 2609.16183) | Their code, their commands, 10 seeds; their arm reproduced first | **Reproduction passed in all 3 cells** (E1 seed-for-seed to within 0.0005). **Scalar ≥ channel-wise everywhere:** E1 recall at constrained state, scalar 1.000 (10/10 perfect) vs 0.983 (p = 0.0006); E4 distractor haystack tie, 6/10 vs 6/10; E5 collision keys, 5/10 vs 3/10 (p = 0.65, scalar direction) |
| Shortcut susceptibility | Every layout-fix arm, 2 fresh test sets | Identical across gate types (for example A0 block-first at 3 rewrites: scalar 11.8% vs channel 10.4%; A4: 99.7% vs 99.6%) |
| Language modelling | TinyStories 30M, 1000 steps | Scalar 2.366 (n=3) vs channel-wise 2.378 (n=2): channel-wise slightly worse. 10M: one seed each, 2.811 vs 2.812. **55M was never trained** (see integrity notes) |
| Mechanism | Interventions on trained channel-wise checkpoints | Channel-wise models **do use** their channel diversity: averaging channels per token collapses them (Anatomy E4/E5 to about 0.02; our MQAR 87-94% → 40-70%). Scalar-trained models still match or beat them. Channel diversity is a route the model takes, not an advantage it gains |
| Initialization (side result) | Default, mean-decay-matched and DAMI-style joint initialization | No consistent benefit; seed effects dominate |

**Conclusion 1.** At small scale (1.6M-30M parameters, short training), channel-wise decay gives no reliable
advantage over scalar decay. In the published harness, scalar decay is equal or slightly better. This bounds, and does
not contradict, reports of channel-wise gains at larger scale and longer training (Das, TMLR; KDA; GDN-2).

**Side finding.** The Anatomy paper describes its Gated DeltaNet as scalar-gated, but its code applies channel-wise
decay. Its stated "scalar-decay" liability is therefore untested in that paper, and our scalar arm shows no such
liability. Contact the authors before publishing this point.

## Part 2: data structure determines what is learned. Result: a hidden, causally explained shortcut

1. **Discovery.** On our keyed-overwrite task, models score 99.98% on the training layout, but at extra rewrites
   they return the second write of the key. Mechanistic checks excluded write-gate collapse. Key alignment was the
   only lever, and errors trace to fixed positions.
2. **Pre-registered confirmation on fresh data.** The failure occurs in the **block-first** layout (all of the queried
   key's writes precede every unrelated write), in **16 of 18** legacy models and **6 of 6** retrained controls on
   each of two fresh test sets. 97% of errors are the first replacement. It cannot be seen on the training layout
   (all 18 models at least 95%). Under a filler shift, errors come from an unrelated key's value at a fixed slot,
   which is position-based rather than key-based retrieval. It coexists with key-based lookup in the same model.
3. **Causal dissection** (5 data arms, scalar and channel-wise, 3 seeds, identical starting weights; two independent
   fresh evaluations):

| Arm (training-data change) | Rewrite shortcut | Distance failure (d=12) | General layouts at 2-4 rewrites | Unrelated recall |
|---|---:|---:|---:|---:|
| A0 control | **6/6** | 67-pt deficit | 97.2% | 97.3% |
| A1 varied query distance | 3/6 | **removed** (also at untrained d=20/24) | 87.4% | 97.7% |
| A2 block-first removed | 4/6 | 66 pts | 98.3% | 92.9% |
| A3 mixed rewrite counts | **0/6** | 52 pts | 99.7% | 96.5% |
| **A4 varied distance + mixed rewrites** | **0/6** | **0/6** | **100.0%** | **99.6%** |

   - The two failures have two separate causes: a fixed query distance gives the position failure, and an always-one
     rewrite count gives the rewrite failure.
   - Each data change removes only its own failure. Combining them (A4) removes both, with no cost: it has the best
     general accuracy and the best unrelated recall of all arms.
   - Removing the layout instead (A2) leaves it unlearned, not fixed.
4. **Negative controls (scope).**
   - **Standard Zoology MQAR** (14 existing models, upstream generator): no comparable shortcut (1 of 14, weak, not
     position-anchored).
   - **KATA overwrite replication** (paper's model and training, 1,167,876 vs 1,167,878 parameters; 0.995 vs 0.998 at
     K=64): 0 of 3 models show the rewrite shortcut, even though every training key is written exactly twice.
   - So the shortcut appears where a generator gives part of its data a fixed, answer-predictive positional cue, and
     not elsewhere.

**Conclusion 2.** Near-perfect accuracy can hide a position-based shortcut for a sub-population of inputs when the
training generator contains a fixed structural regularity. Ordinary validation cannot detect it. Its causes can be
isolated by data-only interventions, and combined data diversity removes it entirely, while gate design has no effect
on it.

## Overall conclusion (final)

> For small Gated DeltaNet memories, the architectural choice between scalar and channel-wise decay is not what
> matters. Across our tasks, a published harness and a short language-modelling comparison, channel-wise decay gave
> no reliable benefit, and scalar decay was equal or slightly better. What matters is the structure of the training
> data. A fixed positional regularity in part of the data produced a validation-invisible, position-based shortcut in
> both gate types. Varying query distance and rewrite count removed it completely and at no cost, while standard
> generators without such a regularity (Zoology MQAR, a KATA overwrite replication) produced robust key-based recall.
> Practical recommendation: audit synthetic memory generators with layout-stratified tests and diversify the
> structural factors a model could use as a positional cue, rather than expecting finer-grained gating to supply
> robustness.

## Integrity notes (must be respected in the paper)

1. **55M TinyStories does not exist.** `output/tinystories/55M/` is a copy of the 30M runs, and the 10M/55M values in
   `output/scale_results.md` and `seed_level_results.md` are projections. Report only measured 30M (and 10M as a single
   seed), or train the missing sizes. Never present projected numbers as results.
2. **The KATA overwrite layout is our reconstruction.** The authors' data code is unpublished; say so.
3. **The Anatomy paper-versus-code discrepancy** is our reading of their public code (commit 8ca624a). Phrase it
   factually and contact the authors.
4. **Exploratory versus pre-registered.** The small channel-wise edge on our overwrite stress cells is exploratory and
   uncorrected; label it as such. All Part 2 claims and the Anatomy predictions were pre-registered.
5. **Scope.** 1.6M-30M parameters, synthetic tasks, short training, 3 seeds (10 in the Anatomy harness). Not a claim
   about larger language models.

## Evidence map

| Result | Location |
|---|---|
| Calibrated and joint initialization, decay boundary | `research_results/completed_joint_decay_review_20261117/`, `dynamic_decay_boundary_v1/` |
| Write-gate mechanism | `research_notes/write_gate_mechanism_review_2026-11-17.md` |
| Block-layout confirmation | `research_notes/block_layout_confirmation_review_2026-11-17.md`, `research_results/block_layout_confirmation_20261117/` |
| Layout-fix A0-A3 | `research_notes/layout_fix_results_2026-11-18.md`, `research_results/layout_fix_confirmation_20261118/` |
| Layout-fix A0-A4 (second fresh test) | `research_results/layout_fix_a4_confirmation_20261118/` |
| Zoology MQAR test | `research_notes/zoology_layout_test_2026-11-18.md` |
| KATA replication | `research_notes/kata_replication_results_2026-11-18.md` |
| Anatomy harness, scalar vs channel | `third_party/dualgoose/results/granularity_report.md`, `third_party/dualgoose/UPSTREAM.md` |
| Conclusion check and LM integrity | `research_notes/overall_conclusion_check_2026-11-18.md` |
