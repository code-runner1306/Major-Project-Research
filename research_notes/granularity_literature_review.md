# Gate granularity study: literature and paper-potential review

Date: 2026-11-13

## Scope

This review evaluates the proposal's revised question: when initial memory timescales are controlled, how much independent decay control is useful, and does the answer differ between association retention and overwrite recall? It covers the 10M/30M scalar, grouped, and channel-wise design with calibrated versus default initialization, using TinyStories, MQAR, and overwrite recall.

## What the primary literature already establishes

- Gated DeltaNet introduced scalar head-wise gating combined with the delta rule. Kimi Linear's Kimi Delta Attention (KDA) then made the decay channel-wise and reported large-model efficiency and quality results in a hybrid architecture. The Kimi paper therefore establishes the practical importance of fine-grained decay, but does not isolate granularity in a small matched factorial study. Sources: [Gated DeltaNet](https://arxiv.org/abs/2412.06464), [Kimi Linear](https://arxiv.org/abs/2510.26692).
- Gated DeltaNet-2 keeps channel-wise decay and adds separate channel-wise erase and write gates. Its reported 1.3B-parameter, 100B-token results show strong quality and long-context retrieval, but the architectural change is broader than a scalar-versus-channel-wise decay ablation. The qualitative claim that channel-wise control can help is therefore no longer novel by itself. Source: [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791).
- Das's controlled factorial study is the closest overlap. It compares scalar versus channel-wise decay with and without delta-rule writes at 18M and 125M on TinyStories and WikiText-103, with three seeds, and reports that the delta rule is the dominant factor under that protocol. This substantially reduces the novelty of a paper framed only as “scalar versus channel-wise gating in small models.” Source: [Delta-Rule Writes in Linear Attention: A Controlled Factorial Study of Decay Design](https://tmlr.infinite-conf.org/paper_pages/FdXB4u27LD.html).
- Systolic Linear Attention groups decay gates into four hardware-sized tiles for TPU efficiency. Its goal is hardware co-design rather than a clean statistical study of learned granularity, but it means grouped gates are also an active design point and cannot be presented as an unexplored idea. Source: [Systolic Linear Attention](https://arxiv.org/abs/2611.06828).

## Where the proposed study can still contribute

The strongest remaining gap is the interaction between initialization and granularity under a fixed small-model protocol. The closest factorial paper varies decay and delta-rule structure, but the proposal's calibrated geometric initialization asks a different question: whether an apparent advantage of more independent gates survives after initial memory timescales are matched. That is a defensible mechanistic contribution if the initialization is genuinely matched in realized gate behavior and the analysis separates initialization effects from capacity and optimization effects.

The second useful angle is task structure. MQAR stresses retention and association retrieval; overwrite recall stresses replacing stale values. Gated DeltaNet-2 motivates this distinction conceptually through separate erase and write roles, but the proposal's scalar/channel-wise × initialization comparison on these two synthetic memory tasks could make the interaction concrete at a much smaller scale.

The result must be framed as a controlled small-scale study and diagnostic, not as a new architecture or a first demonstration that channel-wise gating works. The most convincing outcome would be one of these:

1. Calibration removes most of the channel-wise advantage, showing that prior gains partly reflect initial timescale diversity.
2. Calibration preserves or increases a channel-wise advantage on retention but not overwrite, showing task-dependent value for independent decay.
3. Grouped-4 captures most of the benefit at a lower parameter and hardware cost, giving a practical operating point.

## Honest scorecard

Scores are for the study as currently specified, before the missing C6 runs, held-out evaluation, multiple seeds, and 30M confirmation.

| Aspect | Score | Assessment |
|---|---:|---|
| Research-question clarity | 8/10 | The revised question is specific, falsifiable, and tied to a mechanism: initialization × granularity × memory demand. |
| Novelty after literature review | 5/10 | Scalar versus channel-wise gating is established and a close factorial study exists. Initialization matching and retention-versus-overwrite interaction preserve moderate novelty. |
| Mechanistic value | 7/10 | Calibration and post-hoc gate sharing can explain why a quality gap appears, especially if realized timescales and gate diversity are reported. |
| Experimental control | 8/10 in design | The paired conditions, fixed data and budgets, separate splits, calibration provenance, and artifact checks are strong. |
| Current empirical support | 4/10 | The pushed run has one seed, a short 1,000-step exploratory budget, incomplete C6 training, and no held-out evaluation yet. |
| Statistical strength | 2/10 currently; 7/10 after plan | One seed cannot support an inferential claim. The planned five seeds at 10M and confirmation at 30M would materially improve this. |
| Practical relevance | 7/10 | Small-model and hardware-constrained practitioners care about whether extra gate resolution is worth its cost, but the study must measure latency, memory, and parameter overhead alongside quality. |
| Feasibility | 6/10 | The protocol is implementable, but the current sequential 10M runs take roughly three hours each, so the full factorial and seed plan need careful budgeting. |
| Likely workshop paper potential | 7/10 after completion | A clear controlled result with matched initialization, task-specific effects, grouped intermediate points, and uncertainty could make a solid workshop or specialized venue paper. |
| Likely top-tier main-track potential | 3/10 as currently framed | The core comparison overlaps existing work; top-tier potential would require a stronger generalization, a sharper mechanistic explanation, broader validation, or a new efficient grouped design. |
| Overall paper potential | 6/10 now; 7.5/10 if the full protocol succeeds | Promising and publishable as a careful empirical study, but not yet a decisive or high-novelty result. |

## Interpretation of the latest exploratory run

The partial VM results are consistent with the proposal's motivating question but are not yet evidence for a final claim. At step 650, calibrated scalar C4 is substantially ahead of default scalar C1 on both MQAR and overwrite validation loss, while default channel-wise C3 is only slightly better than C1 on MQAR and worse on overwrite. This makes initialization look like a major factor and supports continuing the factorial study. It does not establish the channel-wise effect because calibrated channel-wise C6 and held-out results are missing.

The current outcome is therefore interesting as a lead: it suggests that a paper centered on “initialization can dominate apparent granularity gains” may be more original and better aligned with the literature than a paper centered on “channel-wise beats scalar.”

## Requirements before making a strong claim

1. Complete C6 for both families and run the full held-out evaluation.
2. Run the prescribed multiple seeds, at least at 10M, and report paired confidence intervals or bootstrap intervals rather than one-seed rankings.
3. Add the grouped-4 condition to the main 10M factorial if the paper's practical contribution includes a quality/cost operating point.
4. Report exact parameter counts, gate/projection overhead, decode latency, throughput, recurrent-state memory, and energy where available.
5. Verify that calibrated conditions match realized initial decay distributions and that calibration does not use test or final evaluation data.
6. Compare against the closest controlled factorial baseline and explain what the new initialization factor adds beyond it.
7. Treat the 1,000-step run as exploratory evidence and avoid claims about convergence or scaling until the longer fixed-budget protocol is locked.

## Bottom-line assessment

The question is worth pursuing and can support a credible paper, especially if the result is that initialization changes the apparent value of fine-grained decay and that the effect differs between retention and overwrite. The literature makes a simple channel-wise-versus-scalar claim too familiar to carry the paper by itself. The publishable contribution is the controlled interaction study and its mechanistic, cost-aware explanation.
