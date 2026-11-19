# Experiment plan: when does dynamic decay support reliable memory updates?

Status: proposed, 17 November 2026. No new training or evaluation is launched by this document. The currently running joint-initialization experiment continues. Freeze a machine-readable protocol before generating the new discovery data; this planning document is not a retrospective preregistration of the existing results.

## Question and scope

For our approximately 1.6M-parameter, one-head, four-layer GDN adaptation, when does replacing input-dependent decay with fixed channel constants impair recall or updating, and does that dependence change with initialization and training coverage?

The intended contribution is a reproducible empirical boundary and a tested explanation. It is not a new architecture, the invention of static channel gates, a general claim about all GDNs, or proof of an optimal training recipe. “Necessary” means necessary for a specified performance criterion in a fixed trained model under a defined intervention. It does not mean no static architecture can learn the task.

The evidence base and competing work are in [the systematic review](systematic_review_20261117/README.md) and [direction cross-check](systematic_review_20261117/direction_crosscheck.md). Closest precedents include [Das](https://tmlr.infinite-conf.org/paper_pages/FdXB4u27LD.html), [DAMI](https://arxiv.org/html/2610.41994v2), [decay design space](https://arxiv.org/html/2509.05282), [KATA](https://arxiv.org/html/2607.17419), [Muir](https://arxiv.org/html/2611.08376), and [gating and learning dynamics](https://arxiv.org/abs/2609.16540). The contribution must go beyond repeating their broad comparisons.

## Fixed starting point

- Existing: 24 checkpoints = two tasks × scalar/channel-wise × default/mean-alpha initialization × three seeds (42, 43, 44).
- Incoming: up to 12 joint-initialized checkpoints = two tasks × scalar/channel-wise × three seeds. Their completeness and compatibility must be verified before inclusion; do not assume the run has succeeded.
- All current training comparisons use 16 epochs / 3136 optimizer steps, vocabulary 8192, length 64, width 128, FP32, effective batch 512, microbatch 32, LR 0.00215443469, and a cosine scheduler retaining the 64-epoch horizon. Save actual configuration rather than inferring it from names.
- The channel model has about 4.17% more parameters. Checkpoint interventions do not change parameter count, but architecture comparisons retain that confound.
- The old mean-alpha fit and joint mean-log-decay/sensitivity fit change more than sensitivity. Analyze initialization as a recipe factor; a sensitivity-only causal claim requires an additional matched-log-decay-only control.
- Existing stress/probe examples have been observed. They are exploratory evidence, not final untouched tests. The earlier static-gate success used training-side probes and must be rechecked on held-out data.

## Stage 0: freeze and qualify the comparison

1. Inventory checkpoint hashes, completed steps/epochs, exact data manifests, settings, software versions, source hashes, model shapes, and pairing. Keep each session separate under one aggregation root.
2. Preserve the original source identities. Use a separate evaluation adapter for the new protocol ID; do not patch old checkpoint metadata to force acceptance. Record the joint-run source differences explicitly.
3. Reproduce one known score per task, then run independent generator/label/latest-value/unrelated-retention/counterfactual-query audits. Check disjoint key/value roles and duplicate-value handling. If multiple obsolete values are identical, count a stale answer once and mark ordinal ambiguity rather than assigning it arbitrarily.
4. Check proposed cell feasibility before freezing final manifests. Fail on an invalid cell; amend and version the protocol before looking at model scores rather than silently removing cells afterward.
5. Fit static decay constants without labels on the same 256 cached training examples per family/seed under the answer-free policy. Never reconstruct the MQAR prefix by simply requesting fewer examples: its RNG call order depends on dataset size. Save the constants and hashes for each checkpoint.
6. Measure evaluator cost on a small timing batch and extrapolate the full evaluation workload. This is an infrastructure timing check, not another training experiment or evidence of learnability.

Stage-0 deliverable: immutable checkpoint/data/intervention manifest and an evaluation-time estimate. Stop on compatibility/audit failure.

## Stage 1: held-out intervention map, no training

Use 512 fresh discovery examples per task/seed/cell. Proposed discovery seed offset: 700000 plus training seed; final confirmation offset: 900000 plus training seed. The current top-level source search found no use of these offsets, but a manifest/content-overlap audit is still required. Keep supported generator split names such as `held_out`; name the evidence phase in the manifest instead of introducing unsupported split strings.

Same examples are shared across compatible variants and initializations. Use answer-free scored-query evaluation; score newest-value recall separately from unrelated retention. Answer-free means earlier gold answer tokens are erased; it is not free-running autoregressive generation.

### Discovery cells

| Task | Cells | Purpose |
|---|---|---|
| MQAR | length 64/128 crossed with 4/8/12 associations: six cells | Separate association load from nominal context length. Report actual per-query distance distributions; length is not identical to retrieval distance. |
| Overwrite | baseline K4/D8/R1; K8/D8/R1; K4/D24/R1; K4/D8/R2; K4/D8/R3; K4/D8/R4 | Separate load, target distance, and number of rewrites. Do not pool these factors as a single difficulty scale. |

Here R is the number of replacement writes after the initial value. These are one-factor stress slices, not a complete interaction factorial. Additional combinations belong to a separately labeled extension.

### Interventions

For channel-wise checkpoints:

1. Original input-dependent decay.
2. Fixed per-channel mean decay, fitted only on the training prefix.
3. Per-token average across channels, preserving input dependence while removing within-head channel differences.

For scalar checkpoints, original evaluation supplies an architecture reference. The channel-averaged checkpoint is not a separately trained scalar model. All other computations remain active and weights unchanged. Verify parameter fingerprints and restore hooked functions between conditions.

With all 36 checkpoints available, each family has nine scalar references and nine channel models × three conditions = 36 executions per cell. Six cells per family produce **36 × 6 × 2 = 432 executions**. This count includes the full scalar reference grid; each execution evaluates 512 examples. Without joint checkpoints, the corresponding count is 288. Keep this explicit when budgeting.

### Prespecified contrasts and interpretation

- Primary MQAR contrast: dynamic-minus-static accuracy gap at L64/K8 minus that gap at L64/K4.
- Primary overwrite contrast: dynamic-minus-static newest-value gap at R3 minus that gap at R1, with K4/D8 fixed.
- Initialization interactions and the remaining cells are secondary. Report every cell and seed, including reversals; do not select a favorable initialization after seeing the map.
- Operational evidence of reliance: a stress gap at least 5 percentage points with the same direction in all three training seeds, and a baseline gap no greater than 2 points in magnitude. These are practical decision thresholds, not significance tests.
- Stronger “necessary for reliable performance in this checkpoint” language additionally requires original accuracy at least 95% and static accuracy below 90%. If both fail, say “relative robustness difference,” not “dynamic decay solves the task.”
- If static decay matches original throughout, report that input-dependent decay was dispensable on the tested range. Do not enlarge the task indefinitely until a desirable difference appears.

Report per-seed differences, sequence-level paired uncertainty, and the three seed values. Query counts are not independent trained-model replications. A sequence bootstrap cannot turn three training seeds into strong population-level significance.

Stage-1 deliverables: all-cell metric table, paired intervention-gap plots, seed-level boundary map, raw predictions/error types, and a continue/stop decision.

## Stage 2: test the explanation at the observed boundary

Choose at most two discovery boundary cells using the frozen rule, plus their baseline controls. If there is no repeatable gap, stop the dynamic-necessity claim here.

1. **Statistic sensitivity:** compare arithmetic per-channel mean alpha with a fixed geometric mean alpha. This checks whether a result depends on the replacement statistic, rather than simply removing input dependence.
2. **Input alignment:** replay a decay schedule from an independent, length/layout-matched sequence, with answer tokens erased. This keeps a varying channel schedule while breaking alignment with current content. It is a diagnostic counterfactual, not a deployable model or perfect marginal-preserving control. Track schedule statistics and changes to downstream activations.
3. **Layer localization:** intervene in one layer at a time only in the selected cells. Do not assume effects add linearly across layers.
4. **Ordinal versus position:** relocate the same target update history while controlling available tokens, query distance, and unrelated writes. Test whether obsolete predictions follow the first replacement's identity or an absolute/relative position.
5. **Content controls:** consistently rename keys and values within their allowed vocabularies, and perturb only the latest target value. Score expected answer changes and unrelated-answer stability. Use paired examples and regenerated oracle labels.

These controls separate hypotheses; no single intervention proves the complete learning mechanism. In particular, freezing gates changes the recurrent dynamics experienced by a model trained with dynamic gates.

Stage-2 deliverable: one supported explanation or an explicit unresolved result, not just another set of heatmaps. Freeze all choices before opening the final confirmation set.

## Stage 3: minimal new training only if needed

Start with overwrite, not a rerun of the entire factorial. Use default scalar/channel-wise models across three seeds: **six new models**, training from scratch on mixed rewrite counts R=1/2/3 with randomized valid layouts. Preserve 100000 examples, length 64, 3136 steps, batch, precision, LR, and scheduler. Log per-cell train counts and scored targets.

Construct examples so the scored newest/unrelated target count is constant across rewrite counts. Equal examples/tokens/steps do not equal equal write-event exposure; report that difference. Include a matched single-rewrite control if the new sampler changes anything beyond rewrite/layout coverage. Existing defaults can be reused only if that control is provably equivalent; otherwise rerun the single-rewrite pair, starting with one diagnostic seed before expanding.

Evaluate both seen counts and unseen R4, with a reserved R5 cell only if feasible and declared before training. The held-out load/distance cells remain unchanged. Do not choose a different number of epochs for models that learn slowly.

Interpretation:

- Mixed training removes both obsolete errors and the static/dynamic gap: training coverage is a sufficient explanation in this regime; do not claim an intrinsic recurrence failure.
- Mixed training improves accuracy but a replicated intervention gap remains: dynamic decay has a more credible role under the specified conditions.
- Mixed training does not learn: revisit optimizer/data/audit controls; inability to learn is not evidence of impossibility.

An independently trained static-channel baseline is a conditional follow-up if the paper claims architectural sufficiency. It is not required to describe functional reliance of the existing checkpoints, but is required to generalize from post-hoc freezing to what static models can learn. Likewise, a parameter-matched control is needed for a strong scalar/channel architectural advantage claim; otherwise retain the narrower same-backbone limitation.

## Stage 4: untouched confirmation

After freezing the chosen primary comparisons, evaluate 512 newly generated examples per selected cell at the reserved confirmation offset, with no retuning. Include baseline and failure cells, all three seeds, newest-value recall, unrelated retention, and obsolete-answer categories. The same training seeds with new examples test data replication, not new-seed replication. Seek additional training seeds only if the conclusion depends on general seed prevalence; do not promise precision that three seeds cannot provide.

## Runtime and compute budget

Measured evidence: the previous CPU stress study evaluated 24 checkpoints × four cells × 256 examples in 12.46 minutes, while another CPU study ran concurrently. The intervention study took 10.29 minutes. These are rough planning anchors, not a hardware-independent benchmark.

| Work | Estimate | Qualification |
|---|---|---|
| Full Stage-1 map, all 36 checkpoints | Roughly 2–4 CPU hours; reserve up to half a day | 432 executions × 512 examples is nine times the old scored workload before length and intervention overhead. Measure the timing batch first. |
| Stages 2 and 4, selected cells only | Roughly 1–3 additional CPU hours | Depends on how many cells meet the fixed rule; no new GPU training. |
| Six mixed-rewrite overwrite models, two workers on measured RTXA6000-12Q | About 2h50m measured-training extrapolation; budget approximately 4h | Three paired sessions × 56.7 minutes; new sampler and convergence not yet measured. |
| Optional default-only static-channel overwrite control | Three new models; reserve roughly two paired-session lengths pending timing | A separate conditional follow-up, not included in the four-hour mixed-training budget. Crossing three initializations would expand this to nine models and needs a separate budget. |
| New matched single-rewrite controls if necessary | Additional cost, not included in the four-hour estimate | One default scalar/channel pair is about 57 minutes on the measured profile. Three seeds add about 2h50m before margin. |

The running 12-model joint experiment is already allocated and is not an additional Stage-3 cost. Its result is still unknown. Two-worker timing is measured for our implementation and virtual GPU profile, not an intrinsic architectural throughput ranking. Do not reuse A6000 wall time as a guaranteed Kaggle estimate.

## Artifacts and implementation checklist

Future run root: `research_results/dynamic_decay_boundary_v1/<session>/`, with a manifest, environment, source/checkpoint hashes, fit constants, data audits, per-cell raw predictions, metrics, timing, decision log, and figures. New training stays in separate session/family/seed/condition folders with latest checkpoints. Publication defaults to false unless explicitly enabled.

- [ ] Freeze discovery/final manifests and primary contrasts.
- [ ] Add an evaluation adapter compatible with legacy and joint protocols.
- [ ] Audit generators and content/ordinal perturbations independently.
- [ ] Implement fresh-held-out intervention map and restore/fingerprint checks.
- [ ] Generate seed-level paired summaries and fixed decision-rule output.
- [ ] Implement boundary-only mechanistic controls if Stage 1 justifies them.
- [ ] Add mixed-rewrite training only if Stage 2 requires a training-coverage test.
- [ ] Perform reserved confirmation and write claim-to-evidence table.

The currently implemented stress/intervention scripts provide components but do not yet execute this full plan. This document authorizes no claim that the proposed tests have run.
