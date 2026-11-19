# Alternative research angles using the current checkpoints

Date: 17 November 2026. The requested independent subagent was launched but hit an account usage limit before producing findings. The assessments below are the primary agent's provisional follow-up, not an independent completed review. No experiment or new training was run for this note.

## Alternative A: do apparently successful models obey the rules of updating?

Question: can clean-task accuracy conceal failures of basic update consistency, and do initialization/gate variants fail different rules?

This changes the headline from which gate is necessary to whether the learned model behaves like an updating memory. It can start with all existing checkpoints, including the incoming joint comparisons. The obsolete-first-replacement pattern is motivation, not evidence that every rule is violated.

### Paired tests

1. **Obsolete-prefix irrelevance:** change an earlier value while keeping the final write and query fixed. The expected newest answer is unchanged.
2. **Latest-write sensitivity:** change only the final value. The newest answer should change accordingly, while unrelated answers stay correct.
3. **Independent-update order:** swap complete writes to different keys while preserving the per-key final binding. The expected answers stay fixed. Match length and analyze query-distance changes; where distance cannot be matched, explicitly stratify or exclude that pair from a distance-controlled contrast.
4. **Duplicate-write consistency:** repeat an identical binding and compare against a length/distance-matched control. The expected binding should not change. This requires careful positional controls; merely adding tokens is not an isolated duplication intervention.
5. **Symbol renaming:** apply a consistent permitted key/value bijection and transform the expected answers. This probes dependence on token identities and training frequencies, not a theorem that the trained embedding network must be permutation-equivariant.

Report joint pair correctness, expected-answer transformation success, and correctness-conditioned interference. Do not use raw output agreement as success: a constant wrong answer can be perfectly consistent. Break down effects by seed, update ordinal, and gate intervention.

### Prior overlap and distinct contribution required

[How Linear Attention Remembers](https://arxiv.org/html/2609.33093) already performs causal multi-fact/interference interventions. [KATA](https://arxiv.org/html/2607.17419) already studies latest-write overwrite; [Schlag et al.](https://proceedings.mlr.press/v139/schlag21a.html) already establishes delta-rule correction. [Anatomy of Associative Recall](https://arxiv.org/abs/2609.16183) and [On the Importance of Gating](https://arxiv.org/abs/2609.16540) constrain shortcut and training-dynamics claims. Metamorphic testing in general is established software-testing methodology; paired tests are not an invented technique here.

Potential contribution: a specific rule-violation taxonomy with a mechanism and a demonstrated repair, showing that ordinary held-out loss/accuracy fails to distinguish reliable from unreliable updating. The exact taxonomy has not been comprehensively priority-checked; the three inaccessible close-looking manuscripts in the systematic review remain unresolved. Do not call this the first such benchmark or method.

Cost: no GPU training for an initial paired evaluation. A 36-checkpoint, two-task, five-pair-type study at 256 base examples plus their transforms has about 92160 example-forwards before baselines and controls; CPU cost needs timing and depends on actual sequence lengths. Start with overwrite and a subset of pair types. A repair test, if warranted, can share the six-model mixed-rewrite follow-up in the main plan, about 2h50m measured-runtime extrapolation / roughly four hours budget on the measured A6000-12Q profile, subject to generator and control costs.

Failure criterion: violations disappear after length/distance matching, occur only in one seed, or simply reproduce low single-example accuracy without new explanatory value. Then retain these tests as supporting controls rather than a separate novelty claim.

Subjective residual-novelty potential: **about 5/10**, execution fit high for evaluation. Not established as higher novelty than the main boundary study. Its advantage is a cleaner behavioral question, not an unoccupied literature.

## Alternative B: can a cheap validation test prevent wasting a small training budget?

Question: does a small, deliberately varied memory-validation set select more robust checkpoints than ordinary IID loss under the same GPU-hour ceiling?

This is a training/selection-protocol study, rather than an architecture mechanism claim. It would compare fixed-last checkpoint, best ordinary validation loss, and a prespecified robust-validation selection rule. Evaluate each selection on untouched, different stress cases and report selection overhead as part of the budget. Avoid selecting on the same stress cells used for the final claim.

Current final-only checkpoints and development curves are insufficient for a retrospective causal comparison of selection policies: earlier model states must exist. A future matched run must preserve per-epoch model weights (small weight-only snapshots), with the latest optimizer state saved separately. First confirm whether any relevant snapshots already exist; do not claim they do.

Prior art is substantial: [Revisiting Associative Recall](https://arxiv.org/abs/2508.19029) studies optimization sensitivity; [RULER](https://arxiv.org/abs/2404.06654) and [Zoology](https://arxiv.org/abs/2312.04927) motivate better memory evaluation; [Anatomy](https://arxiv.org/abs/2609.16183) already uses performance-responsive curricula. General robust validation/model selection is not new. A credible contribution needs a reliable, low-cost selection gain across held-out training seeds or recipes, not a metric correlated with the same results that motivated it.

Cost: potentially six fresh default overwrite models plus per-epoch diagnostic evaluation, roughly four GPU hours before extra control/seed costs; unmeasured validation overhead could change this. Three training seeds are too few for a broadly trained/tested diagnostic predictor. Do not fit a many-feature predictor on 36 correlated checkpoints and report it as generalizable.

Failure criterion: selection gives no held-out gain after overhead, or its advantage depends on tuning against the final tests. Subjective residual-novelty potential: **about 4/10**; substantial extra validation is required.

## Recommendation

Keep the dynamic-decay boundary experiment as the main plan. Use Alternative A's paired update tests as an early, cheap probe. Promote it to the paper's central question only if it yields a repeatable failure class and an explanation beyond current accuracy/interference findings. Alternative B is a useful practical extension if checkpoint snapshots and extra seeds become available, not the next expensive commitment.

No alternative examined here is honestly established as an obviously higher-novelty direction. A stronger contribution will come from a discriminating result and explanation, rather than renaming the same experiment. The main plan is [dynamic_decay_boundary_experiment_plan_2026-11-17.md](dynamic_decay_boundary_experiment_plan_2026-11-17.md).
