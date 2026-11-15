# Repair specification: scientifically valid gating-granularity study

Status: local planning draft; implementation has not started.
User direction: use to-spec; keep this specification local and do not publish it.
Baseline audited: maink at 9f2cc3a and the completed exploratory seed-42 run.

## Problem Statement

The researcher needs to know whether scalar, grouped, and channel-wise decay
behave differently under controlled initialization and memory demand. The current
pilot cannot support the intended claims because the recurrence differs from
the published rules, overwrite has a perfect key-free positional solution, and
nominal context lengths append filler after the scored queries. Expanding seeds
before correcting these problems would spend scarce GPU time reproducing the
same methodological artifacts.

The pilot remains useful engineering and exploratory evidence. Preserve its
checkpoints, logs, manifests, and metrics, but do not combine it with corrected
experiments or present it as a canonical Gated DeltaNet/KDA result. The current
AMP recovery is not itself a reason to discard the run; the confirmed model and
task semantics are the blockers.

## Solution

Repair the existing research pipeline in dependency order. First establish that
its math implements the intended model and its tasks require the intended
behavior. Then qualify a feasible training recipe, run a small corrected pilot,
and measure quality and efficiency using explicit, reproducible contracts.

Fresh research training must remain blocked until the correctness and task
validity gates pass. Multi-seed training must remain blocked until the corrected
pilot demonstrates learnability and its full resource estimate is recorded.
No launcher should start a large sweep merely because a smoke test passes.

## User Stories

1. As a researcher, I want the scalar recurrence to match the published gated delta rule, so that the named baseline is scientifically accurate.
2. As a researcher, I want channel decay to act on the intended key axis, so that the granularity comparison tests the stated mechanism.
3. As an implementer, I want an independent mathematical oracle, so that agreement between two implementations cannot conceal a shared mistake.
4. As an implementer, I want gradient, state-continuation, and cached-decoding checks against that oracle, so that both training and inference use the validated rule.
5. As a researcher, I want every enabled backend to implement the same corrected semantics, so that backend choice cannot change the experiment.
6. As a researcher, I want published defaults and intentional simplifications documented, so that calibration is not compared with a misleading architectural baseline.
7. As a researcher, I want actual parameter counts and natural gate overhead reported, so that nominal scale names cannot misrepresent the models.
8. As a researcher, I want initialization factors to describe all changes they make, so that timescale effects are not confused with channel-function matching.
9. As a researcher, I want calibration to measure informative write/query contexts separately from padding and filler, so that reported retention targets describe the relevant behavior.
10. As a researcher, I want queried overwrite keys and relevant write positions randomized, so that retrieving the final two values cannot solve the task.
11. As a researcher, I want unrelated writes after the target's latest write, so that the task measures key-specific retention and overwrite rather than fixed recency.
12. As a researcher, I want randomized query order in both synthetic families, so that query ordinal cannot reveal the requested association.
13. As a researcher, I want counterfactual query-key checks, so that changing the queried key changes the correct answer while memory stays fixed.
14. As a researcher, I want simple recency, position, and no-memory baselines, so that benchmark shortcuts are visible before model training.
15. As a researcher, I want sequence length to change meaningful context before scored answers, so that length sweeps test actual memory demand.
16. As a researcher, I want actual per-answer distance and distractor load audited, so that nominal difficulty labels match the generated sequences.
17. As a researcher, I want separate deterministic training, validation, and held-out streams, so that recipe selection does not consume the final test set.
18. As a researcher, I want query accuracy, example accuracy, and history conditioning distinguished, so that metrics do not overstate retrieval success.
19. As a researcher, I want easy-task learnability established for both endpoints, so that a broken recipe cannot masquerade as limited architectural capacity.
20. As a researcher, I want comparable validation-only tuning effort, so that one shared recipe is not selected to favor an observed test winner.
21. As a researcher, I want exact accepted-step accounting with AMP recovery, so that mixed precision cannot silently reduce one arm's training budget.
22. As a researcher, I want bounded precision-sensitivity checks, so that finite recurrent computations are not mistaken for numerical agreement.
23. As a researcher, I want gate-sharing interventions reported beside the trained-model comparison, so that optimization and inference-time explanations remain distinguishable.
24. As a researcher, I want paired seed-level analysis and a prespecified primary interaction, so that examples and model replicates are not conflated.
25. As a researcher, I want failed and incomplete runs retained in reports, so that exclusions cannot make average performance or capacity look better.
26. As a researcher, I want artifact reuse to validate the full scientific protocol, so that corrected runs cannot silently reuse legacy results.
27. As a researcher, I want isolated inference benchmarking even when training is parallel, so that GPU contention cannot invalidate efficiency claims.
28. As a researcher, I want energy and memory measurement boundaries aligned with their metric labels, so that deployment cost is not misreported.
29. As a researcher, I want CPU smoke, GPU correctness, GPU capacity, and research qualification distinguished, so that passing one does not imply passing the others.
30. As a researcher, I want a costed pilot and explicit stage gates before additional seeds or scales, so that the study stays feasible.
31. As a collaborator, I want repaired protocol documents and paper claims to agree with the executable settings, so that the old proposal cannot reintroduce unsupported claims.

## Implementation Decisions

### Stage 0 — Freeze the exploratory baseline and isolate the correction

- [ ] Preserve the completed exploratory run as immutable legacy evidence, including any VM-only checkpoints. Record where large checkpoint artifacts are stored and how their hashes are verified.
- [ ] Give corrected model/recurrence semantics and synthetic generation new versions. Use a fresh experiment identity and output root; do not resume corrected training inside the legacy experiment.
- [ ] Record the audit findings and pilot limitations in research documentation. Remove instructions that encourage extending the invalid protocol as the next research step.
- [ ] Define a migration boundary: existing checkpoints may be evaluated as explicitly labeled legacy models, but never relabeled as trained with corrected math or data.

Exit gate: legacy evidence is preserved, and corrected task identities cannot resolve to legacy run directories.

### Stage 1 — Repair and validate the intended mathematics

- [ ] Pin primary equations and a maintained official reference revision for scalar GDN and KDA-style key-axis diagonal decay. Record state orientation explicitly as value-by-key or key-by-value throughout the implementation and oracle.
- [ ] For value-by-key storage, implement the scalar contract as decayed state D = alpha*S, residual e = v-D*k, and updated state S' = D+beta*e*k-transpose. Implement channel decay as key-column scaling before the same residual/update; scalar alpha must reduce to the scalar equation.
- [ ] Correct gradient calculations, continuation state, cache handling, and any optimized recurrence/decode paths. Disable an optimized path until it qualifies against the corrected independent oracle.
- [ ] Replace tests that establish only local self-consistency with equation-based numeric and gradient checks. Include nonzero initial states, multiple time steps, nonsymmetric states, one-hot query/key probes, and nonuniform alpha.
- [ ] Version semantics and implementation hashes so old qualification artifacts cannot authorize a corrected backend.
- [ ] Review the complete token-mixer block: effective retention parameterization/defaults, Q/K/V convolution activations and normalization, output normalization/gating, and any hybrid layers. Match the declared reference or explicitly scope the model as a simplified common-backbone ablation.
- [ ] Use a defensible, documented default memory initialization. Do not call a zero-bias sigmoid gate a faithful published default without reference evidence. Keep gate-function family and non-granularity choices common across paired arms.

Exit gate: independent forward/state/gradient checks pass, scalar and grouped endpoints agree under mapped weights, cached decoding agrees, and every enabled backend has fresh qualification evidence.

### Stage 2 — Rebuild synthetic tasks around genuine keyed retrieval

- [ ] Randomize relevant write locations and query ordering for overwrite. Permit unrelated writes after the last target update; avoid making either query's answer identifiable from fixed final memory positions.
- [ ] Randomize the first queried MQAR association independently of memory order. Avoid a guaranteed latest-write first query.
- [ ] Require a keyed lookup/last-write-wins oracle to solve all feasible generated cases. Check oracle results from the input prefix rather than trusting target-placement metadata.
- [ ] Add query-key counterfactuals: changing a key while holding the memory fixed must select the corresponding value. Include stale-versus-latest contrasts and unrelated-retention queries without making their answers positionally predictable.
- [ ] Benchmark key-free recency/position baselines, latest-value copying, first-value copying, no-memory/frequency predictions, and query-order rules on the frozen distribution. The audit's perfect overwrite baseline must fail as a task solution.
- [ ] Prespecify a shortcut criterion before training: within each main difficulty cell, these baselines must remain below the model learnability threshold and show no deterministic near-perfect solution. For a randomized uniform choice among K distinct latest values, document the approximate 1/K baseline and its example-level uncertainty; do not claim every arbitrary baseline must have zero accuracy.
- [ ] Vary actual query distance, distractors, and prefix length before prediction. Longer nominal context must change the scored causal prefix; filler after the last scored answer cannot constitute a long-context condition.
- [ ] Audit feasibility and balance jointly over key count, actual distance, rewrite count, length, and distractors. Reject infeasible cases rather than silently relabeling them.
- [ ] Persist generated-case identities and split domains. Check paired conditions receive identical examples and establish content-level split checks in addition to distinct configuration hashes.
- [ ] Separate padding from intentional task filler, report effective/nonpadding/supervised tokens, and keep packing/loss masks deterministic.

Exit gate: the symbolic oracle succeeds, the known shortcut attacks fail as solutions, query-key counterfactuals behave correctly, scored-prefix length changes as intended, and train/validation/test provenance is auditable.

### Stage 3 — Fix evaluation contracts and initialization interpretation

- [ ] Choose and freeze the primary query format. Prefer answer-free query inputs for clean associative-recall measurement. If retaining oracle-history evaluation as a secondary task, label it explicitly and implement a separate autoregressive rollout for generation claims.
- [ ] Retain causal prediction before each answer token. Add suffix-perturbation checks proving a scored prediction cannot depend on its own answer or future content.
- [ ] Report per-query accuracy, per-example all-queries-correct accuracy, and prespecified macro/micro summaries separately. Use latest-value accuracy as the primary overwrite endpoint; keep unrelated retention and stale/other errors distinct.
- [ ] Report results jointly by actual per-answer distance and load. Include first versus later query diagnostics so an aggregate cannot hide positional shortcuts.
- [ ] Define calibration as a package of bias/timescale fitting and within-head initial function matching, unless bounded controls separately identify their effects. Rename claims accordingly.
- [ ] Calibrate on training-only data with explicit weighting and padding masks. Report realized gate distributions on write/query/filler states, input sensitivity, and per-layer/head residuals. A half-life computed from mean alpha must be labeled a decay diagnostic rather than actual full-state memory lifetime.
- [ ] Preserve a small decomposition check if attributing effects specifically to timescales: calibration without shared projection rows, and function matching without the full timescale change. Do not expand this into a large sweep automatically.
- [ ] Repeat posthoc sharing after repairs using the same examples and checkpoint for every setting. Specify averaging alpha versus log(alpha) correctly and fix inconsistent report descriptions. Treat retained accuracy after scalarization as evidence against a necessary inference-time channel-diversity explanation, while avoiding claims about independently trained scalar/grouped models.

Exit gate: metric units/history conditioning are explicit, causality holds, calibration provenance matches the declared treatment, and intervention interpretation is consistent with the measured behavior.

### Stage 4 — Qualify the training recipe and numerical behavior

- [ ] Verify identical non-gate backbone initialization and data order across paired arms. Record all trainable gate biases and natural parameter differences; do not silently add capacity only to calibrated arms.
- [ ] Establish easy-task learning for both scalar and channel-wise endpoints on corrected generators. Start from the plan's frozen engineering criterion of at least 90% answer accuracy at two consecutive validations; change it only before observing qualification outcomes and record why.
- [ ] Select a feasible shared recipe using validation only and comparable tuning effort. Inspect curves versus accepted steps, effective tokens, and wall time. Label results as fixed-budget effects until convergence is demonstrated.
- [ ] Retain bounded same-batch AMP retries: reduce loss scale, restore relevant RNG state, clear rejected gradients, and advance optimizer/scheduler/data counters only on accepted updates. Persistent nonfinite values must fail loudly and remain reportable.
- [ ] Compare bounded recurrent forward/state/gradient trajectories across FP32 and the intended mixed-precision setting, including long prefixes and near-one retention. Disable autocast around an explicit FP32 reference; casts alone are not a precision guarantee. Set justified tolerances before qualification.
- [ ] Freeze a protocol lock containing versions, parameterization, initialization, recipe, budgets, splits, metric weighting, evaluation grids, and stopping/exclusion rules before corrected research training.

Exit gate: both endpoints learn the easy tasks; a validation-selected recipe works on the intended difficulty; numerical checks and budget accounting pass; the protocol lock exists.

### Stage 5 — Harden provenance, skipping, failure handling and analysis

- [ ] Build scientific task identity from the full effective model/data/calibration/training/evaluation protocol, not a short field whitelist. Include relevant source-content hashes and a reproducible source snapshot or patch if dirty.
- [ ] Validate the stored task identity before overwriting any task configuration. A changed scientific setting must produce an incompatibility error or a new identity, never a silent skip.
- [ ] Permit harmless operational changes such as log locations only under an explicit documented compatibility policy; changes to scientific semantics must invalidate reuse. Record training concurrency and evaluation concurrency separately.
- [ ] Require lifecycle, identity, checkpoint hash, completed accepted steps, and mandatory evaluation artifacts for a complete run. Missing checkpoints in a copied metrics-only archive must not count as locally resumable training evidence.
- [ ] Keep interrupted training non-resumable unless a complete optimizer/scheduler/scaler/RNG/data-position recovery contract is implemented and verified. Do not imply that extending seed count resumes a partial optimizer run.
- [ ] Exclude incomplete/failed records from eligible aggregate estimates while retaining explicit exclusions and seed-level records. Fail the primary paired interaction if required cells or prespecified seeds are missing; do not silently substitute available-case pairs for the planned cohort.
- [ ] Report threshold failures explicitly. Avoid presenting mean capacity among only successful seeds as an unconditional mean; use an explicit censored/not-achieved representation and include every seed in the table.
- [ ] Prespecify the primary granularity-by-initialization interaction, secondary contrasts, uncertainty method, missing-data policy, and multiple-comparison treatment. Use training seeds as the unit of model replication, with example-level precision clearly separate.
- [ ] Keep seed 42's inspected legacy test outcomes exploratory. Use fresh versioned evaluation data for corrected confirmation and record which decisions used pilot outcomes.

Exit gate: intentionally changed model/data/calibration/evaluation settings are rejected for reuse; failed/incomplete runs cannot enter primary summaries unnoticed; provenance and the planned seed cohort are reconstructable.

### Stage 6 — Repair efficiency measurements and worker qualification

- [ ] Separate training parallelism from inference measurement. Capacity pilots may exercise concurrent evaluation, but publishable latency/throughput/energy benchmarks must run with isolated GPU access and recorded concurrency.
- [ ] Align energy regions with metric labels: prefill, decode-only, and end-to-end generation must have distinct boundaries. Exclude prefill from decode-only energy or explicitly name the combined metric.
- [ ] Reset and sample peak memory per phase; distinguish allocated/reserved memory, recurrent state, convolution cache and attention KV memory. Preserve unavailable power/energy as unavailable.
- [ ] Report actual trainable parameter counts and nominal backbone labels together. Reconcile gate-overhead definitions across metadata and static reports. Use capacity-matched sensitivities if claiming effects independent of added parameters.
- [ ] Label reference-backend timings as implementation measurements on the recorded hardware. Do not generalize Python-loop timings to optimized architectural cost without separate kernel evidence.
- [ ] Distinguish CPU plumbing smoke, independent numerical correctness, GPU execution/precision, two-worker capacity, and research recipe qualification in reports and launch gates.
- [ ] Measure representative sequential versus parallel throughput, per-run slowdown, peak VRAM, and contention using the same corrected workload. Select two workers only when safe and useful; do not infer a twofold speedup from worker count.

Exit gate: benchmark boundaries pass checks, isolated measurements are available, and any selected worker count has workload-specific resource and throughput evidence.

### Stage 7 — Run the minimum corrected pilot and decide whether to scale

- [ ] Estimate total cost from measured corrected training plus calibration, validation, evaluation, isolated benchmarking, and failure allowance. Record the estimate before requesting a seed sweep.
- [ ] Run one fresh paired seed across the four scalar/channel-wise by default/calibrated conditions on both corrected synthetic families, after the earlier gates pass. A tiny CPU smoke or a 20-step GPU capacity pilot does not satisfy this requirement.
- [ ] Compare results with key-free baselines and the symbolic oracle, inspect learning curves and error strata, and review posthoc sharing. Investigate apparent effect reversals before declaring a mechanism.
- [ ] Record a go/no-go decision: scale only when tasks are valid, the recipe works, the result addresses the revised question, and the budget fits. Otherwise repair the relevant stage or narrow scope before additional training.
- [ ] Only after that decision, freeze the replication seed set and whether grouped-4, language modeling, and a second scale are affordable. Do not commit automatically to the old 126-run plan.
- [ ] Refresh the proposal, run guide, claims table, and limitations to match the actual protocol and primary literature. Distinguish a simplified granularity ablation from reproducing full production architectures.

Exit gate: the researcher has a valid corrected pilot, a costed replication proposal, and an explicit decision before further compute is spent.

## Testing Decisions

Tests must evaluate observable scientific behavior, rather than copy the
implementation's own recurrence or assert that artifacts merely exist. Prefer
the existing public interfaces and the highest end-to-end level that can give a
sharp failure signal. Introduce no new production abstraction solely for tests.

The proposed boundaries are:

1. **Existing recurrence interface:** independent equation/reference oracle for
   forward values, state, and gradients; nonzero initial states and nonsymmetric
   channel probes; scalar/grouped endpoints; full versus cached continuation;
   precision and backend comparisons. The audit's scalar and axis counterexamples
   become permanent regressions. Existing cache/backend checks are useful prior
   art, but the oracle must not call the implementation being checked.
2. **Existing dataset and evaluation interfaces:** black-box symbolic-oracle,
   query-key counterfactual, shortcut-baseline, actual-distance, scored-prefix,
   history-conditioning and split-provenance checks. Include the audit's known
   perfect overwrite shortcut and first-query MQAR shortcut. Test the complete
   frozen grid and a bounded randomized set, not just one handpicked example.
3. **Existing worker and launcher commands:** small offline workflows that create
   complete paired artifacts, reject changed scientific identity, retain failures,
   preserve accepted-step accounting through AMP recovery, and enforce stage gates.
   Exercise two-worker plumbing separately from real GPU qualification. Existing
   opt-in launcher smoke and real GradScaler retry tests provide prior art.
4. **Existing analysis and benchmarking entry points:** fixture-driven checks for
   incomplete cells, paired seed counts, non-achieved capacity, metric weighting,
   and known interaction signs. Use controlled clocks/energy providers to check
   measurement boundaries, followed by isolated hardware measurements when the
   GPU environment is available. Require serial benchmarking during a parallel
   training workflow.

Run fast CPU regressions first. Gate GPU runs on their success and on a clean,
versioned protocol. GPU-only checks must report unavailable/skipped on CPU hosts;
they must not be represented as a passed production qualification. Record
commands, versions, hardware, outcomes and material numerical tolerances.

The user requested a local spec rather than publication. Test-boundary approval
for implementation remains to be confirmed; it is not inferred from elapsed time
or from choosing local storage.

## Out of Scope

- Reframing the incorrect legacy recurrence as a newly proposed architecture.
- Editing, deleting, or retroactively relabeling legacy training results.
- Starting new training, changing production launch settings, or implementing
  this specification as part of writing the plan.
- Assuming all previously inspected artifacts establish standard-model results.
- Automatically launching five seeds, the full factorial, or larger scales.
- Building new infrastructure or optimizing kernels before scientific validity.
- Adding unrelated datasets, model families, or broad hyperparameter sweeps.
- Publishing this spec to GitHub/Jira, applying triage labels, or committing and
  pushing it without a further user instruction.

## Further Notes

The audit contains observed counterexamples, not merely speculative risks:
the scalar recurrence yields 0.5 versus the published 0.75 in the minimal case;
channel decay acts on the wrong axis; a key-free overwrite baseline solves all
1,024 held-out queries; and longer nominal sequences can share the same scored
prefix. These are release-blocking scientific requirements.

The existing MQAR posthoc result (94.72% unmodified versus 94.28% scalarized at
inference) is a reason to evaluate optimization explanations, not proof that a
separately trained scalar model would perform equally well. Existing recovered
AMP overflows do not justify discarding a model by themselves and should not
be conflated with the confirmed mathematical/data errors.

Primary mathematical references to pin during Stage 1:

- [Gated DeltaNet: equation 10 and block description](https://arxiv.org/html/2412.06464v3).
- [Kimi Linear: state notation, equation 1 and neural parameterization](https://arxiv.org/html/2510.26692).

This is a repair backlog in dependency order, ready for review before implementation,
not a completed verification or authorization to begin a GPU research sweep. Preserve
the stage-gate evidence with the resulting protocol so readiness is auditable.
