# Methodology audit of the completed exploratory study

Audit date: 2026-11-13. Code reviewed at maink commit 9f2cc3a;
completed artifacts: output/granularity_exp_1, seed 42, eight conditions.

## Decision

Pause additional research training with this implementation. Two mathematical
discrepancies and a reproducible overwrite-data shortcut prevent treating these
results as a controlled study of published Gated DeltaNet/KDA retention versus
overwrite behavior. Additional seeds would quantify the same implementation and
task artifacts, rather than resolve these issues. The research question remains
viable, but the present result needs to be treated as an exploratory result of a
custom recurrence on custom tasks. Correcting the method requires a new protocol
identity/output root and fresh training; reevaluating old checkpoints cannot
retroactively change how they were trained.

This is a source/artifact audit with CPU numerical counterexamples and dataset
baselines. It does not certify GPU kernels, independently reproduce training, or
inspect the trained tensors: the production checkpoints were not included in the
pulled artifacts. No training/model/data implementation was changed during this
audit. The probe is research_notes/methodology_audit_probe.py.

## Confirmed blockers

### 1. The residual in the recurrence is evaluated before decay (critical)

recurrence.py:114-117 computes, in value-by-key state storage:

    error = v - S @ k
    S_new = alpha * S + beta * error @ k.T

The published scalar GDN update is equivalent to:

    decayed = alpha * S
    error = v - decayed @ k
    S_new = decayed + beta * error @ k.T

The old-state subtraction in the implementation is missing a factor of alpha.
For the one-dimensional case S=1, k=q=v=1, alpha=beta=0.5, the implemented state
is 0.5 and the published state is 0.75. The CPU probe reproduces this difference.
The implementations agree in some special cases, including zero initial state
for a single token; those cases are insufficient oracles.

Source: [Gated DeltaNet, equation 10](https://arxiv.org/html/2412.06464v3#S3.SS1).

Impact: the experiment does not currently implement the named scalar baseline.
In this custom update, changing alpha changes the balance between decay and an
undecayed erase term, so the apparent initialization benefit cannot be assigned
the published mechanism's interpretation. Existing backend and cache tests often
compare implementations to this same local reference, which cannot establish
agreement with the published equation. Add an independent equation oracle before
training corrected models.

### 2. Channel decay acts on the value axis, rather than KDA's key axis (critical)

The outer product error @ k.T and read S @ q establish value-by-key storage.
alpha.unsqueeze(-1) therefore scales value rows. KDA's diagonal decay operates
on the key axis; in this storage convention it scales columns before the
delta update. Equal key/value dimensions conceal this semantic difference in
shape tests. Transposing the published state convention does not remove it.

With beta=0, S=[[1,2],[3,4]], alpha=[0.2,0.8], the implementation gives
[[0.2,0.4],[2.4,3.2]], whereas key-axis decay gives
[[0.2,1.6],[0.6,3.2]]. The probe reproduces both arrays.

Source: [Kimi Linear, notation and equation 1](https://arxiv.org/html/2510.26692#S3).

Impact: the channel-wise arm is not the claimed KDA-style granularity extension.
Fix axis and update ordering together, and verify forward values, gradients,
cached continuation, and any fused backend against an independent oracle.

### 3. A baseline ignoring all queried keys solves overwrite at 100% (critical for overwrite claims)

retrieval.py:_overwrite_v3 forces the final target write to be immediately before
one final unrelated write for the current distance grid. The first query always
asks for the target; the second always asks for that final unrelated association.
Consequently the first answer is the penultimate value in the memory block and
the second answer is its final value. The query keys need never be examined.

The probe reads only value tokens strictly before the first query, and returns
the final two values in that order. On the exact seed-42 held-out generator:
**1,024/1,024 answers correct over 512 examples**, without using a queried key,
stale-value labels, learned parameters, or gold answers from the query block.

Impact: 50.29% versus 33.20%, or zero stale-value errors, does not establish
key-selective overwrite ability. Randomize final relevant-write positions and
query order, add unrelated later writes, and require recency/position baselines
to fail while a keyed last-write-wins oracle succeeds. Version and retrain on
the repaired generator.

### 4. Sequence-length settings change a suffix after the answers (high)

Both v3 generators put the memory and queries near the start, then append filler
to reach the requested length. For fixed sampled symbols, keys, and distances,
the scored causal prefixes are identical at lengths 128 and 512. The probe
confirms this for both families. The current maximum MQAR query-key input index
is 109 (16 keys, requested distance 32); overwrite queries are earlier still.

Impact: plots by nominal sequence length do not demonstrate longer-context
retention or extrapolation. The distance controls are real for the anchors, but
they range only from 8 to 32 tokens, with larger realized distances for other
MQAR queries. Insert controlled content before queries to vary actual retrieval
distance, then analyze actual distance jointly with key count and distractors.

## High-priority interpretation and design problems

### 5. The default baseline is a custom gate with approximately one-token half-life

model.py uses sigmoid(alpha_proj(x)) with zero study biases. At small initialized
logits this gives alpha approximately 0.5, versus calibrated targets of half-lives
16/32/64/128. Thus calibration moves the model from extremely short initial
retention to a range aligned with the task. The standard GDN/KDA implementations
use additional decay parameterization and block details; our direct sigmoid,
missing Q/K/V post-convolution SiLU, and missing recurrent-output head norm make
this a simplified architectural implementation even apart from findings 1/2.

Sources: [Gated DeltaNet block description](https://arxiv.org/html/2412.06464v3#S3.SS4),
[Kimi Linear neural parameterization](https://arxiv.org/html/2510.26692#S4).

Impact: a large calibration gain is plausible for this deliberately short-memory
baseline and is not evidence that the published defaults are generally defective.
Document the simplifications and include an appropriate standard/default-decay
baseline if making claims about the named architectures.

### 6. Calibration changes both timescales and initial channel-function diversity

model.py:initialize_study_weights gives default channels independently seeded
projection rows; calibrated channels repeat one row per head, and calibration
then changes biases. The factor is therefore a package of timescale adjustment
and within-head function matching. A package comparison is legitimate, but a
claim that timescales alone explain the difference is not identified. Add bounded
controls (bias calibration retaining diverse rows; function matching without the
timescale change) if the paper makes that causal claim. The existing paired
backbone initialization and equal bias availability are positive controls.

### 7. Existing intervention results qualify the apparent MQAR mechanism

The saved calibrated channel-wise MQAR checkpoint achieves 94.719% with all
40 gates per head and 94.279% when their effective alpha values are averaged to
one gate per head at inference. Calibrated scalar trained separately achieves
12.028%. Intervention examples match the original held-out subset hash.

Impact: the performance gap is not evidence that this checkpoint requires large
within-head differences in decay at inference. This is compatible with effects
on optimization, training-time specialization, or other learned weights. It
does not show that a scalar model trained from scratch would attain 94%, nor
that removing gate projection computation would be equally fast or accurate.
The overwrite intervention behaves differently: calibrated C6 drops from 33.20%
to 2.05% when scalarized. The dataset shortcut blocks a strong overwrite memory
interpretation of that difference.

### 8. MQAR also has an easy first-query shortcut; evaluation is teacher-forced

retrieval.py:_mqar_v3 always puts the first queried association last in memory.
Returning the final memory value for that query alone gets 512/4,772 = 10.729%
query accuracy. Calibrated scalar's 12.028% aggregate is close to this descriptive
baseline; aggregate accuracy alone does not establish arbitrary association
retrieval for that arm. Later MQAR queries are genuinely shuffled, so this simple
shortcut does not explain C6's 94.719%.

The inputs also include ground-truth answers to preceding queries. Prediction
at a query's key position is causal and does not see its own future answer, so
this audit did NOT find direct same-answer label leakage. However, these scores
are per-query predictions with oracle history, not free-generation success or
the probability of getting all answers in an example correct. Report that
conditioning and add an answer-free query stream or autoregressive evaluation
if making broader deployment/benchmark claims. This differs from the MQAR
format illustrated in the Kimi paper.

### 9. Scale and capacity claims need correction

The nominal 10M synthetic runs have actual trainable counts of 3,329,784 for
scalar and 3,480,480 for channel-wise (about 4.53% extra). The smaller synthetic
vocabulary and tied output head matter. The metadata records actual counts,
which is good; the paper must use those counts and explain the nominal scale.
The same-backbone comparison can include natural gate overhead, but it is not an
exact parameter-matched ablation. Avoid attributing all effects exclusively to
granularity without an appropriate capacity sensitivity/control.

### 10. Current protocol primarily measures a short-budget learning outcome

The exploratory launcher bypasses E0/E1 recipe/learnability qualification and
uses one LR and 1,000 steps. Default MQAR arms scarcely learn the task. Multiple
seeds under that recipe would estimate a valid fixed-budget package effect on
the custom system, but would not distinguish optimization speed from eventual
ability/capacity. Prove easy-task learnability, provide validation-only recipe
selection with comparable effort, and inspect longer learning curves before
claiming architectural inability. The full E0/E1 workflow is stronger than this
exploratory launcher, but was not the path used for granularity_exp_1.

Seed 42's held-out results have already informed decisions and hypotheses. Keep
them exploratory and prespecify primary contrasts and stopping rules for fresh
confirmation. Additional training seeds are the relevant replicates; thousands
of queries from one trained model are not thousands of independent model runs.

## Pipeline and reporting risks

- **Artifact reuse can conceal method changes.** study_runner.py:task_complete
  checks selected task fields, step/token counts, checkpoint hash, and lifecycle,
  but not the whole current retrieval/calibration/model implementation. The PS1
  overwrites task configs before this skip check. orchestration.py also permits
  source revision drift for study runs; model/data files lack a full content
  snapshot in the source metadata. The stored source records a revision plus
  dirty=true, not a reproducible patch. A method correction must not be silently
  mixed with seed-42 artifacts in the same experiment identity/root.
- **Precision sensitivity remains unmeasured.** Training uses CUDA autocast with
  input-dtype state accumulation; evaluation/calibration run without autocast.
  Finite losses and successful AMP retry handling establish budget/recovery,
  not FP32 numerical equivalence of long recurrent trajectories. Compare a
  bounded FP32 reference trajectory including gradients and retention near one.
  Merely casting operands to float32 inside an active autocast region is not
  enough to guarantee full-precision matmuls.
- **Calibration includes padded/filler tokens.** The first 256 training examples
  are padded to the maximum length, and calibration pools every token equally.
  Its reported mean retention can be dominated by filler, rather than relevant
  write/query states. Report conditional distributions and calibration-domain
  weighting. A half-life derived from a mean alpha is a diagnostic, not a
  measured full-memory lifetime in a delta-rule system.
- **Parallel inference timing is contaminated.** PS1 uses Workers for evaluation
  as well as training, while benchmark_inference times generation in each child.
  Concurrent GPU benchmark results do not estimate isolated latency/cost. Train
  in parallel if desired; remeasure publishable inference timings in isolation.
  The Python reference backend's host-loop overhead also limits generalization
  to optimized kernels. Preserve workload, dtype, hardware and concurrency labels.
- **Energy/memory measurements need qualified interpretation.** The cached-decode
  energy region includes repeated prefill, while the decode latency timer
  excludes prefill. Its joules/generated-token metric therefore includes prompt
  processing. Peak GPU memory is reset once per prompt setting, not separately
  per measured phase. VM energy is unavailable and must remain unavailable.
- **Some analysis helpers are unsafe to interpret without status checks.** E6
  collects exclusions but still accumulates metrics from those records; report
  status can be incomplete while its means exist. Capacity summaries average
  only seeds that reach the threshold, excluding null-capacity seeds from that
  mean. Preserve failures and all seed records in paper tables.
- **Statistical endpoints need explicit weighting.** Exact-query accuracy weights
  16-key MQAR examples more heavily than 4-key examples. Validation averages
  batch cross-entropies. Those are choices, not necessarily errors; report them,
  add example-level summaries if needed, and avoid mixing nominal distance,
  realized per-answer distance, and query-weighted effects.

## Positive checks and bounds on the conclusions

- Labels are scored at query-key positions, before their own answer token.
- Study train/validation/held-out streams use distinct split domains and seeds;
  corresponding conditions share data identities and paired backbone seeds.
- Study default and calibrated arms both have trainable alpha biases. This
  avoids the straightforward confound of adding biases only to calibration.
- Completed artifacts record the intended 1,000 steps, accepted finite gradients,
  and successful AMP loss-scale recovery. Rare recovered overflows are not the
  main invalidating finding here.
- Posthoc grouping explicitly retains the computation and is appropriately
  documented as an inference intervention rather than independently trained groups.
- The current source's CRLF-normalized recurrence hash matches the recorded VM
  implementation hash (24f5a1f82b55560af7f02b4ebca034a6803d7ba9fac735557e68fa0a33418fcc).
  The recurrence findings apply to the implementation recorded for these runs,
  not just an unrelated newer checkout. The Git version at the recorded revision
  also has the same source text.

## Repair order before further research training

1. Choose the intended published recurrence; fix residual ordering and decay axis,
   validate against an independent equation/official implementation, and version
   the model/recurrence. Do not treat cache self-consistency as that validation.
2. Remove query-position/recency shortcuts, separate query targets from fixed write
   positions, and make context/distance difficulty causally meaningful. Version
   the dataset. Require key-free baselines to fail the intended task.
3. Specify simplified versus standard architecture/default gate parameterization,
   actual parameter counts, and the exact causal claim for calibration.
4. Lock a feasible validation-selected training protocol and primary metrics;
   establish learnability before any broad factorial/multi-seed sweep.
5. Start corrected runs in a fresh identity/root. Keep granularity_exp_1 immutable
   as exploratory legacy evidence. Repeat seeds only after blockers 1/2 are fixed.
6. Measure inference cost separately from parallel training/evaluation workloads,
   and interpret the saved posthoc intervention before making mechanism claims.

The right decision is to repair and reassess the pilot, not spend more compute
replicating the current evidence as though it validates the proposed paper.
