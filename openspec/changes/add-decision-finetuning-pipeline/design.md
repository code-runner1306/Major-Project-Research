## Context

The repository already trains and evaluates Gated DeltaNet language models and contains complete 30M TinyStories checkpoints. The selected inputs for this study are scalar seed 43 and channel-wise seed 42. `decision_feasibility.py` verified on a Tesla T4 that both backbones can be fully fine-tuned at batch size 4 and context length 512 with roughly 3 GiB peak reserved memory, but its fixed five-class head, 48/16 development sample, and loss-only report cannot answer the research question.

Typed Decisions contains four workflows with 300 training and 100 official-test cases per workflow. Each case contains five typed questions and soft probability targets. The production pipeline must therefore support variable option sets and all `choice`, `score`, and `noul` primitives, prevent case leakage, survive Colab interruptions, and compare predictions produced by locally executed models under one evaluator.

The primary users are the researchers running separate Colab sessions and the paper authors consuming immutable predictions, metrics, tables, and figures. Free-tier session lifetime, mounted-Drive latency, external checkpoint archives, and a one-model-per-architecture budget constrain the design.

## Goals / Non-Goals

**Goals:**

- Convert the two selected pretrained 30M GDN checkpoints into variable-option typed decision models without changing the tokenizer vocabulary.
- Train on all four workflows and all three primitives using soft gold distributions and one locked protocol.
- Preserve deterministic data identity, training provenance, exact optimizer-step resumption, and recoverable Colab checkpoints.
- Select models without consulting the official test labels, fit calibration on a separate holdout, and emit paper-grade quality, calibration, robustness, and efficiency results.
- Execute Laya Typed-Decisions locally as the required direct baseline through the same normalized evaluator.
- Retain a CPU-safe smoke path and the existing feasibility pilot.

**Non-Goals:**

- Re-pretraining the GDN backbones or adding 55M/100M models.
- Claiming a seed-controlled causal effect of gating; the selected scalar and channel-wise checkpoints have different pretraining seeds.
- Reproducing Laya or Kev training.
- Treating published Jev values as measurements from this experiment.
- Supporting unrestricted text generation, arbitrary open-ended answers, or production automation of consequential decisions.
- Making Kev execution, adversarial benchmarks, or long-context extrapolation prerequisites for the initial study.

## Decisions

### 1. Build a separate research pipeline around shared modules

The implementation will add dedicated data, model, training/checkpoint, evaluation, baseline, and orchestration modules. `decision_feasibility.py` remains a disposable fit-and-timing probe and will not become the production entry point. The orchestrator exposes explicit prepare, train, calibrate, evaluate, baseline, and report stages so an interrupted Colab session can continue from the first incomplete compatible stage.

Alternative considered: expand the feasibility script in place. Rejected because its fixed-head assumptions and deliberately minimal artifacts would make test isolation and resumption difficult to audit.

### 2. Use a dynamic pointer-style option scorer

Each decision is represented as one sequence containing the canonical state, one question, all option labels/descriptions, and a final decision marker. Tokenization records every option span. The GDN produces hidden states once for the sequence. With projection width `r=128`, the head computes:

`q = W_q h_decision`

`o_i = W_o mean(h_option_i)`

`z_i = q dot o_i / sqrt(r)`

`p = softmax(z over valid options)`

Padding options are masked. `choice` criteria become candidates, ordered `score` levels remain ordered candidates, and `noul` becomes explicit yes/no candidates. The head is shared across workflows, question IDs, primitives, and option counts. It adds approximately 82K projection weights at `d_model=320`, plus small biases, and preserves the approximately 30M scale.

The model uses existing GPT-2 tokens and tracked spans rather than resizing the pretrained embedding table. Training deterministically permutes option order while remapping targets; canonical and permuted orders are retained for evaluation.

Alternative considered: one fixed classifier per question. Rejected because it memorizes dataset schemas and cannot represent unseen option counts. Alternative considered: one full backbone pass per candidate. Rejected as the primary design because it repeats state computation and weakens the efficiency claim.

### 3. Split at case level and keep the official test set locked

Within each workflow's 300-case training split, a persisted deterministic split assigns 240 cases to training, 30 to validation, and 30 to calibration. Across four workflows this yields 960/120/120 cases and 4,800/600/600 decisions. All five questions from a case remain together. The released 400-case test split remains untouched until the configuration, selected checkpoints, and temperatures are locked.

Dataset revision/fingerprint, case IDs, split seed, question schemas, class distributions, and content hashes are persisted. The pipeline rejects overlaps or unexpected cardinality/schema changes instead of silently rebuilding a different experiment.

Alternative considered: a random decision-level split. Rejected because questions from the same state would leak across partitions.

### 4. Preserve the decision schema when fitting a 512-token backbone

The primary context budget is 512 because the backbones were pretrained at that length. Question instructions, every option, delimiters, and the decision marker are never truncated. Only the state may be shortened. A deterministic head-and-tail policy retains both ends of an overlength state and records the original length, retained length, and truncation flag. The retained canonical text is persisted so GDN and baseline adapters receive the same semantic content.

The production configuration stays at 512 if the training audit shows at least 99% of decisions fit. Otherwise a pre-test pilot may qualify a larger budget such as 768; the selected value and rationale become part of the immutable run fingerprint. Test labels do not influence this decision.

Alternative considered: prefix-only state truncation. Rejected because important recent events often occur at the end of a state.

### 5. Lock one soft-supervision training protocol for both variants

The loss is soft-target cross-entropy, `-sum_i y_i log p_i`. Both variants use the same split, fine-tuning seed, ordered minibatches, option permutations, maximum epochs, optimizer, schedule, and effective batch size. The starting production profile is full-backbone FP16 fine-tuning for at most three epochs with AdamW, gradient clipping at 1.0, 5% warmup, cosine decay, backbone learning rate selected from `{1e-5, 2e-5}`, and head learning rate `1e-4`.

A bounded development pilot evaluates the two backbone learning rates using validation NLL aggregated across both variants. Once selected, hyperparameters are frozen before official-test evaluation. Physical batch size is qualified between 4 and 8 on the assigned T4; gradient accumulation supplies an effective batch size of 16 or 32. The manifest records microbatch and effective batch sizes separately.

The final optimizer-step budget is derived from the persisted training size and fixed epoch count. The lowest validation soft NLL chooses `best_model`; `final_model` always represents the completed fixed budget. Both are retained and distinguished in reporting.

Alternative considered: tune each architecture independently. Rejected because unequal search could explain observed differences.

### 6. Make checkpoints crash-safe, fingerprinted, and exact at optimizer boundaries

Every run has a schema-versioned fingerprint covering source revision/dirty state, model architecture and variant, pretrained-checkpoint SHA-256, dataset revision/fingerprint and split hash, tokenizer/version, serialization/truncation rules, training/evaluation configuration, seeds, package/runtime versions, expected artifacts, and baseline revision.

Lifecycle states are written atomically and progress through `pending`, `training`, `trained`, `calibrating`, `calibrated`, `evaluating`, and `complete`; `failed` carries a structured reason. A resumable checkpoint contains model/head, optimizer, scheduler, scaler, epoch, completed optimizer step, sampler position, gradient-accumulation boundary, RNG states, and best-validation state. Saves occur only after a complete optimizer step.

Two rolling files, `resume_A.pt` and `resume_B.pt`, are written locally, checksummed, copied to the output root, validated, and then referenced by an atomically updated pointer. The previous valid slot remains recoverable. Saves occur every 100 optimizer steps and at epoch/stage boundaries. Model-only `best_model.pt` and `final_model.pt` are separately checksummed. Only the two rolling resume files are retained to bound Drive storage.

Resume rejects a mismatched fingerprint and never treats an incomplete/corrupt artifact as complete. A trained run can resume directly at calibration or evaluation.

### 7. Separate selection, calibration, and final evaluation

Validation soft NLL selects the checkpoint. The calibration partition fits positive temperature parameters by minimizing soft NLL, with separate temperatures for `choice`, `score`, and `noul` when all groups are valid. Raw and calibrated predictions are preserved.

The official test produces top-1 accuracy, soft accuracy, soft NLL, Brier score, ECE, and score expected-value MAE, with workflow, primitive, option-count, and truncation breakdowns. Metric definitions and reduction conventions are schema-versioned. Option-order robustness uses the canonical order plus at least three deterministic permutations and reports flip rate and metric deltas.

Uncertainty is estimated with a deterministic paired bootstrap over 400 cases, not 2,000 decisions, preserving within-case dependence. Reports include 95% intervals for each model and paired scalar-minus-channel-wise and GDN-minus-Laya differences. Test prediction rows are immutable inputs to report regeneration.

Alternative considered: fit calibration on validation or test predictions. Rejected because checkpoint selection already consumes validation and test fitting would invalidate the estimate.

### 8. Treat Laya as the required direct baseline and Jev as contextual

`convaiinnovations/laya-typed-decisions` is pinned by model revision, loaded once, and run locally on the same normalized official-test decisions. Its outputs are adapted into the same per-option prediction schema and scored by the repository evaluator. Raw native outputs and any common held-out recalibration are reported separately.

Efficiency measurements run sequentially on the same GPU/runtime with batch size 1, fixed examples, 20 warmups, at least 200 synchronized timed observations, and no diagnostic hooks. Reports include parameter count, checkpoint size, peak allocated/reserved memory, median/p95 latency, decisions/second, and five-question cases/second. Because implementations may batch or share state differently, both per-decision and per-case timings are mandatory.

The optional resource record preserves total GPU-energy counter deltas or sampled-power estimates, power/utilization/memory/clock/temperature samples, visible other-process counts, and device/software identity. It also records synchronized per-example latency with primitive, full serialized input length, and option count, summarized by those groups. Fine-tuning sessions accumulate wall-clock time and energy across Colab resumes and report completed optimizer steps; all such training cost excludes the already completed TinyStories pretraining. Unavailable measurements remain explicitly unknown and do not block the run.

Kev is an optional adapter after the core study. Published Jev numbers may appear only in clearly labeled contextual tables and must not be mixed with locally measured paired differences.

### 9. Optimize the workflow for separate Colab sessions

The output root is configurable and may point to mounted Drive. Dataset/model downloads and active batches use Colab-local storage; only manifests, lifecycle updates, rolling checkpoints, predictions, metrics, and final exports are persisted to the output root. Scalar and channel-wise training are independent resumable invocations sharing one immutable run manifest and split identity. A final packaging command validates required artifacts and creates a compressed research-results archive without duplicating source pretrained checkpoints.

## Risks / Trade-offs

- **[Risk] One selected checkpoint per architecture cannot estimate training-seed variance.** -> Describe the result as a selected-checkpoint comparison, use paired case-level intervals only for evaluation uncertainty, and avoid causal gating claims.
- **[Risk] TinyStories pretraining is substantially out of domain.** -> Treat adaptation quality as an empirical question, include majority/random baselines, and retain negative results rather than changing the test protocol.
- **[Risk] Causal option representations can retain order effects.** -> Randomize order during training and make permutation robustness a required result.
- **[Risk] State truncation may remove decisive evidence.** -> Audit before training, preserve both ends, expose truncation breakdowns, and qualify a larger context before test access if the 99% fit criterion fails.
- **[Risk] Colab disconnects during a large Drive write.** -> Alternate two checksummed resume slots and update the latest pointer only after validation.
- **[Risk] Laya's native calibration and interface differ from the GDN pipeline.** -> Preserve native raw outputs, run all models through one metric implementation, and report common held-out recalibration separately.
- **[Risk] Direct latency comparisons can reward different batching semantics.** -> Report both per-decision and per-five-question-case timing with explicit execution protocols.
- **[Risk] The dataset is synthetic and gold reflects teacher distributions.** -> Frame accuracy as agreement with benchmark gold and avoid claims about real-world autonomous decision safety.

## Migration Plan

1. Add the normalized data schema, deterministic split manifest, serializer, length audit, and CPU fixtures without changing existing experiment entry points.
2. Add the dynamic GDN decision head and verify variable options, all primitives, target remapping, masking, and a tiny-batch overfit test.
3. Add the trainer, fingerprint, lifecycle, rolling checkpoints, exact resume, model-only exports, and CPU smoke orchestration.
4. Add validation, calibration, official-test guards, metrics, bootstrap, prediction exports, tables, and figures.
5. Add the pinned Laya adapter and same-device benchmark behind an optional dependency boundary, then make its completed artifact mandatory for final report packaging.
6. Add Colab documentation and validate a short interruption/resume run before launching scalar and channel-wise production sessions.
7. Keep prior TinyStories outputs and feasibility reports unchanged. Rollback consists of removing the new decision output root and invoking the existing pretraining/feasibility entry points.

## Open Questions

- Whether batch size 8 improves T4 throughput without instability; resolve with the required pre-test capacity pilot and lock the result in the manifest.
- Whether the complete training length audit meets the 99% fit criterion at 512 tokens; resolve before production training and before reading official-test labels.
- Which exact Laya package/model revisions are compatible with the Colab PyTorch environment; resolve during adapter smoke testing and pin both revisions before baseline inference.
