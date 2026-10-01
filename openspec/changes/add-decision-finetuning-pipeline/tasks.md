## 1. Pipeline Foundations

- [x] 1.1 Add decision-research configuration dataclasses for data identity, model/head settings, training, calibration, evaluation, baseline, artifact schemas, and smoke/production profiles.
- [x] 1.2 Add a staged decision-research command-line entry point with prepare, preflight, tune, train, calibrate, evaluate, baseline, report, package, and resume operations.
- [x] 1.3 Define the schema-versioned run directory layout and typed JSON/JSONL readers and writers for manifests, lifecycle, predictions, metrics, and artifact inventories.
- [x] 1.4 Isolate optional Laya dependencies so core development and CPU smoke tests install and run without downloading the baseline model.
- [ ] 1.5 Add small local fixtures covering `choice`, `score`, and `noul`, variable option counts, soft labels, long states, and five questions sharing one case.

## 2. Typed Decision Data

- [x] 2.1 Implement pinned loading and schema/cardinality validation for all four Typed Decisions workflows without using the official test split during development stages.
- [x] 2.2 Implement deterministic per-workflow 240/30/30 case-level train/validation/calibration assignment and persist reusable split IDs and hashes.
- [x] 2.3 Add overlap checks for case IDs and normalized state content across train, validation, calibration, and the official test identity.
- [x] 2.4 Normalize every case into five decision records while preserving workflow, case/question IDs, primitive, instructions, candidate labels/descriptions, ordered score levels, and soft gold probabilities.
- [x] 2.5 Normalize `noul` questions into explicit yes/no candidate distributions and validate every target for finiteness, non-negativity, label coverage, and normalization.
- [x] 2.6 Implement canonical GPT-2 serialization that returns token IDs, decision position, candidate token spans, target vectors, and provenance without adding tokenizer tokens.
- [x] 2.7 Implement deterministic state-only head-and-tail truncation that never truncates question/options and records original/retained lengths and canonical retained text.
- [x] 2.8 Implement a collator for variable sequence lengths and candidate counts with explicit token, candidate, span, and target masks.
- [x] 2.9 Implement deterministic training-time option permutation with correct span/target remapping and canonical-order restoration.
- [x] 2.10 Emit the data audit and canonical manifest with counts, hashes, schemas, length distributions, truncation rate, and the 99-percent 512-token fit gate.
- [ ] 2.11 Add unit tests for cardinalities, case-level isolation, normalization, dynamic spans, padding masks, truncation invariants, option permutation, and manifest reuse/rejection.

## 3. Dynamic GDN Decision Model

- [x] 3.1 Add a supported backbone hidden-state encoding path that reuses token embeddings, all GDN/attention blocks, and final RMSNorm without computing the language-model vocabulary head.
- [x] 3.2 Implement the configurable 128-dimensional query/option projection head with candidate-span mean pooling, scaled dot-product logits, and candidate masking.
- [x] 3.3 Implement strict loading for scalar seed 43 and channel-wise seed 42, including variant/configuration checks and pretrained-checkpoint SHA-256 provenance.
- [x] 3.4 Implement dynamic decision-model save/load metadata so model-only checkpoints reconstruct the backbone and head without optimizer state.
- [ ] 3.5 Add forward and gradient tests for all primitives, variable candidate counts, padded candidates, finite normalized probabilities, and no vocabulary resize.
- [ ] 3.6 Add a tiny-data overfit test that demonstrates substantial soft cross-entropy reduction and verifies that candidate remapping remains correct.

## 4. Reproducible Fine-Tuning

- [x] 4.1 Implement masked soft-target cross-entropy, finite-value guards, gradient norm tracking/clipping, and structured failure reporting.
- [x] 4.2 Implement separate backbone/head AdamW parameter groups, 5-percent warmup, cosine decay, FP16 autocast/scaling, and configurable microbatch/gradient accumulation.
- [x] 4.3 Implement a deterministic epoch sampler and option-order generator whose next minibatch can be reconstructed exactly after an optimizer-boundary resume.
- [x] 4.4 Implement validation passes at configured boundaries and lowest-validation-NLL best-model selection with a deterministic tie break.
- [x] 4.5 Implement completed-budget final-model export distinct from the selected best model.
- [x] 4.6 Implement the bounded common learning-rate comparison over `1e-5` and `2e-5`, aggregate validation NLL across both variants, and persist an immutable locked training profile.
- [x] 4.7 Reject paired production runs whose controlled data, seed, order, budget, optimizer, scheduler, context, or effective-batch fields differ by variant.
- [ ] 4.8 Add a CPU smoke trainer and GPU preflight mode covering all primitives, mixed precision when available, a short overfit, model reload, and interruption/resumption.
- [ ] 4.9 Add trainer tests for effective batch accounting, scheduler progression, best/final distinction, non-finite failure, and paired-profile validation.
- [x] 4.10 Persist per-session and cumulative fine-tuning wall time, optimizer-step counts, device conditions/activity, and optional GPU energy across resumed sessions.

## 5. Colab-Safe Checkpointing and Resumption

- [ ] 5.1 Implement the canonical run fingerprint over source, pretrained checkpoints, model/head, data/splits, tokenizer/serialization, training/evaluation, seeds, runtime, expected artifacts, and baseline identity.
- [x] 5.2 Implement atomic lifecycle transitions for pending, training, trained, calibrating, calibrated, evaluating, complete, and failed states.
- [x] 5.3 Implement complete optimizer-boundary resume payloads containing model/head, optimizer, scheduler, scaler, epoch/step, accumulation boundary, sampler position, RNG states, and best-validation state.
- [x] 5.4 Implement local-temporary writes, SHA-256 validation, alternating `resume_A.pt`/`resume_B.pt`, and an atomically updated latest pointer for external output roots.
- [x] 5.5 Implement fallback to the older rolling slot when the pointed checkpoint is missing, partial, unloadable, or checksum-invalid.
- [x] 5.6 Implement checksummed model-only `best_model.pt` and `final_model.pt` exports and strict evaluation-time compatibility validation.
- [x] 5.7 Implement stage-aware resume so trained models continue at calibration and calibrated models continue at evaluation without retraining.
- [x] 5.8 Bound retained resume artifacts to two slots per variant and exclude source pretrained checkpoints from final result packages.
- [ ] 5.9 Add interruption, corruption, mismatch, alternate-slot fallback, exact-next-batch, atomic lifecycle, and evaluation-only recovery tests.

## 6. Calibration and Scientific Evaluation

- [ ] 6.1 Define and validate immutable prediction rows containing model/run identity, case/question metadata, labels, option order, raw logits/probabilities, gold distribution, truncation, and timing fields.
- [ ] 6.2 Implement inference over validation, calibration, and official-test partitions with complete case/question coverage and canonical-label restoration.
- [x] 6.3 Enforce a test-access guard requiring a locked training profile, validated best checkpoint, and compatible calibration stage before official-test predictions.
- [x] 6.4 Fit positive held-out temperatures for `choice`, `score`, and `noul` by calibration soft NLL and preserve raw and calibrated channels.
- [x] 6.5 Implement versioned accuracy, soft accuracy, soft NLL, multiclass Brier, ECE, and expected-score MAE definitions with aggregate and grouped reductions.
- [x] 6.6 Implement workflow, primitive, option-count, and truncation breakdowns with completeness and finite-value validation.
- [ ] 6.7 Implement canonical plus at least three deterministic option-order evaluations, canonical remapping, flip rate, and quality/calibration deltas.
- [x] 6.8 Implement deterministic paired case-level bootstrap intervals for individual metrics and scalar-minus-channel-wise/model-minus-Laya differences.
- [ ] 6.9 Add metric reference tests, calibration isolation tests, test-guard tests, prediction-coverage failures, permutation invariance/remapping tests, and case-level bootstrap tests.
- [x] 6.10 Save synchronized per-decision latency with full serialized token length and option count, then summarize by primitive, length bucket, and option count.

## 7. Laya Baseline and Efficiency Benchmark

- [ ] 7.1 Verify a Colab-compatible Laya package/model combination and pin the exact package and `convaiinnovations/laya-typed-decisions` revisions in configuration and documentation.
- [ ] 7.2 Implement and runtime-validate a resident Laya adapter for all three primitives using the same canonical retained state, questions, options, case IDs, and official-test coverage.
- [ ] 7.3 Normalize and runtime-validate Laya native outputs into complete canonical per-option prediction rows while preserving native calibration metadata and rejecting missing/invented labels.
- [x] 7.4 Apply the shared evaluator to Laya and optionally fit/report the same held-out recalibration protocol without overwriting native outputs.
- [ ] 7.5 Implement a sequential same-device efficiency harness with batch size 1, 20 warmups, at least 200 synchronized observations, and diagnostics excluded from timing.
- [ ] 7.6 Report parameter count, checkpoint bytes, peak allocated/reserved memory, median/p95 latency, decisions per second, and five-question cases per second for both GDN variants and Laya.
- [ ] 7.7 Add mocked baseline-adapter tests, label/schema validation tests, timing-protocol tests, and direct-versus-contextual result classification tests.
- [x] 7.8 Record optional NVML energy telemetry for official-test inference and GDN option-order robustness, persist unsupported status without blocking evaluation, and include the artifact in result packaging.
- [x] 7.9 Save runtime/device metadata and sampled utilization, memory, clocks, power, and visible other-process activity with each measured training/evaluation stage.

## 8. Reports, Figures, and Packaging

- [x] 8.1 Generate machine-readable aggregate metrics and paired comparisons from validated immutable predictions rather than from training logs.
- [ ] 8.2 Generate paper-ready CSV and Markdown tables for overall metrics, workflow/primitive breakdowns, calibration, option-order robustness, and efficiency.
- [ ] 8.3 Generate academic reliability diagrams, quality/calibration comparison plots, robustness plots, and efficiency trade-off figures with deterministic styles and source-data exports.
- [ ] 8.4 Include selected-checkpoint, differing-pretraining-seed, synthetic-teacher, truncation, and external-Jev-comparison limitations in the generated research summary.
- [ ] 8.5 Implement final completeness validation requiring scalar, channel-wise, Laya, calibration, test, robustness, bootstrap, and same-device efficiency artifacts while allowing Kev to be absent.
- [x] 8.6 Implement a checksummed compressed results package and artifact inventory that excludes source pretrained checkpoints and temporary/duplicate resume data.
- [ ] 8.7 Add report regeneration and package completeness tests using fixture prediction artifacts.

## 9. Colab Workflow and Documentation

- [ ] 9.1 Document environment setup, Drive mounting, checkpoint archive acquisition, local cache use, output-root selection, and staged commands without heredoc syntax.
- [ ] 9.2 Document separate scalar and channel-wise sessions, automatic resume behavior, lifecycle/log inspection, expected checkpoint files, and safe Colab disconnect recovery.
- [ ] 9.3 Document data audit, batch-size/context qualification, common learning-rate lock, calibration, one-time official test, Laya baseline, reporting, packaging, and download commands.
- [ ] 9.4 Document measured runtime/storage estimates and clearly distinguish feasibility pilots, development tuning, production training, and final research results.
- [ ] 9.5 Update project dependency metadata and ignore rules for generated decision checkpoints, archives, predictions, and local baseline caches without hiding source fixtures.

## 10. Verification and Research Execution

- [ ] 10.1 Run the full local unit/integration suite and the CPU decision smoke profile in an isolated output directory.
- [ ] 10.2 Run the Colab T4 GPU preflight, interruption/resume test, and batch-size 4-versus-8 capacity/throughput qualification; record and lock the selected physical/effective batch profile.
- [ ] 10.3 Run the full training-data length audit and either lock 512 tokens after passing the 99-percent fit gate or qualify and document a larger context before production.
- [ ] 10.4 Execute both-variant learning-rate development runs, choose the common profile by aggregate validation NLL, and freeze the immutable production/test manifest.
- [ ] 10.5 Execute or resume the complete scalar production fine-tune and verify rolling, best, final, history, lifecycle, and checksum artifacts on Drive.
- [ ] 10.6 Execute or resume the complete channel-wise production fine-tune under the identical locked protocol and verify all corresponding artifacts.
- [ ] 10.7 Calibrate both selected best checkpoints, then perform the one-time complete official-test and option-order evaluations.
- [ ] 10.8 Execute the pinned Laya official-test and same-device efficiency runs and validate common evaluator coverage.
- [ ] 10.9 Generate final metrics, paired case-level intervals, tables, figures, limitations, package inventory, and checksummed research-results archive.
- [ ] 10.10 Independently reload the packaged model-only checkpoints and regenerate a sample of predictions and all derived reports before declaring the study complete.
