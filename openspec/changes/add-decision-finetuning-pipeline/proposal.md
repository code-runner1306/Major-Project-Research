## Why

The existing `decision_feasibility.py` demonstrates that full fine-tuning of the selected 30M scalar and channel-wise Gated DeltaNet checkpoints fits on a Colab T4, but it is not a research training pipeline: it covers one fixed question, uses a small internal split, reports only loss and runtime, and does not save resumable or final fine-tuned checkpoints. The project now needs a reproducible, Colab-resilient pipeline that can train and evaluate both selected checkpoints as general typed decision models against a directly executed open baseline.

## What Changes

- Add deterministic preparation of all four `LocalLLaMA/typed-decisions` workflows, expanding each case into its five decisions while keeping case-level train, validation, calibration, and official test boundaries intact.
- Add a shared dynamic option-scoring head for the 30M Gated DeltaNet backbone that supports variable option counts and the `choice`, `score`, and `noul` primitives without adding tokenizer vocabulary.
- Fine-tune the selected scalar seed-43 and channel-wise seed-42 TinyStories checkpoints under one locked training protocol with soft-distribution supervision, mixed precision, option-order randomization, and validation-based model selection.
- Add atomic, rolling, resumable checkpoints suitable for interrupted Colab sessions, plus model-only best and final checkpoints and complete provenance manifests.
- Add held-out temperature calibration and paper-grade decision metrics, workflow/primitive breakdowns, option-order robustness measurements, paired case-level uncertainty estimates, prediction exports, and research figures.
- Add a locally executed `convaiinnovations/laya-typed-decisions` adapter and same-device efficiency benchmark as the required direct baseline. Keep Kev support optional and treat published Jev values as contextual rather than directly comparable measurements.
- Preserve `decision_feasibility.py` as a small feasibility probe; the production research pipeline is introduced separately.

## Capabilities

### New Capabilities

- `typed-decision-data`: Deterministic loading, case-level splitting, serialization, dynamic option representation, token-length auditing, and leakage prevention for all Typed Decisions workflows and primitives.
- `gdn-decision-finetuning`: Dynamic option scoring and reproducible full-backbone fine-tuning for the selected scalar and channel-wise 30M GDN checkpoints.
- `decision-checkpoint-resumption`: Atomic lifecycle, rolling resumable checkpoints, model-only exports, integrity checks, and exact continuation across Colab sessions.
- `decision-model-evaluation`: Locked validation, calibration, official-test evaluation, calibration/quality/robustness metrics, paired uncertainty estimates, and research artifact generation.
- `decision-baseline-benchmarking`: Direct Laya Typed-Decisions inference and same-hardware parameter, memory, latency, and throughput comparison using the shared evaluation protocol.

### Modified Capabilities

None. This change adds a separate decision-research pipeline and does not alter the requirements of the existing pretraining and synthetic-retrieval experiment families.

## Impact

- New decision data, model, training, checkpoint, evaluation, baseline, and orchestration modules will be added alongside the existing TinyStories pipeline.
- The selected pretrained checkpoints remain external binary inputs and continue to be excluded from Git.
- The output tree gains schema-versioned decision-run manifests, split identities, lifecycle state, rolling training checkpoints, model-only exports, predictions, metrics, tables, and figures.
- Runtime dependencies may expand to include the official Laya package and metric utilities required for direct baseline execution and calibration.
- Colab documentation will describe Drive-mounted checkpoint storage, resumption, separate scalar/channel-wise sessions, and final artifact download.
- Tests will cover all decision primitives, variable option counts, split leakage, truncation, checkpoint interruption/resumption, metric correctness, baseline adaptation, and a CPU-safe smoke path.
