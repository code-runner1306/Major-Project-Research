## 1. Runtime and Configuration Foundation

- [x] 1.1 Add Linux/Python 3.11 `requirements.txt` with `torch==2.12.0+cu126`, the official `https://download.pytorch.org/whl/cu126` index, all runtime/reporting packages, `nvidia-ml-py`, and `pytest`.
- [x] 1.2 Align `pyproject.toml` and lock metadata with Python 3.11 and dependency constraints compatible with `requirements.txt`.
- [x] 1.3 Implement a CUDA/driver preflight that records Python, PyTorch, CUDA, driver, GPU, and NVML capability information and blocks production CUDA launch when incompatible.
- [x] 1.4 Add structured production and smoke configuration profiles without mutating module constants at runtime.
- [x] 1.5 Add flags for matched scalar controls, experiment-family selection, MQAR, overwrite recall, cache benchmarks, gate diagnostics, and energy collection.
- [x] 1.6 Add a pytest test layout and reusable temporary-output/config fixtures that never target existing `output/` data.

## 2. Model Variants and Exact Parameter Controls

- [x] 2.1 Refactor model configuration to accept per-layer FFN widths and explicit hybrid versus pure-Gated-DeltaNet attention layouts.
- [x] 2.2 Implement a reusable trainable-parameter component breakdown for embeddings, gates, projections, convolutions, FFNs, norms, and output head.
- [x] 2.3 Implement `scalar-matched` construction by distributing useful FFN-width increments to equal the channel-wise trainable parameter count exactly, rejecting configurations where exact useful matching cannot be constructed.
- [x] 2.4 Add pre-training assertions and manifest fields for parameter equality, gate overhead, control strategy, and per-layer FFN schedule.
- [x] 2.5 Add tests for the current 10M/30M/55M/100M hybrid counts and their documented exact matched-control schedules.
- [x] 2.6 Add tests that matched-control construction also works or fails explicitly for pure-Gated-DeltaNet synthetic and tiny smoke backbones.

## 3. Experiment Manifests and Resume Integrity

- [x] 3.1 Define schema-versioned run/seed manifest and metrics models for TinyStories, MQAR, overwrite recall, benchmarks, energy, and diagnostics.
- [x] 3.2 Implement canonical JSON serialization and SHA-256 fingerprints covering model, data, seeds, budgets, evaluation protocol, environment, expected artifacts, and source revision state.
- [x] 3.3 Implement atomic JSON writes and lifecycle transitions for `pending`, `training`, `trained`, `evaluating`, `complete`, and `failed`.
- [x] 3.4 Save and validate checkpoint SHA-256 checksums, loadability, completed optimizer steps, and processed-token totals.
- [x] 3.5 Replace file-existence resume checks with experiment-family-specific fingerprint and completeness validators.
- [x] 3.6 Implement evaluation-only recovery from a compatible `trained` checkpoint whose mandatory evaluations are incomplete.
- [x] 3.7 Rebuild global trackers, summaries, and plots from validated per-seed artifacts and emit exclusion reasons for stale, smoke, incompatible, corrupt, or partial artifacts.
- [x] 3.8 Add tests for smoke-to-production rejection, configuration drift, corrupt metrics/checkpoints, atomic-write interruption, stale global trackers, and evaluation-only recovery.

## 4. Fixed-Budget Deterministic Training

- [x] 4.1 Seed Python, NumPy, PyTorch CPU, and CUDA generators and persist distinct model/data/evaluation seed fields.
- [x] 4.2 Remove validation-based early termination from production training while retaining periodic validation, diagnostic best-checkpoint saving, and explicit NaN/OOM/divergence failures.
- [x] 4.3 Save the fixed-budget final checkpoint separately and make it the primary evaluation source; label any best-validation result secondary.
- [x] 4.4 Track processed tokens, optimizer/microbatch counts, wall time, optimizer-step duration, training tokens/second, unclipped/clipped gradient norms, and non-finite/loss-spike events.
- [x] 4.5 Persist deterministic TinyStories split/order and fixed natural-language diagnostic-suite identities so all three variants for one seed consume identical inputs.
- [x] 4.6 Replace three-seed significance-star reporting with raw seed tables, mean/standard deviation, and paired variant differences.
- [x] 4.7 Add tests proving equal fixed budgets and paired data order across scalar, channel-wise, and scalar-matched runs.

## 5. Stateful Prefill and Decode

- [x] 5.1 Add one-token cache support to causal depthwise convolutions using the prior `kernel_size - 1` projected values.
- [x] 5.2 Extend the recurrent Gated DeltaNet forward path to accept/return state matrix `S` and incremental convolution caches for scalar and channel-wise gates.
- [x] 5.3 Add full-attention K/V cache support for prefill and one-token causal decode.
- [x] 5.4 Thread typed/validated layer caches through blocks and `GatedDeltaNetLM` without changing uncached training behavior.
- [x] 5.5 Implement explicit prefill and cached greedy-decode APIs that never recompute prior prompt tokens.
- [x] 5.6 Add float32 and mixed-precision equivalence tests comparing full-sequence and cached token-by-token logits for scalar and channel-wise models.
- [x] 5.7 Implement cache/state byte accounting and validate expected recurrent-state and KV-cache growth across context lengths.

## 6. Efficiency, Energy, and Gate Diagnostics

- [x] 6.1 Refactor inference benchmarking into separately named full-prefix generation, prefill, and cached-decode protocols with fixed warmup, repetitions, synchronization, lengths, batch, and dtype metadata.
- [x] 6.2 Add context-length sweeps reporting latency, throughput, time to first token, peak allocated memory, and cache/state bytes.
- [x] 6.3 Implement an NVML energy provider using total-energy counter deltas and complete device/driver/power-limit metadata.
- [x] 6.4 Implement sampled instantaneous-power integration fallback with configurable interval, average/peak watts, and method labeling.
- [x] 6.5 Implement the CPU/unsupported energy provider returning structured `supported: false` results without zero-valued pseudo-measurements.
- [x] 6.6 Instrument training, prefill, and cached-decode regions and calculate joules/run, joules per million training tokens, joules per prefill token, and joules per generated token.
- [x] 6.7 Implement online gate summaries for mean, standard deviation, quantiles, saturation fractions, estimated decay half-life, and channel diversity on a persisted diagnostic corpus.
- [x] 6.8 Add static reports for parameter components, theoretical gate/projection operation overhead, recurrent-state sizes, and cache-size formulas.
- [x] 6.9 Ensure gate hooks are disabled in all timed/energy regions and add tests using mocked NVML counter, power-sampling, and unsupported paths.

## 7. Synthetic MQAR and Overwrite Experiments

- [x] 7.1 Implement versioned deterministic MQAR generation following the Zoology key-value/query formulation with masked query-answer labels.
- [x] 7.2 Implement deterministic overwrite-recall generation where repeated keys target their most recently assigned value and label stale-value predictions explicitly.
- [x] 7.3 Persist generator parameters, hashes, training data identity/order, and held-out suite manifests; pair them across variants for each seed.
- [x] 7.4 Add production MQAR defaults for vocabulary 8,192, 100,000 training examples, 3,000 held-out examples, and configured length/key-count/distance sweeps through length 512.
- [x] 7.5 Add pure-Gated-DeltaNet synthetic model construction using study backbone dimensions, synthetic vocabulary, actual parameter labels, and all enabled gating/control variants.
- [x] 7.6 Implement separately configurable fixed-budget MQAR and overwrite training with masked target loss, independent final checkpoints, lifecycle/resume integration, and exact query-token accuracy.
- [x] 7.7 Implement separate MQAR and overwrite evaluation/reporting; for overwrite include latest-value accuracy, stale-value error rate, unrelated-association retention, overwrite count/distance, memory load, and sequence length.
- [x] 7.8 Replace runtime-random TinyStories needle prompts with a persisted paired suite and label those results exploratory/zero-shot.
- [x] 7.9 Add deterministic generator, masking, no-train/test-overlap, latest-overwrite-target, paired-order, and metric-breakdown tests.

## 8. Orchestration and Research Reporting

- [x] 8.1 Update TinyStories orchestration to schedule two default variants or three variants when the matched-control flag is enabled, preserving per-seed pairing and isolated output paths.
- [x] 8.2 Add separate CLI/entry points for TinyStories, MQAR, overwrite recall, and the complete smoke suite with shared manifests and reporting utilities.
- [x] 8.3 Expand per-seed and aggregate schemas with fixed-budget, parameter, cache, timing, memory, energy, retrieval, gate, stability, and environment metrics.
- [x] 8.4 Generate first-submission comparison tables for TinyStories, MQAR, and separately trained overwrite recall, showing same-backbone and parameter-matched controls without mixing them into one ambiguous effect.
- [x] 8.5 Generate plots for quality/convergence, prefill/decode efficiency, energy, cache/memory scaling, retrieval difficulty, paired seed differences, and gate behavior without significance stars.
- [x] 8.6 Ensure incomplete configurations remain visible with explicit status/reasons and never cause aggregation shape/index errors.

## 9. CPU-Only Smoke Pipeline

- [x] 9.1 Add a local deterministic toy language dataset and tiny three-layer model profile that avoids Hugging Face access and production-scale embeddings/models.
- [x] 9.2 Add tiny MQAR and overwrite profiles with small vocabulary/data, short sequences, one seed, two optimizer steps, and exact parameter-control coverage.
- [x] 9.3 Add smoke-sized prefill/decode/cache-equivalence, gate-diagnostic, reporting, and unsupported-energy checks with minimal repetitions.
- [x] 9.4 Add a smoke recovery scenario that stops after `trained`, resumes evaluation only, and proves the same smoke artifacts cannot resume under production fingerprints.
- [x] 9.5 Create a single end-to-end CPU smoke command writing only under a unique smoke output directory and returning nonzero on any missing required artifact.
- [x] 9.6 Run the full CPU smoke suite, verify no CUDA/NVML/network dependency, confirm completion within the five-minute/2-GB target, and record measured runtime/resource results.

## 10. Documentation and Final Verification

- [x] 10.1 Document Linux Python 3.11 installation, official CUDA wheel URL, mandatory `nvidia-smi` preflight, CUDA 12.x driver requirement, and how to change the wheel channel if the V100 driver is older.
- [x] 10.2 Update the README with experiment-family commands, matched-control interpretation, fixed-budget policy, metric definitions, energy caveats, output schemas, and resume behavior.
- [x] 10.3 Document why MQAR and overwrite recall are separately trained, required first-submission mechanistic experiments and why TinyStories natural-language retrieval remains exploratory.
- [x] 10.4 Run all unit/integration tests and validate every generated JSON artifact against its schema.
- [x] 10.5 Confirm legacy outputs remain unchanged/readable but are rejected for hardened production resumption with actionable reasons.
- [x] 10.6 Leave production training-duration fields explicitly pending V100 pilot measurements rather than selecting them from CPU smoke timing.
