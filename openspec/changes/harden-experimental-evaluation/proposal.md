## Why

The current pipeline can execute the scalar-versus-channel-wise grid, but it cannot yet support a defensible paper claim: the variants have unequal useful parameter counts, training budgets can diverge through early stopping, retrieval prompts are not controlled, decoding does not use recurrent caches, energy is not measured, and stale smoke-test artifacts can be mistaken for completed production runs. This change hardens the study before expensive V100 training and adds CPU-safe smoke coverage for every new experiment path.

## What Changes

- Preserve the primary same-backbone scalar versus channel-wise comparison and add an opt-in, exactly parameter-matched scalar control whose FFN capacity absorbs the channel-wise gate parameter overhead.
- Replace early-stopped headline training with fixed optimizer-step and token budgets; retain periodic validation and emergency failure handling.
- Split inference measurement into full-prefix generation, prefill, and genuinely stateful decode using Gated DeltaNet recurrent/convolution caches and attention KV caches.
- Measure training and inference wall time, throughput, peak memory, GPU energy, average/peak power, and normalized energy efficiency, with graceful unsupported results on CPU-only systems.
- Capture gate distributions, saturation, channel diversity, estimated memory half-life, gradient norms, loss spikes, and theoretical parameter/operation overhead so any quality difference can be interpreted mechanistically.
- Replace runtime-random retrieval prompts with versioned, persisted evaluation suites and add two separately trained synthetic experiments required for the first paper submission: standard MQAR and overwrite recall, each with held-out evaluation and context-length/key-count/distance breakdowns.
- Make all Python, NumPy, and PyTorch randomness deterministic and pair datasets/evaluation cases across compared variants.
- Strengthen resumption with experiment fingerprints, explicit lifecycle states, atomic artifacts, completeness validation, checkpoint verification, and partial evaluation recovery.
- Add a Linux/Python 3.11 `requirements.txt` with an official PyTorch CUDA wheel index, documented driver preflight, and dependency pins compatible with the college V100 environment.
- Add a CPU-only smoke profile using tiny models, tiny synthetic datasets, short contexts, and minimal repetitions so the entire orchestration path can be verified on the development laptop without CUDA or large checkpoints.
- Report descriptive multi-seed results and paired seed differences without unsupported significance claims from only three seeds.

## Capabilities

### New Capabilities

- `parameter-matched-controls`: Configure and verify same-backbone and exactly parameter-matched scalar controls at every study scale.
- `fixed-budget-research-training`: Enforce equal optimizer-step/token budgets, deterministic paired runs, and paper-safe descriptive reporting.
- `stateful-efficiency-benchmarking`: Measure full-prefix, prefill, cached decode, memory, latency, throughput, and cache scaling under a reproducible protocol.
- `energy-efficiency-measurement`: Capture GPU energy and power for training and inference with normalized metrics and CPU-safe unsupported behavior.
- `gating-behavior-diagnostics`: Measure how scalar and channel-wise decay gates behave and relate gate dynamics to training stability and theoretical overhead.
- `synthetic-retrieval-experiments`: Generate, separately train, evaluate, and produce first-submission paper artifacts for deterministic MQAR and overwrite-recall experiments independently of TinyStories language modeling.
- `experiment-resumption-integrity`: Resume only compatible, complete work using fingerprints, lifecycle states, checkpoint validation, and atomic artifacts.
- `reproducible-runtime-and-smoke-testing`: Provide a Python 3.11 Linux CUDA requirements file and an end-to-end CPU smoke profile for all experiment paths.

### Modified Capabilities

None. The repository has no archived baseline capability specifications; this change defines the hardened behavior as new capabilities while superseding the corresponding experimental assumptions in active change artifacts.

## Impact

- **Core files:** `config.py`, `model.py`, `dataset.py`, `train.py`, `eval.py`, and `run_all.py`.
- **New experiment modules/artifacts:** synthetic retrieval generation/training/evaluation, cache-aware generation state, energy collection, run manifests, lifecycle status files, and expanded reports/plots.
- **Environment:** new `requirements.txt` targeting Linux x86-64, Python 3.11, and an official CUDA-enabled PyTorch wheel channel; CPU execution remains supported.
- **Output compatibility:** existing metrics remain readable for inspection, but artifacts lacking the new fingerprint/schema/completeness fields will not be eligible for production resumption.
- **Execution count:** enabling the matched scalar control changes the TinyStories grid from 24 to 36 runs; MQAR and overwrite recall are two additional required experiment families with their own smoke and production budgets.
