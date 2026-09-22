## Context

The repository studies scalar head-wise versus channel-wise decay in small Gated DeltaNet language models. The existing TinyStories pipeline uses four backbone scales, two gating variants, and three paired seeds, but several implementation choices weaken the intended controlled comparison: channel-wise gates add a growing number of trainable parameters, early stopping permits unequal token budgets, autoregressive benchmarking recomputes the full prefix, retrieval cases are generated from unseeded Python randomness, and resumption accepts incomplete or stale metric files.

The college NVIDIA V100 32 GB is the target device for reported production measurements. An RTX 3050 8 GB may be used for development, but cross-device performance and energy measurements are not comparable. Development smoke tests must run on a CPU-only laptop without downloading TinyStories or instantiating the 10M-100M production grid. The Linux deployment environment will use Python 3.11. The V100 driver version is not yet known.

## Goals / Non-Goals

**Goals:**

- Preserve the scientifically useful same-backbone comparison while exposing its natural parameter overhead.
- Add an opt-in scalar capacity control with exactly the same total number of trainable parameters as the channel-wise model.
- Guarantee equal training-token/optimizer-step budgets within every paired comparison.
- Measure both the current full-prefix generation behavior and true cache-aware prefill/decode behavior.
- Add reproducible GPU energy and power measurements without making CUDA mandatory for smoke tests.
- Add canonical MQAR and selective overwrite-recall training and held-out evaluation as two required experiment families for the first paper submission.
- Make resume decisions depend on experiment identity and artifact completeness rather than file existence.
- Verify every path quickly on CPU with tiny local data and models.
- Produce descriptive, paired multi-seed reports suitable for a paper using three seeds.

**Non-Goals:**

- Choosing the final production training duration or promising a complete-grid wall-clock time before V100 pilots.
- Implementing custom CUDA/Triton chunkwise training kernels.
- Claiming carbon emissions as a hardware-independent measurement; joules/kWh are primary and carbon estimates are optional.
- Treating three seeds as sufficient for confirmatory null-hypothesis significance testing.
- Resuming from an arbitrary dataloader microbatch inside a partially completed optimizer step.
- Combining V100 and RTX 3050 timing/energy measurements in one headline comparison.

## Decisions

### 1. Keep two scientific comparisons and gate the capacity control

The default comparison remains `scalar` versus `channel-wise` with identical `d_model`, layer count, head count, attention layout, and baseline FFN widths. This measures the real architectural substitution, including its natural parameter overhead.

`ENABLE_PARAMETER_MATCHED_SCALAR_CONTROL` adds a third `scalar-matched` configuration. Its scalar gate is unchanged; useful FFN capacity is expanded until its total trainable parameter count exactly equals the paired channel-wise model. No unused/dummy parameters are allowed. The implementation computes the channel-wise gate overhead and distributes the required integer FFN-width increments across layers. It asserts exact equality before training and records the per-layer FFN schedule.

For the current hybrid backbones, the control schedules are:

| Scale | Scalar params | Channel-wise params | Matched scalar FFN schedule |
|---|---:|---:|---|
| 10M | 10,030,880 | 10,155,680 | four layers at 469 and two at 470 |
| 30M | 29,272,320 | 30,071,040 | eight layers at 936 and two at 937 |
| 55M | 53,348,288 | 55,122,368 | all layers at 1304 |
| 100M | 96,918,912 | 100,817,280 | all layers at 1677 |

Alternative considered: adding unused scalar parameters. Rejected because it equalizes a reported count without equalizing useful capacity. Alternative considered: shrinking channel-wise FFNs. Rejected as the only control because it weakens the experimental treatment; widening scalar provides a stronger capacity baseline.

### 2. Use fixed-budget final checkpoints for headline quality

Production runs execute a predetermined number of optimizer steps and tokens. Early stopping is removed from the headline path. Periodic validation and best-checkpoint saving remain diagnostic, but the fixed final checkpoint supplies primary perplexity and downstream results. NaN, divergence, and OOM remain explicit failures and are never counted as completed seeds.

Python `random`, NumPy, PyTorch CPU, and all CUDA generators are seeded. Data-generation seeds, model-initialization seeds, and evaluation-suite versions are recorded. Within a seed, all variants consume the same ordered examples. Reports show all seed values, mean and standard deviation, plus paired channel-wise-minus-scalar differences. Inferential significance stars are removed for the three-seed study.

Alternative considered: retaining early stopping equally for both variants. Rejected because independent convergence decisions still produce unequal token budgets and checkpoint-selection opportunities.

### 3. Introduce explicit cache state for prefill and decode

Model forward APIs accept an optional cache and return an updated cache when requested. Each Gated DeltaNet layer cache contains:

- recurrent state matrix `S` per batch and head;
- the last `kernel_size - 1` pre-convolution projected Q, K, and V values needed for one-token causal convolution;
- any shape/device/dtype metadata required for validation.

Each full-attention layer cache contains accumulated K and V tensors. Prefill processes a prompt once and returns all layer caches. Decode consumes one token and updates caches without recomputing the prefix. A cache-equivalence test compares full-sequence logits with token-by-token cached logits within dtype-appropriate tolerances.

The benchmark reports three separately named protocols: full-prefix autoregressive generation (legacy behavior), prefill, and cached decode. It fixes hardware, dtype, batch size, prompt length, generated length, warmup, repetitions, and synchronization behavior in the run manifest.

Alternative considered: replacing the legacy benchmark. Rejected because keeping it provides continuity with existing artifacts, as long as it is labeled accurately.

### 4. Measure GPU energy through a provider abstraction

An energy provider exposes start/stop regions and structured support metadata. On supported NVIDIA hardware it first uses NVML total-energy counter deltas. If the counter is unavailable, it samples instantaneous board power at a configured interval and integrates power over time. It records total joules, duration, average/peak watts, GPU utilization where available, and the measurement method.

Training reports joules/run and joules per million training tokens. Prefill reports joules per processed token. Decode reports joules per generated token. Measurements run sequentially on the designated device, after warmup, and record GPU name, UUID, driver, CUDA runtime, power limit, and dtype. On CPU or unsupported hardware the provider returns `supported: false` with a reason and never fails the experiment.

`nvidia-ml-py` is the direct production dependency. CodeCarbon may be used later for optional emissions estimates but is not the primary measurement path because its grid assumptions and sampling interval are not appropriate for short decode regions.

### 5. Make MQAR and overwrite recall a separate synthetic experiment family

The retrieval harness is independent of TinyStories and is exposed through a separate entry point/configuration. It uses a deterministic generator modeled on the official Zoology Multi-Query Associative Recall task:

- symbolic vocabulary size 8,192 in production;
- 100,000 generated training examples and one fixed 3,000-example held-out suite per configured protocol;
- masked loss and exact-token accuracy at query-answer positions;
- paired data order across variants;
- breakdowns by sequence length, number of key-value pairs, and query distance/depth;
- in-distribution evaluation and configured length extrapolation through length 512.

MQAR models are trained from scratch rather than fine-tuned from TinyStories. They reuse the study backbone dimensions and gating implementations but use the synthetic vocabulary and a pure-Gated-DeltaNet attention layout so a full-attention layer cannot bypass the recurrent memory being studied. Output labels call these `10M-backbone`, `30M-backbone`, and so on rather than claiming that the smaller-vocabulary models retain the original total parameter counts.

Overwrite recall is a distinct, required synthetic experiment in the same harness: a key is assigned a value, later overwritten, and queries target the latest value. It is trained separately from standard MQAR and has its own deterministic train/test manifests, checkpoints, metrics, aggregate tables, and plots. Its first-submission metrics include latest-value accuracy, stale-value error rate, unrelated-association retention, and breakdowns by overwrite count and distance. Natural-language needle-in-a-haystack remains an exploratory evaluation of TinyStories checkpoints and uses a persisted, paired prompt suite rather than runtime-random cases.

Alternative considered: evaluating unmodified TinyStories checkpoints on ad hoc question-answer prompts. Rejected as the primary retrieval result because failure may reflect prompt-format unfamiliarity instead of memory quality.

### 6. Fingerprint experiments and model resumption as a state machine

Every run has a schema-versioned canonical manifest. Its SHA-256 fingerprint includes the experiment family, scale/backbone, variant, per-layer configuration, all training and evaluation hyperparameters, seeds, dataset name/revision/fingerprint or synthetic-generator version, tokenizer/version, expected artifacts, package/runtime versions, and source revision/dirty state.

Each seed has an atomically written lifecycle document:

`pending -> training -> trained -> evaluating -> complete`, with `failed` carrying a structured reason.

A seed is complete only when the fingerprint matches, the expected fixed step/token budget was reached, the checkpoint exists and can be loaded (with checksum), and all mandatory finite metrics for that experiment family exist. A `trained` seed with missing evaluation results resumes from its checkpoint at evaluation rather than retraining. Incompatible legacy/stale artifacts are preserved for inspection but are never resumed as production work. Global trackers and reports are rebuilt from validated per-seed artifacts.

Alternative considered: checking only for validation keys in a metrics JSON. Rejected because a crash between training and benchmarking currently creates a false cache hit.

### 7. Separate production configuration from CPU smoke configuration

CPU smoke mode substitutes a local deterministic toy language dataset, one seed, a tiny three-layer model, very short contexts, two optimizer steps, one validation batch, and minimal benchmark repetitions. It exercises scalar, channel-wise, and matched-scalar construction; TinyStories orchestration; MQAR; overwrite recall; cache equivalence; partial evaluation resumption; reporting; and the unsupported-energy path. It must not instantiate production-scale models, download Hugging Face data, require NVML, or create large checkpoints.

Smoke outputs live under a clearly marked smoke directory and their fingerprints include `run_mode: smoke`, preventing production resumption. A target of five minutes and 2 GB RAM on a CPU-only laptop is used as the acceptance budget.

### 8. Target Linux Python 3.11 with a driver-gated CUDA requirement

`requirements.txt` targets Linux x86-64 and Python 3.11. It pins the CUDA PyTorch build through the official `https://download.pytorch.org/whl/cu126` index and pins the research/runtime dependencies, including `nvidia-ml-py` and `pytest`. Installation documentation requires checking `nvidia-smi` first. CUDA 12.x wheels require a sufficiently recent NVIDIA driver; if the college driver is older than the documented minimum, the PyTorch wheel channel/version must be changed before production installation rather than bypassing the check.

`pyproject.toml` is aligned to Python 3.11-compatible constraints so `uv` and `pip` describe the same environment. The CPU development environment remains able to install a CPU PyTorch build independently of the Linux GPU requirements file.

### 9. Collect mechanistic gate and training-stability diagnostics

Evaluation hooks collect decay-gate statistics on one fixed, persisted diagnostic corpus without retaining per-token tensors after aggregation. For every Gated DeltaNet layer/head they report mean, standard deviation, configured quantiles, fractions near zero and one, and the distribution of estimated decay half-life. Channel-wise models additionally report within-head channel variance/diversity. Comparisons aggregate these statistics by layer and scale while preserving raw summaries needed for analysis.

Training records unclipped and clipped gradient norms, non-finite/loss-spike events, optimizer-step duration, processed tokens, and training tokens/second. A static model-cost report records parameter components, recurrent-state size, cache size formulas, and theoretical projection/gate operation overhead. These diagnostics explain possible quality/efficiency differences but are not promoted to independent hypothesis tests.

Alternative considered: saving every gate activation. Rejected because it would create prohibitively large artifacts and unnecessary privacy/storage risk; online summary accumulation is sufficient.

## Risks / Trade-offs

- **[Risk] The matched scalar control changes FFN capacity and is not a pure gating ablation.** -> Present it explicitly as a capacity control alongside, not instead of, the same-backbone comparison.
- **[Risk] Cached full-attention KV state may dominate MQAR in hybrid models.** -> Use pure Gated DeltaNet blocks for the synthetic mechanistic experiment and retain hybrids for TinyStories.
- **[Risk] The Python recurrent loop may make the 36-run production grid infeasible.** -> Defer the final training budget until production-shaped V100 timing pilots; do not weaken smoke correctness to predict runtime.
- **[Risk] NVML energy readings include whole-board activity and can be contaminated by other processes.** -> Run sequentially on an otherwise idle GPU, record the method/device metadata, warm up, and repeat inference regions long enough for stable measurement.
- **[Risk] The unknown college driver may not support the default CUDA 12.6 wheel.** -> Make driver preflight mandatory and keep the CUDA wheel selection isolated/documented for adjustment.
- **[Risk] Synthetic recall may not predict natural-language quality.** -> Treat it as a mechanistic complement to TinyStories perplexity, not a replacement.
- **[Risk] Old results become non-resumable under the stricter schema.** -> Preserve them read-only and provide explicit incompatibility reasons; production rigor takes precedence over legacy cache reuse.
- **[Risk] Three seeds provide weak inferential power.** -> Limit claims to descriptive paired robustness and expose raw seed results.
- **[Risk] Diagnostic collection can perturb performance measurements.** -> Run gate hooks outside timed/energy benchmark regions and report their overhead separately.

## Migration Plan

1. Add configuration schemas, variant construction, and manifest/fingerprint infrastructure while retaining legacy result readers.
2. Add cache-aware model APIs and equivalence tests before using cached performance metrics.
3. Add energy collection and deterministic evaluation-suite persistence.
4. Add the MQAR/overwrite experiment family and reports.
5. Add Linux requirements and reconcile Python constraints.
6. Run unit tests and the complete CPU-only smoke profile in a new smoke output directory.
7. Run short production-shaped V100 preflight/timing/energy pilots only after `nvidia-smi` and dependency checks pass.
8. Keep prior output directories unchanged; rollback consists of disabling the new flags and using the previous entry points, not rewriting legacy artifacts.

## Open Questions

- What driver version is installed on the college V100, and therefore whether the default CUDA 12.6 PyTorch wheel is deployable without changing the wheel channel?
- What fixed TinyStories and synthetic-retrieval training budgets are feasible after production-shaped V100 timing pilots?
