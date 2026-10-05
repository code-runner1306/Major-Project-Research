# Gated DeltaNet gating granularity study

This repository studies scalar head-wise versus channel-wise decay gates in Gated DeltaNet language models. The paper reports two deliberately separate comparisons:

- Same-backbone: scalar and channel-wise models have the same backbone dimensions and baseline FFN widths. Channel-wise models retain their natural gate-parameter overhead.
- Capacity control: an optional `scalar-matched` model keeps scalar gating but widens its SwiGLU FFNs until its trainable parameter count exactly equals the channel-wise counterpart. This is a capacity control, not a pure gating ablation.

The first submission also includes two separately trained synthetic, pure-Gated-DeltaNet experiments: MQAR and overwrite recall. TinyStories needle prompts are persisted and reported only as exploratory zero-shot diagnostics.

## Installation on the college Linux V100 host

The intended production runtime is Linux x86-64 with Python 3.11. The repository pins that version in `.python-version` and `pyproject.toml`.

Before installing, record the V100 driver with:

```bash
nvidia-smi
```

`requirements.txt` uses the official PyTorch CUDA 12.8 wheel index and pins `torch==2.11.0+cu128`:

```bash
python3.11 -m pip install -r requirements.txt
```

No local CUDA toolkit is required for that wheel. CUDA 12.x requires an NVIDIA driver at least `525.60.13`; the production launcher performs the same fail-closed preflight and records Python, PyTorch, CUDA runtime, driver, GPU, NVML, energy-counter, power-sampling, and power-limit information. If the driver is older or the runtime provides a different CUDA/PyTorch channel, do not silently fall back to CPU: select an appropriate official PyTorch wheel channel, update `requirements.txt`, and record the decision before starting the grid.

For local CPU development, install a CPU PyTorch build independently if needed. Do not install the CUDA requirements file on a machine without a suitable NVIDIA driver.

## Commands

Run the complete offline CPU smoke suite:

```bash
python run_smoke.py
```

It writes to a unique `output/smoke_<UTC timestamp>/` directory, uses a three-layer tiny model and local toy language data, executes exactly two optimizer steps per seed, and does not access Hugging Face, CUDA, or NVML. It exercises scalar, channel-wise, and exact scalar-matched controls; TinyStories-compatible toy orchestration; cached prefill/decode; unsupported-energy output; MQAR; overwrite recall; reporting; and checkpoint-to-evaluation recovery. The acceptance target is under five minutes and 2 GB RAM; its measured runtime is written to `smoke_result.json`.

Run only one smoke experiment family:

```bash
python run_mqar.py --smoke
python run_overwrite.py --smoke
```

The production entry point is intentionally blocked until a V100 pilot records a fixed training duration. First run the reference pilot on the college V100S; it does real optimizer work but does not unlock or mutate the fixed training budget:

```bash
python run_all.py --pilot --recurrence-backend reference --pilot-scales 55M,100M --pilot-warmup 25 --pilot-steps 200 --pilot-trace --pilot-data-fed
```

It creates a unique `output/pilots/pilot_<timestamp>_<fingerprint>_<nonce>/` directory containing JSON, Markdown, and (when requested) one Chrome trace. It records GPU-event and wall-clock step distributions, host/launch gap, throughput, allocated/reserved memory, and NVML energy. The resident-tensor measurement isolates model/recurrence work; `--pilot-data-fed` adds a paired window with TinyStories fetch, tokenization, host-to-device transfer, and the same optimizer step after loader setup. Start with the reference backend: this tells us whether recurrence is actually the dominant cost before building a CUDA kernel.

Production commands must state their backend explicitly, preventing silent backend changes:

```bash
python run_all.py --mode production --recurrence-backend reference --matched-control
python run_all.py --mode production --recurrence-backend reference --family mqar
```

Limit a run to selected model scales and seeds with comma-separated overrides. Production profiles also need a fixed optimizer-step budget. For example, this runs both gating variants at 10M for seed 42 for 1,000 steps:

```bash
python run_all.py --mode production --recurrence-backend reference --family tinystories --models 10M --seeds 42 --max-optimizer-steps 1000
```

`cuda-fused` is reserved for the repository-owned `sm_70` extension. It will require a matching V100 qualification artifact, local CUDA toolkit (`nvcc` and headers), and C++ compiler; production preflight records and checks these conditions before data loading. It never falls back to `reference` silently. The reference backend is the portable rollback path and is allowed for explicitly recorded diagnostics. After the V100 pilot, set a recorded `max_optimizer_steps` in the production profile for every family. CPU smoke timing must never be used to select this budget.

## Controlled model counts

| Backbone | Scalar | Channel-wise | Exact scalar-matched FFN widths |
| --- | ---: | ---: | --- |
| 10M | 10,030,880 | 10,155,680 | 470, 470, 469, 469, 469, 469 |
| 30M | 29,272,320 | 30,071,040 | 937, 937, 936, 936, 936, 936, 936, 936, 936, 936 |
| 55M | 53,348,288 | 55,122,368 | 1304 in all 12 layers |
| 100M | 96,918,912 | 100,817,280 | 1677 in all 16 layers |

The manifest records each model’s total, parameter-component breakdown, natural gate overhead, control strategy, and per-layer FFN schedule. A configuration that cannot absorb the gate overhead through an integral useful FFN expansion fails explicitly; it never receives dummy parameters.

## Training policy and metrics

Headline quality uses the fixed-budget final checkpoint, not a validation-selected early checkpoint. Validation still runs periodically; the best validation checkpoint is preserved only as a labeled secondary diagnostic. Failures such as NaN loss/gradients, divergence events, and OOM remain failures.

For each run, the pipeline stores fixed steps, processed tokens, optimizer/microbatch counts, losses, wall time, step durations, training tokens/sec, unclipped/clipped gradient norms, stability events, and primary final validation loss/perplexity.

Inference reports three separately named protocols rather than conflating them:

- `full_prefix_generation`: optional legacy generation that recomputes the prefix; it is disabled by default because it is much slower;
- `prefill`: one prompt pass and time-to-first-token;
- `cached_decode`: one-token stateful decoding using recurrent/convolution caches and full-attention K/V caches.

Enable the slow full-prefix benchmark explicitly with `--full-prefix-benchmark`. Routine runs still perform prefill and cached-decode benchmarks using the configured warmup and repetition counts.

Each context length records latency, throughput, peak memory, cache/state bytes, warmup/repetition/batch/dtype metadata, and cache-equivalence coverage. Energy uses NVML total-energy deltas where available, otherwise sampled board-power integration. It reports joules/run, joules per million training tokens, joules per prefill token, and joules per generated token. CPU/unsupported systems emit `supported: false` with a reason rather than fake zero joules.

Gate diagnostics run outside timed and energy regions on a fixed corpus. They include per-layer/head means, standard deviations, quantiles, saturation fractions, half-life summaries, and channel diversity (scalar is `null`, not invented). Static reports include parameter components, gate/projection overhead, recurrent-state size, and cache-size formulas.

## MQAR and overwrite recall

MQAR uses versioned deterministic symbolic key-value/query examples with a vocabulary of 8,192, 100,000 training examples, 3,000 held-out examples, masked query-answer loss, exact-token accuracy, and paired example order. Production sweeps sequence length, key count, and query distance through length 512.

Overwrite recall is independently trained and evaluated, not a subscore of MQAR. A key is assigned an earlier value and later overwritten; a query must return the latest value. First-submission outputs include latest-value accuracy, stale-value error rate, unrelated-association retention, and breakdowns by overwrite count, overwrite-to-query distance, memory load, and sequence length.

Synthetic models use pure Gated DeltaNet layers to avoid full-attention shortcuts. Their results are labelled with their backbone scale and actual synthetic-vocabulary parameter count, not as the original TinyStories total. Natural-language needle results do not establish retrieval quality; they remain exploratory zero-shot diagnostics with a persisted paired prompt-suite hash.

## Artifacts, resumption, and reporting

Every seed has an atomically written, schema-versioned manifest and lifecycle:

```text
pending -> training -> trained -> evaluating -> complete
                         \-> failed
```

The canonical SHA-256 fingerprint covers run mode, family, model/control schedule, seeds, fixed budget, data/generator identity, evaluation protocol, expected artifacts, runtime environment, and source revision state. Completion requires matching fingerprints, a loadable checksummed final checkpoint, the fixed step/token budget, and all mandatory finite metrics. A compatible `trained` run may resume evaluation only; smoke, legacy, corrupt, partial, or configuration-drift artifacts are visible in exclusion reports but never accepted as production completions.

`aggregate_report.json` and `first_submission_summary.md` are rebuilt from validated per-seed artifacts. They show raw seed values, mean/standard deviation, and paired variant differences. With three seeds the repository deliberately does not calculate p-values or print significance stars.

## Verification

```bash
python -m pytest
python run_smoke.py
openspec validate harden-experimental-evaluation
```

The recurrence change additionally has a V100-only pilot and CUDA qualification path. CPU runs deliberately exercise only `reference`; a request for `cuda-fused` on a CPU machine fails with an actionable reason rather than pretending CUDA work was tested.

The tests use temporary directories and never target existing `output/` data. Legacy output folders are left untouched and remain readable as historical files, but lack the hardened manifest/fingerprint schema and are rejected with an actionable resumption reason.
