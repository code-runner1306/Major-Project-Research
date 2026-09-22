## ADDED Requirements

### Requirement: Stateful Gated DeltaNet and attention decoding
The model SHALL support prefill and one-token decode caches containing each Gated DeltaNet recurrent state and causal-convolution history plus each full-attention layer's K/V history.

#### Scenario: Decoding after prompt prefill
- **WHEN** a prompt is prefetched with caching enabled and a next token is supplied
- **THEN** the model produces the next-token logits and updated caches without recomputing the previous prompt tokens

### Requirement: Cached and full-sequence numerical equivalence
Cached token-by-token inference SHALL match full-sequence inference within documented dtype-specific numerical tolerances for both scalar and channel-wise gating.

#### Scenario: Comparing cached logits
- **WHEN** a deterministic token sequence is evaluated both in one full forward pass and through prefill plus cached decode
- **THEN** corresponding logits agree within the configured tolerance and cache shapes pass validation

### Requirement: Separately named efficiency protocols
The benchmark SHALL separately report full-prefix autoregressive generation, prefill, and cached decode latency/throughput rather than combining them under one decoding metric.

#### Scenario: Completing inference benchmarks
- **WHEN** a trained model is benchmarked
- **THEN** output metrics include protocol-specific time, tokens per second, milliseconds per token where applicable, prompt/generated lengths, batch size, dtype, warmup, repetitions, and synchronization metadata

### Requirement: Context-dependent cache memory reporting
The benchmark SHALL report cache/state bytes and peak allocated memory as context length changes.

#### Scenario: Running a context-length sweep
- **WHEN** prefill/decode is measured at multiple context lengths
- **THEN** the report provides per-length cache size and peak-memory measurements for each gating variant

