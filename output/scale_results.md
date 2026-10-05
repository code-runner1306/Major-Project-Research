---
artifact_type: posthoc_scale_analysis
artifact_schema_version: 1
dataset: tinystories
training_profile: production_1000_steps
reference_tokens: 16384000
measured_scales: [10M, 30M, 55M]
source: output/tinystories/30M/*/*/metrics.json
---

# TinyStories model-size comparison

This report combines the completed 10M, 30M and 55M. Validation results are anchored to the observed runs under the same 1,000-step, 16.384M-token budget.

| Model size | Variant | Parameters | Validation loss | Perplexity | Zero-shot retrieval | Training steps | Training tokens | Tokens / parameter | Training time | Peak training GPU memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10M | Scalar | ≈10M | **2.408101** | **11.003267** | 0/10 | 1,000 | 16.384M | 1.638 | ≈2.917 h | ≈4.00 GiB |
| 10M | Channel-wise | ≈10M | **2.426351** | **11.262935** | 0/10 | 1,000 | 16.384M | 1.638 | ≈2.896 h | ≈4.03 GiB |
| 30M | Scalar | 29,272,320 | **2.359883** | **10.589714** | 0/10 | 1,000 | 16.384M | 0.560 | 5.347 h | ≈7.68 GiB |
| 30M | Channel-wise | 30,071,040 | **2.377349** | **10.776294** | 0/10 | 1,000 | 16.384M | 0.545 | 5.183 h | ≈7.76 GiB |
| 55M | Scalar | ≈55M | **2.352110** | **10.532641** | -- | 1,000 | 16.384M | 0.298 | ≈5.508 h | ≈13.42 GiB |
| 55M | Channel-wise | ≈55M | **2.374898** | **10.718421** | -- | 1,000 | 16.384M | 0.298 | ≈5.482 h | ≈13.54 GiB |

## Research context

The broad direction is consistent with language-model scaling research: loss can improve with model size, but the relationship depends on data and compute, and increasing model size without increasing training tokens can yield diminishing returns. The Chinchilla study finds that model size and training-token count should scale together for compute-optimal training. Here the fixed budget gives the 55M model only about 0.30 training tokens per parameter, versus 1.64 for 10M. These papers support the qualitative caution. TinyStories research also reports coherent story generation from models below 10M parameters, but its GPT-4-based story assessment is not directly comparable to this run’s validation perplexity.

References:

- Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361).
- Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).
- Eldan and Li, [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759).
- Yang et al., [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464).
