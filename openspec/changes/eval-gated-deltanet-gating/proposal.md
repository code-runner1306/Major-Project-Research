## Why

Currently, there is a design disagreement between leading efficient-attention architectures on gating granularity within Gated DeltaNet: Qwen3-Next uses scalar head-wise gating, while Kimi Linear and Gated DeltaNet-2 utilize channel-wise gating. However, no published work has isolated this architectural decision as a single variable in a controlled experiment across scale, particularly at smaller parameter scales (10M–100M) suitable for resource-constrained deployments. 

This change introduces a controlled evaluation pipeline to train and benchmark matched pairs of models across four scales (10M, 30M, 55M, 100M) with three random seeds per configuration (24 runs total) on the TinyStories dataset, resolving whether channel-wise gating's accuracy benefits generalize below scale and if they justify their additional wall-clock latency overhead.

## What Changes

*   **Tokenizer and Data Ingestion Pipeline**: Ingestion of the Hugging Face `roneneldan/TinyStories` dataset with a retry-on-failure loop (to ensure dataset accuracy and avoid fallbacks) and tokenization using standard GPT-2 Tiktoken encoding.
*   **Swappable Gated DeltaNet Layer**: A PyTorch-based token-mixer layer that supports both Variant A (scalar gating) and Variant B (channel-wise gating) with L2 normalization on queries and keys, causal 1D depthwise convolutions, and a stateful recurrent loop compatible with both CPU and CUDA environments.
*   **Hybrid Language Model Architecture**: Alternates 3 Gated DeltaNet blocks with 1 full-attention layer (3:1 hybrid ratio), scaling parameter sizes across 10M, 30M, 55M, and 100M parameter points.
*   **Training Loop and Checkpointing**: Multi-seed training with validation perplexity tracking, saving metrics and config to subdirectories.
*   **Evaluation Engine**: Benchmarking scripts to measure wall-clock latency (ms/token) and decode throughput (tokens/second), plus a synthetic long-context needle-in-a-haystack retrieval probe.
*   **Grid Orchestration Script**: Runs all 24 configurations sequentially and updates a master global tracker with summary stats, including a custom smoke-test execution pathway that runs 2 training steps/epochs to verify correct end-to-end integration.

## Capabilities

### New Capabilities
- `eval-gated-deltanet-gating`: The system mixes Gated DeltaNet models (scalar vs. channel-wise) across 4 parameter scales (10M, 30M, 55M, 100M) and evaluates them on language modeling validation perplexity, wall-clock decoding throughput, and synthetic needle-in-a-haystack retrieval accuracy, managing the grid execution, logging, and aggregation.

### Modified Capabilities
<!-- None -->

## Impact

*   **Dependencies**: Requires PyTorch (CPU and CUDA compatible), `transformers`, `datasets`, `tiktoken`, `tqdm`, `pandas`, and `matplotlib`.
*   **Storage**: Creates an `output/` directory containing subfolders for each of the 24 runs and a global tracker summary file (`global_tracker.json`).
