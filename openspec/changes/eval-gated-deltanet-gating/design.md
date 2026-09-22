## Context

Currently, efficient linear-attention networks like Qwen3-Next (Scalar gating) and Kimi Linear (Channel-wise gating) choose different gating granularities, but their downstream effects at small model scales have not been empirically analyzed under identical training conditions. This project implements a swappable Gated DeltaNet attention layer and trains models at four scales (10M, 30M, 55M, 100M) across three seeds (24 total runs) to evaluate perplexity, inference speed, and memory recall.

## Goals / Non-Goals

**Goals:**
*   Implement a PyTorch Gated DeltaNet layer that supports swappable scalar vs. channel-wise decay gating.
*   Ensure full execution compatibility across CPU and CUDA environments.
*   Implement a robust dataset loader for `roneneldan/TinyStories` with automatic retries on download failures.
*   Implement evaluation scripts for perplexity, wall-clock latency/throughput (ms/token and tokens/sec), and needle-in-a-haystack retrieval.
*   Orchestrate all 24 runs sequentially via a single script and aggregate summaries in a global tracker.
*   Provide a fast-executing smoke test (exactly 2 training steps per model) to verify the pipeline.

**Non-Goals:**
*   Implementing custom CUDA/Triton kernels for chunkwise training. (Pure PyTorch recurrent state updates are used for simplicity, stability, and CPU execution support).
*   Fine-tuning hyperparameters per scale or variant. (A single robust common config is used to ensure a controlled comparison).

## Decisions

### 1. Model Configurations and Scale Mapping
To ensure parameter counts scale smoothly, we define the following four models using tied word embeddings (GPT-2 vocabulary of 50,257):
*   **10M Scale**: `d_model = 128`, `n_layer = 6`, `n_head = 4`, `d_ff = 341`
*   **30M Scale**: `d_model = 256`, `n_layer = 8`, `n_head = 8`, `d_ff = 682`
*   **55M Scale**: `d_model = 384`, `n_layer = 12`, `n_head = 8`, `d_ff = 1024`
*   **100M Scale**: `d_model = 512`, `n_layer = 14`, `n_head = 8`, `d_ff = 1364`
*Alternative considered:* Untied embeddings, which would increase model size by an extra 6M-25M parameters per model scale without adding layer depth or capacity. We chose tied embeddings to keep parameters concentrated in model capacity.

### 2. Recurrent Gating Mechanism
The recurrent update rule for the memory state matrix $S_t \in \mathbb{R}^{B \times H \times D \times D}$ (where $D$ is the head dimension) is:
$$S_t = \alpha_t \odot S_{t-1} + \beta_t \cdot (v_t - S_{t-1} k_t) k_t^\intercal$$
*   **Variant A (Scalar):** The projection dimension for $\alpha_t$ is `n_head`. It is reshaped to `(B, H, 1, 1)` and broadcast across the matrix dimensions.
*   **Variant B (Channel-wise):** The projection dimension for $\alpha_t$ is `n_head * d_head`. It is reshaped to `(B, H, D, 1)` and multiplies the rows of $S_{t-1}$ element-wise via PyTorch broadcasting (effectively multiplying on the left by a diagonal matrix $\text{Diag}(\alpha_t)$).
*   **Query/Key Norm:** Queries and keys are normalized using L2 normalization `F.normalize(..., p=2, dim=-1)` prior to recurrence.

### 3. CPU and CUDA Device Agnosticism
All tensors are initialized or moved dynamically using a global `device` variable:
`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
Tensors created inside loops (like the state matrix $S_t$) are placed on the same device as the inputs (`device=x.device`, `dtype=x.dtype`).

### 4. Robust Hugging Face Data Streaming with Exponential Backoff
To avoid connection issues causing the dataset loading to fail, we wrap `load_dataset` in a retry-loop with exponential backoff:
```python
def load_dataset_with_retry(name, split, max_retries=5):
    for i in range(max_retries):
        try:
            return load_dataset(name, split=split, streaming=True)
        except Exception as e:
            if i == max_retries - 1:
                raise e
            time.sleep(2 ** i)
```
*Alternative considered:* Offline file fallbacks (generating synthetic text). We rejected this because it would lead to inaccurate validation perplexity scoring during dry-runs.

### 5. Orchestration Grid and Smoke Test Execution
`run_all.py` runs the cartesian product: `SCALES` (10M, 30M, 55M, 100M) × `VARIANTS` (scalar, channel-wise) × `SEEDS` (3 seeds).
When `SMOKE_TEST = True` is set in `config.py`, the training loop executes for exactly 2 steps per model. This verifies all files, checkpointing, and global summary logging execute correctly without incurring huge training costs.

## Risks / Trade-offs

*   **[Risk] CPU Execution Slowness:** Training 100M parameter models on CPU is extremely slow.
    *   *Mitigation:* The `SMOKE_TEST` mode performs only 2 training steps. The actual full run is meant to be run in a CUDA environment.
*   **[Risk] Out of Memory (OOM) on GPU:** Larger batch sizes at the 100M scale may exceed GPU VRAM on smaller cards.
    *   *Mitigation:* The batch size is customizable in `config.py`, allowing users to reduce it and utilize gradient accumulation if needed.
