## 1. Setup and Environment

- [x] 1.1 Install dependencies via `uv add` (`torch`, `transformers`, `datasets`, `tiktoken`, `tqdm`, `pandas`, `matplotlib`)
- [x] 1.2 Verify that PyTorch is correctly installed and checks for CUDA capabilities

## 2. Configuration and Data Loading

- [x] 2.1 Create `config.py` with common hyperparameters, scales mapping (10M, 30M, 55M, 100M), grid options, and output paths
- [x] 2.2 Create `dataset.py` with BPE tokenization using `tiktoken` and streaming dataset load for `roneneldan/TinyStories`
- [x] 2.3 Implement the retry-on-failure loop inside the dataset loading module

## 3. Model Architecture

- [x] 3.1 Implement L2 normalization helper and causal 1D depthwise convolution helper
- [x] 3.2 Implement recurrent Gated DeltaNet attention update loop supporting both scalar and channel-wise gating
- [x] 3.3 Implement `GatedDeltaNetAttention` module with swappable gating modes and causal convolutions on Q, K, V
- [x] 3.4 Implement standard self-attention and `GatedDeltaNetBlock` alternating 3:1 ratio
- [x] 3.5 Implement `GatedDeltaNetLM` combining embeddings, blocks, RMSNorm, SwiGLU, and tied language model head

## 4. Training and Evaluation Engines

- [x] 4.1 Implement training loop in `train.py` with checkpointing, logging, and CPU/CUDA agnosticism
- [x] 4.2 Implement inference latency/throughput benchmarking in `eval.py`
- [x] 4.3 Implement needle-in-a-haystack retrieval evaluation probe in `eval.py`

## 5. Grid Orchestrator and Verification

- [x] 5.1 Implement `run_all.py` sequential orchestrator running all 24 configurations
- [x] 5.2 Implement global tracker metric logging and summary table generation in `output/global_tracker.json`
- [ ] 5.3 Implement the fast smoke test path (exactly 2 training steps) and verify the end-to-end execution on CPU
