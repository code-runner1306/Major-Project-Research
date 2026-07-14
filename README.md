# Gated DeltaNet Gating Granularity Study

This repository contains the complete empirical evaluation pipeline for studying **Gating Granularity (Scalar vs. Channel-wise)** in Gated DeltaNet language models at various parameter scales (10M, 30M, 55M, and 100M). The study compares Qwen3-Next-style scalar gating against Kimi Linear-style channel-wise gating.

---

## 🚀 How It Works

The codebase is structured around a sequential grid-search orchestrator that sweeps over:
1. **4 Parameter Scales**: 10M, 30M, 55M, and 100M parameters.
2. **2 Gating Variants**: Scalar Gating (Variant A) vs. Channel-wise Gating (Variant B).
3. **3 Random Seeds**: 42, 43, and 44 per configuration.

This yields a total of **24 independent runs** executed sequentially. The pipeline handles data downloading (streaming from Hugging Face), model initialization, training, validation, benchmarking, long-context retrieval evaluation, statistical seed aggregation, and automated figure plotting.

---

## 📁 Directory Structure

```
├── config.py              # Central hyperparameters and model architecture configs
├── dataset.py             # Streaming TinyStories dataset loader with offline/retry fallback
├── model.py               # PyTorch Gated DeltaNet and LM block implementations
├── train.py               # Training loop and validation evaluation logic
├── eval.py                # Latency/throughput benchmarking & Needle-in-a-Haystack probe
├── run_all.py             # sequential orchestrator grid runner
├── pyproject.toml         # Dependencies managed by uv
└── output/                # Target outputs folder
    └── run_<UUID>/        # Folder for a single pipeline run (e.g., run_dae0c498)
        ├── run_scalar_10M/
        │   ├── seed42_metrics.json
        │   ├── seed42_config.json
        │   ├── model_checkpoint_seed42.pt
        │   └── aggregate_metrics.json  # Generalized stats over all seeds
        ├── ...
        ├── summary_table.md
        ├── global_tracker.json
        ├── perplexity_scaling.png
        ├── throughput_scaling.png
        └── retrieval_scaling.png
```

---

## 🛠️ Installation & Quickstart

This repository uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

### 1. Install Dependencies
```bash
uv sync
```

### 2. Run the Smoke Test (CPU-friendly)
By default, `SMOKE_TEST = True` in `config.py`. This executes exactly 2 training steps, 2 validation steps, and 2 token generation benchmark runs per model. The entire grid of 24 runs will finish in ~10 minutes on a CPU.
```bash
uv run python run_all.py
```

### 3. Run Production Training (GPU-enabled)
To run full-scale training on the full TinyStories dataset:
1. Open `config.py` and set `SMOKE_TEST = False`.
2. Execute the orchestrator (it will automatically use CUDA if available):
   ```bash
   uv run python run_all.py
   ```

---

## 📊 Core Evaluated Metrics

For every model run, we capture and aggregate:
* **Validation Perplexity**: Evaluates the language modeling prediction performance.
* **Decoding Throughput (tokens/second)**: Measures causal decoding speed by sequentially generating tokens.
* **Decoding Latency (ms/token)**: Measures the average time elapsed to produce a single token.
* **Needle-in-a-Haystack Accuracy**: Evaluates how well the linear attention state retrieves factoids hidden at random depths inside long context sequences.
