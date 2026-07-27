import os

# Smoke test config flag
SMOKE_TEST = False  # Set to False for actual full training

# Grid search parameters
SCALES = ["10M", "30M", "55M", "100M"]  # 4 scales
VARIANTS = ["scalar", "channel-wise"]  # 2 variants
SEEDS = [42, 43, 44]  # 3 seeds -> 4 scales * 2 variants * 3 seeds = 24 runs

# Common Hyperparameters (Full Run vs. Smoke Test)
if SMOKE_TEST:
    BATCH_SIZE = 2
    GRAD_ACCUM_STEPS = 1
    MAX_STEPS = 4        # Set to 4 steps in smoke test to check periodic eval logic
    VAL_STEPS = 2
    CONTEXT_LENGTH = 64
    DATASET_SUBSET_SIZE = 100
    NUM_EPOCHS = 1
    BENCHMARK_WARMUP = 2
    BENCHMARK_REPEATS = 5
    NEEDLE_LENGTH = 64
    EVAL_INTERVAL = 2    # Run validation checks every 2 steps in smoke test
    PATIENCE = 2         # Early stopping patience
else:
    BATCH_SIZE = 8       # Micro-batch size to prevent CUDA OOM (0.82 GB peak logits vs 3.29 GB)
    GRAD_ACCUM_STEPS = 4 # Effective batch size = 8 * 4 = 32
    MAX_STEPS = 15000    # Optimal steps for convergence on TinyStories
    VAL_STEPS = 200
    CONTEXT_LENGTH = 512
    DATASET_SUBSET_SIZE = None  # Use full dataset
    NUM_EPOCHS = 1
    BENCHMARK_WARMUP = 10
    BENCHMARK_REPEATS = 50
    NEEDLE_LENGTH = 512
    EVAL_INTERVAL = 1000 # Run validation checks every 1000 steps
    PATIENCE = 3         # Stop if validation perplexity does not improve for 3 checks

# Model architecture scaling mappings
# Vocab size is 50257 (GPT-2 tokenizer size). Embeddings are tied.
SCALE_CONFIGS = {
    "10M": {
        "d_model": 160,
        "n_layer": 6,
        "n_head": 4,
        "d_ff": 426,  # SwiGLU expansion 8/3
    },
    "30M": {
        "d_model": 320,
        "n_layer": 10,
        "n_head": 8,
        "d_ff": 853,
    },
    "55M": {
        "d_model": 448,
        "n_layer": 12,
        "n_head": 8,
        "d_ff": 1194,
    },
    "100M": {
        "d_model": 576,
        "n_layer": 16,
        "n_head": 12,
        "d_ff": 1536,
    }
}

# General settings
VOCAB_SIZE = 50257
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
GRAD_CLIP = 1.0

# Parallel execution settings
PARALLEL_TRAININGS = 2  # Number of model configurations to train in parallel (Default: 1 for sequential)

# Paths & Resume Options
OUTPUT_DIR = "output"
GLOBAL_TRACKER_FILE = os.path.join(OUTPUT_DIR, "global_tracker.json")
DATA_DIR = "data"
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
RESUME_RUN_ID = os.environ.get("RESUME_RUN_ID", None)  # Set to run ID string (e.g. "run_dae0c498") to resume or target specific directory

