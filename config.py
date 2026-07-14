import os

# Smoke test config flag
SMOKE_TEST = True  # Set to False for actual full training

# Grid search parameters
SCALES = ["10M", "30M", "55M", "100M"]  # 4 scales
VARIANTS = ["scalar", "channel-wise"]  # 2 variants
SEEDS = [42, 43, 44]  # 3 seeds -> 4 scales * 2 variants * 3 seeds = 24 runs

# Common Hyperparameters (Full Run vs. Smoke Test)
if SMOKE_TEST:
    BATCH_SIZE = 2
    MAX_STEPS = 2        # Exactly 2 training steps in smoke test per model
    VAL_STEPS = 2
    CONTEXT_LENGTH = 64
    DATASET_SUBSET_SIZE = 100
    NUM_EPOCHS = 1
    BENCHMARK_WARMUP = 2
    BENCHMARK_REPEATS = 5
    NEEDLE_LENGTH = 64
else:
    BATCH_SIZE = 32
    MAX_STEPS = 100000
    VAL_STEPS = 500
    CONTEXT_LENGTH = 512
    DATASET_SUBSET_SIZE = None  # Use full dataset
    NUM_EPOCHS = 3
    BENCHMARK_WARMUP = 10
    BENCHMARK_REPEATS = 50
    NEEDLE_LENGTH = 512

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

# Paths
OUTPUT_DIR = "output"
GLOBAL_TRACKER_FILE = os.path.join(OUTPUT_DIR, "global_tracker.json")
DATA_DIR = "data"
