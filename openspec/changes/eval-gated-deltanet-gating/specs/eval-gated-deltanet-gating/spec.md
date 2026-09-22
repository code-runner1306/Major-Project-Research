## ADDED Requirements

### Requirement: Model Initialization and Swappable Gating Granularity
The system SHALL initialize Gated DeltaNet models in four configurations (10M, 30M, 55M, 100M parameter scales) with a swappable decay gating mechanism supporting both Variant A (scalar, head-wise gating) and Variant B (channel-wise gating).

#### Scenario: Swapping gating mechanism
- **WHEN** the model config specifies either "scalar" or "channel-wise" variant
- **THEN** the model initializes the appropriate decay projection dimension and performs the recurrent update with that gating resolution.

### Requirement: CPU and CUDA Compatible Recurrent Training Loop
The training module SHALL execute training steps and validation loops, automatically detecting and running on CUDA if available, and falling back to CPU otherwise, while maintaining identical results across platforms.

#### Scenario: Auto device detection
- **WHEN** PyTorch runs on a system with a CUDA-enabled GPU vs. CPU-only
- **THEN** the system moves tensors to the active device and executes training without raising device mismatch errors.

### Requirement: Robust TinyStories Data Ingestion
The dataset loading module SHALL stream or load the `roneneldan/TinyStories` dataset from Hugging Face and retry downloads upon failure until successful, ensuring complete dataset integrity.

#### Scenario: Retrying on connection failure
- **WHEN** the initial Hugging Face API call fails due to network timeout or rate limits
- **THEN** the system logs the failure and retries the connection until the dataset is loaded successfully.

### Requirement: Inference Latency and Throughput Benchmarking
The evaluation module SHALL measure wall-clock latency (milliseconds/token) and decode throughput (tokens/second) for each trained model using standard warm-up steps and multiple repeated measurement loops.

#### Scenario: Running throughput benchmark
- **WHEN** the benchmark function is called on a model
- **THEN** it executes warmup iterations, times subsequent generation steps, and returns the average ms/token and tokens/sec metrics.

### Requirement: Needle-in-a-Haystack Retrieval Evaluation
The evaluation module SHALL probe long-context retrieval capabilities by inserting a key-value fact ("needle") inside a long distractor text ("haystack") and checking if the model predicts the correct target value.

#### Scenario: Evaluating fact retrieval
- **WHEN** a needle fact like "The mouse is in the bedroom" is hidden in the haystack and queried
- **THEN** the evaluation logs whether the model successfully outputs the target word "bedroom".

### Requirement: Global Tracker Aggregation and Smoke Test Execution
The orchestrator SHALL run the full cartesian product grid of 4 scales × 2 variants × 3 seeds (24 runs) sequentially, saving individual run metrics and updating a master `global_tracker.json` file. It SHALL also support a `SMOKE_TEST` mode that runs exactly 2 training steps/epochs to verify the end-to-end flow.

#### Scenario: Smoke test execution
- **WHEN** the orchestrator runs with SMOKE_TEST enabled
- **THEN** it completes all 24 runs within a few minutes on a CPU, writes individual run folder outputs, and updates the global tracker.
