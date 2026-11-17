# Joint timescale and sensitivity experiment

Run from the activated training environment:

```powershell
powershell.exe -File .\run_joint_recipe.ps1
```

Publishing defaults to **false**. Results, logs, timing, optimizer state, and the latest model checkpoint stay under a unique `output/joint_recipe_study/joint-timescale-sensitivity-v1/<session>/` folder. Each family/seed has separate scalar and channel-wise subfolders. No existing checkpoint is resumed or overwritten. Checkpoints are saved at every epoch, replacing that model's previous epoch checkpoint.

Default: seeds 42, 43, 44; both tasks; one new initialization; 12 models total. Two processes train scalar and channel-wise in parallel on GPU 0, then proceed to the next task/seed. Use `-Devices '0,1'` for two GPUs. Validation uses the unchanged 512-example development split every epoch, including answer-free and secondary teacher-forced overwrite evaluation. Training uses 16 epochs / 3136 steps, effective batch 512, microbatch 32, FP32, vocabulary 8192, length 64, LR 0.00215443469, and the original 64-epoch scheduler horizon.

```powershell
# Inspect configuration without training or publishing
.\run_joint_recipe.ps1 -PlanOnly

# One seed, both families: four models
.\run_joint_recipe.ps1 -Seeds 42

# Split sessions by family
.\run_joint_recipe.ps1 -Family overwrite -SeedCount 3 -StartSeed 42

# Tiny end-to-end CPU infrastructure check (not scientific evidence)
.\run_joint_recipe.ps1 -CpuSmoke -Seeds 42
```

The 11-hour workload ceiling reserves time before a 12-hour session limit. A pair is deferred if its conservative 100-minute admission estimate does not fit. During training, the first 50 optimizer steps trigger a speed check; nonfinite gradients, failed data audits, backend mismatch, or budget exhaustion stop the run. Inspect `session_status.json` for complete/deferred/failed status and individual logs for errors. Do not interpret a successful tiny CPU smoke as GPU qualification or task learnability.

## Scientific definition

This is a **DAMI-style adaptation**, not an exact published reproduction. Following [DAMI equations 10–11](https://arxiv.org/html/2610.41994v2), collect projected activations from the first 256 training examples, without labels or gradients. In layer order, solve the bias to match mean sigmoid sensitivity, then set the decay multiplier to match mean log decay. Only `dt_bias` and `A_log` change. Shared backbone initialization and training data must match within each scalar/channel pair; numerical reference/chunked optimizer qualification runs before training each model.

We retain the previous four layer half-life targets **16, 32, 64, 128** and adopt sensitivity targets **0.25, 0.75, 0.75, 0.25**. With one head there is no multihead spectrum. Our channel extension fits each existing channel independently, preserving the model's existing parameterization; it does not reproduce KDA's head-shared multiplier. No hyperparameters were selected using new training results.

The matched quantity is `E[log(alpha)] = -log(2)/half_life`; this differs from the previous `E[alpha] = 2**(-1/half_life)` calibration. Thus this comparison changes both the sensitivity treatment and decay-matching statistic. It tests the joint recipe as a package and cannot isolate sensitivity alone without an additional matched-log-decay-only control. Sensitivity refers to the Softplus factor, not the full decay derivative or a measured memory lifetime.

The original runner/model/calibration files are unchanged. The isolated runner records source hashes and a distinct protocol ID. Existing 24 models remain separate reference runs, with compatibility assessed before aggregation. Fresh held-out evaluation and the same stress grid are still needed after training; this launcher does not automatically run the CPU stress suite or assert success of the scientific hypothesis.

Estimated time on the measured **RTXA6000-12Q** profile with two workers: 1h53m per seed, 5h38m for all three; budget about 7h. Joint calibration and GPU qualification overhead are not yet measured for the full-size new initialization. Hardware and load can change this estimate.
