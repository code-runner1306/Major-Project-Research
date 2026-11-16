# Single A6000 recipe timing

From the repository root in PowerShell, with the CUDA-enabled Python environment
already installed:

```powershell
.\run_A6000_recipe_timing.ps1 -Python .\.venv\Scripts\python.exe -Family both
```

Use `-Family mqar` or `-Family overwrite` to select just one task. The launcher
uses physical GPU 0, restores its environment variables afterward, and does not
install or replace PyTorch. All repository dependencies must already be installed.

The benchmark first checks recurrence outputs, states, gradients, and five-step
optimizer trajectories against the reference implementation. A failed check stops
the benchmark. It then measures each task with scalar/channel-wise models first
sequentially and then simultaneously sharing the same GPU. Each worker starts
fresh and performs 50 optimizer steps on the actual 100,000-example task with
FP32, microbatch 32, effective batch 512, and validation on 512 examples. The
development training recipe is unchanged. No final training run is launched.

Find `benchmark_report.json` and worker logs in the printed output directory.
The report includes individual step times, measured end-to-end throughput, and
pair completion projections for 64 epochs / 12,544 steps. Estimates include setup,
validation, and checkpoint writing; conservative estimates add timing margins and
a reserve. These are short-run projections, not guarantees. Timing accuracy is not
learnability evidence. GPU out-of-memory or worker errors fail the benchmark and
preserve completed reports/logs. Checkpoints remain local; nothing is pushed.

For local plumbing verification only (tiny CPU data, not A6000 performance):

```powershell
.\run_A6000_recipe_timing.ps1 -Python .\.venv\Scripts\python.exe -CpuSmoke
```
