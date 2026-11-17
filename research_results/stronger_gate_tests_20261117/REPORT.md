# Stronger gate interventions: completed CPU study

All twelve channel-wise checkpoints (two tasks, default/calibrated initialization, seeds 42–44) were evaluated without changing trained weights. These are 16-epoch exploratory checkpoints. Runtime: 10.29 minutes on CPU, concurrently with the stress study.

## Protocol

Use the cached exact training-pool prefix: first 256 examples fit per-channel mean decay values; the next 256 examples measure interventions. Earlier answer tokens are erased during evaluation. Static constants use no accuracy optimization. Consequently these are training-side mechanism probes, not fresh held-out generalization evidence.

Global mean replaces all channel decay gates with a shared channel average at each token. Global static retains distinct learned channel constants while removing their input dependence. Global shuffle permutes gate coordinates, preserving each local gate multiset; results average three fixed permutation seeds per trained model. Single-layer tests average gates in one layer only. Other input-dependent operations, including write gates, remain active. Parameter fingerprints were checked and intervention methods restored.

## Accuracy averaged over three training seeds (%)

| Task / initialization | Original | Shared mean | Static per channel | Shuffled |
|---|---:|---:|---:|---:|
| MQAR / default | 100.00 | 82.16 | 100.00 | 98.94 |
| MQAR / calibrated | 100.00 | 69.63 | 100.00 | 80.51 |
| Overwrite / default | 99.61 | 83.66 | 99.48 | 78.23 |
| Overwrite / calibrated | 99.48 | 88.28 | 93.49 | 87.65 |

All six MQAR checkpoints retain 100% with fixed per-channel decay. Thus the shared-mean failure alone cannot establish that token-dependent decay is necessary: preserving different channel settings suffices on this probe. This does not establish that static decay suffices on harder inputs or when trained from scratch.

Sensitivity to gate-coordinate assignment is uneven. MQAR calibrated seed 43 falls to 42.15% averaged over shuffles, whereas the other five MQAR models remain near 98–100%. Its layer-one mean intervention loses 68.95 percentage points. Default seed 44 instead loses 25 points when layer three is averaged. Effects are not additive: some models tolerate every individual layer intervention but suffer when all layers are changed together.

Overwrite depends more on dynamic decay in some runs: calibrated seed 44 falls from 100% to 86.52% with static decay. Default overwrite seed 43 tolerates all these interventions. Report seed-level results rather than claiming a universal mechanism or architectural advantage.

## Scope and artifacts

These interventions reveal reliance of already-trained networks; they are not fair comparisons against independently trained scalar/static architectures. Channel models also have about 4.17% more parameters. No novelty claim follows from static-channel gating itself.

See `per_model_summary.csv`, `models.csv`, `condition_means.csv`, `results.json`, `protocol.json`, `global_interventions.png`, and `layer_interventions.png`. Reproduce with `stronger_gate_tests.py`; summarize with `summarize_followup_cpu.py` (see script arguments).
