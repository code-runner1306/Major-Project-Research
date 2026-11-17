# Experiment three: runtime estimate only

Proposed comparison: add a joint timescale/sensitivity initialization to scalar and channel-wise models on MQAR and overwrite, for seeds 42–44. This requires **12 freshly trained models**, four per seed, at 16 epochs / 3136 optimizer steps each. Existing 24 checkpoints can be retained as reference conditions after compatibility checks. Resuming old checkpoints would not test a new initialization.

Using measured two-worker sessions on the **NVIDIA RTXA6000-12Q virtual GPU profile**:

| Work | Estimated elapsed time |
|---|---:|
| MQAR scalar/channel pair | 56.1 minutes |
| Overwrite scalar/channel pair | 56.7 minutes |
| One seed, both tasks / four models | 1 hour 53 minutes |
| Three seeds / twelve models | 5 hours 38 minutes |
| Three seeds plus 20% margin and 15 minutes setup/publication | About 7 hours |
| Also run a separate four-model qualification pilot, including margin | About 9 hours 16 minutes |

These estimates assume the same model, batches, precision, evaluation cadence, implementation, and two-worker throughput. Joint fitting overhead has not been benchmarked. Implementation/debugging time is separate. This is not a guaranteed Kaggle or unrestricted A6000 estimate; hardware/session throughput must be checked. There is no new training result for experiment three yet.

The initializer needs a verified adaptation to our one-head, four-layer setting; describe it as DAMI-style unless it exactly reproduces the published recipe. Select target settings before observing new comparison results. See `estimate.json` for numerical assumptions.
