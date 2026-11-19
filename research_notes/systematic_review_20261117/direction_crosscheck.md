# Direction cross-check: subjective novelty and feasibility scores

Date: 17 November 2026. These are **research-positioning scores**, not acceptance probabilities and not estimates of publication outcome. They measure residual room after the literature reviewed so far, assuming competent execution and honest attribution.

## Newly checked primary overlap

- [Elucidating the Design Space of Decay in Linear Attention](https://arxiv.org/abs/2509.05282), accepted COLM 2025, explicitly studies parameterization, parameter sharing, scalar versus vector decay granularity, and positional encoding. Its abstract reports that vector decay often wins under matched parameterization, while scalar can win under other parameterizations. This is direct prior art against a broad fair scalar/channel decay comparison.
- [Adaptive Memory Decay for Log-Linear Attention](https://arxiv.org/abs/2605.06946), 2026, learns input-dependent per-token/per-level decay and evaluates associative recall, selective copying, and language modeling. It is a different Fenwick/log-linear architecture, but it reinforces that adaptive-decay versus fixed-decay memory tradeoffs are established beyond GDN.

## Direction scores

Scale: residual novelty 0–10 (10 = strong unoccupied question after prior-art controls); direct-overlap risk 0–10 (10 = almost entirely covered); execution fit 0–10 (10 = can be answered with our existing recipe/checkpoints plus a modest new run).

| Direction | Residual novelty | Direct-overlap risk | Execution fit | Judgment |
|---|---:|---:|---:|---|
| Fair initial-decay comparison: default vs mean-alpha matching vs DAMI-style joint matching, scalar/channel crossed on MQAR and overwrite | 3/10 | 9/10 | 8/10 | The factorial structure is close to Das; activation-based joint initialization is close to DAMI; decay-design comparison is covered by COLM 2025. Valuable as a transparent replication/boundary study, weak as a new method. Fresh checkpoints are required for a fair comparison, but the paper must attribute all baselines. |
| Boundary of dynamic-decay necessity: fixed per-channel, shuffled channels, dynamic decay, repeated-write/load/context OOD, with causal interventions and varied rewrite training | 5/10 | 6/10 | 7/10 | There is meaningful remaining room in the exact controlled protocol and failure taxonomy. Prior work already studies static channel gates, causal interventions, load, editability, and repeated overwrite separately. Novelty depends on showing a reproducible boundary and ruling out training shortcuts, parameter-count imbalance, and embedding/gradient failures. |
| Speed versus robust memory: throughput/validation/checkpoint cost against MQAR load, repeated-write recall, stale-answer rate, and retention | 3/10 | 7/10 | 6/10 | Efficient linear attention and hardware studies already report throughput/quality tradeoffs. Our A6000/T4 measurements are useful engineering evidence but not a strong scientific novelty claim unless paired with a prespecified accuracy-per-time or memory-quality frontier across matched models. |
| Resource-constrained small training: reproducible 1.6M-parameter GDN study with two-worker scheduling, complete logs/checkpoints, CPU confirmation, and independent stress evaluation | 4/10 | 5/10 | 9/10 | Reproducibility under a 12-hour GPU budget can be a useful systems/experimental contribution, especially if artifacts are public and the protocol exposes seed sensitivity. It is not novelty merely because the hardware is limited; it needs a general lesson, such as a low-cost diagnostic protocol that changes conclusions relative to clean validation alone. |

## What the scores mean for the current project

The strongest direction is the **boundary study**, with the resource-constrained protocol as its reproducibility contribution. The initial-decay comparison should be included as a prior-art-grounded baseline, not the headline invention. The speed analysis should support the experimental design and report an accuracy-per-hour frontier, not be presented as a new memory mechanism.

Training new checkpoints is still appropriate. It means we are experimentally testing attributed initialization recipes and controls on our fixed one-head/four-layer model; it does **not** imply that we are proposing a new architecture. We should label the resulting cells as fresh comparisons, preserve exact seeds/data manifests, and avoid reusing old checkpoints to claim a new initialization result.

## Minimum design that could justify the boundary claim

1. Train fresh default, old mean-alpha, and DAMI-style joint initializations for scalar and channel-wise variants on both tasks, using the same data exposure and three seeds.
2. Keep the training distribution separate from final tests: include one-rewrite training, varied-rewrite training, and unseen two/three-rewrite evaluation.
3. Report answer-free newest-value accuracy, unrelated-association retention, stale-answer rate by obsolete ordinal, and MQAR accuracy by memory load and distance.
4. Add parameter-matched control or an explicit capacity correction, since the current channel variant has approximately 4.17% more trainable parameters.
5. Audit embedding norms, gradient clipping/nonfinite events, state/readout magnitudes, and optimizer timing before assigning failures to decay gates.
6. Use an accuracy-per-hour plot only as a secondary systems result.

## Bottom line

No score here is a guarantee of novelty or acceptance. The literature makes the broad claims low-novelty. A careful, small-model, reproducible boundary study remains plausible if it demonstrates a specific interaction that survives fresh controls and explains why clean MQAR success does not imply reliable repeated updates.
