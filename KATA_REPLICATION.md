# KATA repeated-key overwrite: replication and rewrite-shortcut test

Paper: Ghriss and Chakraborty, *Kernelized Linear Attention: Breaking the Capacity Wall with Symmetric
Cones*, arXiv 2607.17419v1. We use Section 5.2, Table 2 and Appendix F.

## What is replicated

The **Gated DeltaNet, dh=64** row of Table 2, which the paper reports as 1,167,878 parameters and
overwrite accuracy 0.998 / 0.853 / 0.384 / 0.217 at K = 64 / 128 / 256 / 384.

| Item | Paper (Appendix F) | Here |
|---|---|---|
| Backbone | 2-layer Zoology TransformerBlock, Hybrid(BaseConv k=3, mixer), Identity state mixer, no positional embeddings, tied head, vocabulary 8192 | Same (`kata_model.py`): layer 0 BaseConv, layer 1 GDN |
| GDN mixer | FLA GatedDeltaNet, use_gate=False, short conv k=4, q/k L2 norm, value expansion 2 | Same equations, in our portable implementation of FLA's |
| Size | 1,167,878 parameters | **1,167,876**: d_model 128 with 2 heads, as the Table 2 counts imply (F.4's prose says one head) |
| Optimiser | AdamW β=(0.9, 0.95), weight decay 0.1 except A_log/dt_bias, gradient clip 1.0 | Same |
| Schedule | batch 256, 32-epoch cosine, no warmup; learning rate swept over {1e-3, 3e-3, 1e-2}, best reported; stop when the hardest extrapolation slice reaches 0.999 | Same; cosine stepped per epoch; early stop on K=384 |
| Training data | K ∈ {4, 8, 16, 32}, lengths up to 256, each key written twice | Same; 100,000 examples split over the four K |
| Evaluation | K ∈ {64, 128, 256, 384}, T ≤ 3072 | Same, T = 6K (up to 2304) |

## What we had to choose

The paper does not publish its overwrite data code, and upstream Zoology has no overwrite generator.

- **Sequence layout.** We extend the paper's own MQAR layout (F.1) directly: write pass 1 covers all
  K keys, write pass 2 rebinds all K keys with new values in a new order, then the K queries follow in
  random order, each followed by an answer slot. This gives T = 6K, consistent with the stated limits
  (K = 32 → 192 ≤ 256; K = 384 → 2304 ≤ 3072).
- **Training set size.** 100,000 examples, the Zoology `multiquery_ar` default. The paper does not state
  this.
- **Seed.** 42. The paper reports one run per row.

If the reproduced K = 64..384 accuracies come out close to the paper's, these choices are probably
close to theirs. We report both sets of numbers either way.

## The shortcut test

Training always writes each key exactly twice, so "second write", "write in pass 2" and "latest write"
coincide. The evaluation breaks that coincidence on fresh data:
- one pass;
- three passes;
- four passes;
- a third pass that rewrites only half of the keys.

Every wrong answer is traced to the write it came from. The predictions R1-R5 are frozen in
`run_kata_replication.py` before training.

## Run on the A6000

```powershell
powershell.exe -File .\run_A6000_kata_replication.ps1
```

This trains 3 models (one per learning rate) concurrently on GPU 0. After it finishes:

```powershell
powershell.exe -File .\run_A6000_kata_replication.ps1 -Evaluate
```

Then push `output/kata_replication/<session>` (the `*.pt` checkpoints are ignored by git) and
`research_results/kata_replication_eval_*`.
