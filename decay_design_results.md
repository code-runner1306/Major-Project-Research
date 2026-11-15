# scalar vs channel-wise question

## 1. Variants

The data-dependent pair relevant to us is **DeltaNet (scalar) vs KDA (channel-wise)**: same recurrence, same delta rule, differing only in gate granularity. The source repo also ran GLA and four data-independent static-gate variants (a 2×2 over granularity × delta rule), but those are not implemented in our codebase, so their results have been removed from the tables below. The "DeltaNet" row has a data-dependent scalar gate, so it is the closest analogue of Gated DeltaNet; check the paper's own definition before citing it as GDN.

## 2. Training setup (from the repo's code defaults and logs)

| Item | Value |
|---|---|
| Model sizes | 10M: d=256, L=6, H=4. 55M: d=512, L=8, H=8. 100M: d=768, L=12, H=12 (10.0M, 55.0M and 100.0M parameters respectively) |
| Optimizer | AdamW, lr 3e-4, weight decay 0.1, warmup then decay (min(500, 10% of steps) warmup) |
| Batch and context | batch 8, sequence length 512 (1024 for the seq1024 runs) |
| Data budget | `max_train_tokens` = 10M unique tokens, 3 epochs, about 7,300 optimizer steps (about 30M token-passes). seq1024 runs: batch 8 × 1024, about 3,650 steps |
| Validation | 500k validation tokens available; `evaluate()` uses at most 50 batches. Final value = end of epoch 3 |
| Hardware | developed on a MacBook Pro M4 Max (MPS) with naive recurrent loops, no Triton kernels (per the README) |

One thing worth noting: **this paper's own budget is small** — about 3.0 token-passes per parameter at 10M and 0.3 at 100M, over only 10M unique tokens. A budget of that size was accepted at TMLR, so it is not by itself disqualifying for us, though it does not mean a bigger budget would show the same ordering.

## 3. Main results

### 3.1 10M parameters, TinyStories

| Variant | Gate | Conditioning | Delta rule | Params (M) | Val loss mean ± pop. std (3 seeds) |
|---|---|---|---|---:|---|
| DeltaNet (scalar, gated) | scalar | data-dependent | yes | 10.0 | 2.3222 ± 0.0173 |
| KDA | channel-wise | data-dependent | yes | 10.0 | **2.2820 ± 0.0196** |

### 3.2 55M parameters, TinyStories (single seed, no error bars)

| Variant | Params (M) | Val loss (1 seed) |
|---|---:|---:|
| DeltaNet (scalar, gated) | 55.0 | 2.0176 |
| KDA | 55.0 | **1.9703** |

### 3.3 100M parameters, TinyStories

| Variant | Gate | Conditioning | Delta rule | Params (M) | Val loss mean ± pop. std (3 seeds) |
|---|---|---|---|---:|---|
| DeltaNet (scalar, gated) | scalar | data-dependent | yes | 100.0 | 1.9004 ± 0.0061 |
| KDA | channel-wise | data-dependent | yes | 100.0 | **1.8614 ± 0.0017** |

### 3.4 10M parameters, TinyStories, context 1024 (single seed, about 3,650 steps)

| Variant | Params (M) | Val loss (1 seed) |
|---|---:|---:|
| DeltaNet (scalar, gated) | 10.0 | 2.4476 |
| KDA | 10.0 | **2.4457** |

## 4. Channel-wise minus scalar gaps (negative = channel-wise better)

Computed by me from the per-seed logs. Seeds are paired by label only.

| Setting | Contrast | Per-seed gap (seed 42 / 43 / 44) | Mean gap | Sign |
|---|---|---|---:|---|
| 10M TinyStories | KDA − DeltaNet | -0.0386 / -0.0353 / -0.0468 | -0.0402 | all 3 seeds negative |
| 100M TinyStories | KDA − DeltaNet | -0.0423 / -0.0443 / -0.0302 | -0.0389 | all 3 seeds negative |

**What this says:**

- The **KDA − DeltaNet gap is negative in all 6 seed-pairs**, about 0.04 nats at both sizes (mean −0.0402 at 10M, −0.0389 at 100M). Channel-wise decay wins at every size and every seed we have. This is the comparison closest to our research question.

**Parameter counts.** Both rows at a given size are the same model (10.0M, 55.0M, 100.0M), so the contrast is parameter-matched and is not confounded by differing capacity.


## Appendix: per-seed validation losses

### 10M TinyStories
| Variant | seed 42 | seed 43 | seed 44 |
|---|---:|---:|---:|
| DeltaNet (scalar, gated) | 2.3465 | 2.3125 | 2.3076 |
| KDA | 2.3079 | 2.2772 | 2.2607 |

### 100M TinyStories
| Variant | seed 42 | seed 43 | seed 44 |
|---|---:|---:|---:|
| DeltaNet (scalar, gated) | 1.9013 | 1.9073 | 1.8924 |
| KDA | 1.8590 | 1.8630 | 1.8622 |
