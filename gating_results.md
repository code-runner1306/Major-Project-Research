


**Setup:** 350M-class decoder-only models (20 to 24 layers, hidden size 1024, about 312M parameters, within about ±2% across architectures), 15B FineWeb-Edu tokens, LLaMA-2 tokenizer, sequence length 4096, global batch 128, bf16. AdamW and Muon (NorMuon). Warmup-Stable-Decay schedule, weight decay 0.1, grad clip 1.0. Metric: held-out validation cross-entropy at the final checkpoint ("saturation loss" is measured before the final decay). "Relative speed" is normalized **training** throughput, with Gated DeltaNet + AdamW (pure) = 100%.

Gated DeltaNet uses a scalar decay; Kimi Delta Attention (KDA) uses channel-wise decay.

| Architecture | Optimizer | Stack | Relative speed | Saturation loss | Final loss |
|---|---|---|---:|---:|---:|
| Gated DeltaNet | AdamW | hybrid | 90.5% | 2.517 | 2.356 |
| Gated DeltaNet | AdamW | pure | 100.0% | 2.601 | 2.433 |
| Gated DeltaNet | Muon | hybrid | 89.5% | 2.464 | 2.321 |
| Gated DeltaNet | Muon | pure | 95.0% | 2.543 | 2.397 |
| Kimi Delta Attention | AdamW | hybrid | 74.4% | 2.471 | 2.328 |
| Kimi Delta Attention | AdamW | pure | 73.6% | 2.490 | 2.347 |
| Kimi Delta Attention | Muon | hybrid | 70.9% | 2.409 | 2.273 |
| Kimi Delta Attention | Muon | pure | 68.5% | 2.427 | 2.290 |
| Gated DeltaNet-2 | AdamW | hybrid | 83.3% | 2.520 | 2.379 |
| Gated DeltaNet-2 | AdamW | pure | 81.9% | 2.601 | 2.452 |
| Gated DeltaNet-2 | Muon | hybrid | 81.3% | 2.466 | 2.345 |
| Gated DeltaNet-2 | Muon | pure | 80.7% | 2.542 | 2.415 |

**KDA (channel-wise) minus Gated DeltaNet (scalar), final loss, computed by me:** hybrid AdamW −0.028, pure AdamW −0.086, hybrid Muon −0.048, pure Muon −0.107. Channel-wise is lower in all four.

**KDA training throughput relative to Gated DeltaNet under the same optimizer and stack, computed by me:** 0.82 (hybrid AdamW), 0.74 (pure AdamW), 0.79 (hybrid Muon), 0.72 (pure Muon), i.e. about 18 to 28% lower.

**Why it matters:** this is the only source I found at this scale with a runtime cost for channel-wise gating alongside a loss gain, so it is the closest analogue of the quality-versus-cost question in our proposal. It is a single run per cell, and I did not see whether the kernels are equally optimized. For reference, Gated DeltaNet-2 has a higher final loss than Gated DeltaNet in the pure stacks here (2.452 vs 2.433 with AdamW).



**Setup:** 340M-parameter Gated DeltaNet-2 models (24 layers, hidden size 1024, 6 heads), 15B FineWeb-Edu tokens, train length 2048, AdamW lr 3e-4, WSD schedule, 524,288-token batch. Five arms, each run with two seeds (ten runs).

This paper does **not** compare scalar and channel-wise decay. It is included for the size of run-to-run noise at this scale:

| Seed | GDN-2 | QED (full) | QED-rand | QED-nogate | QED-nogate-noproj |
|---|---:|---:|---:|---:|---:|
| 0 | 2.3714 | 2.3719 | 2.3712 | 2.3718 | 2.3724 |
| 1 | 2.3746 | 2.3735 | 2.3735 | 2.3733 | 2.3741 |

Final validation loss (nats). All ten runs span only 0.0034 nats (2.3712 to 2.3746), and within a seed at most 0.0013. Its ablation effects for the gate and the projection **reverse sign between the two seeds**, and its authors say component-level conclusions need more runs. Takeaway for us: loss gaps of a few thousandths of a nat need many seeds before they mean anything.


