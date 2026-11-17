Reconciliation with memory_stress_tests error_breakdown: 12/12 compared models identical
Min serial/chunked agreement: 1.0000

## Accuracy by rewrite count (mean over models)

| Rewrites | Latest value | Unrelated retention | First-replacement answers | Errors that are first replacement |
|---:|---:|---:|---:|---:|
| 1 | 99.98% | 97.42% | 0.00% | 0/1 |
| 2 | 67.95% | 95.29% | 30.92% | 1425/1477 |
| 3 | 64.50% | 95.03% | 32.29% | 1488/1636 |
| 4 | 65.52% | 93.10% | 31.18% | 1437/1589 |

## Causal patches, three rewrites (mean change vs unpatched over models; models improved/worsened by >=2 points)

| Patch | Layers | Latest value | Unrelated | First-replacement rate | Improved / worse |
|---|---|---:|---:|---:|---|
| beta_one_latest | all | -0.80% | -0.85% | +0.93% | 0 / 2 |
| beta_copy_rewrite1 | all | -0.02% | -0.30% | +0.04% | 0 / 1 |
| beta_zero_rewrite1 | all | +4.12% | -6.86% | -10.24% | 6 / 1 |
| key_align_latest | all | +16.21% | -18.40% | -23.65% | 14 / 0 |
| control_beta_one_unrelated | all | +0.95% | -0.39% | -0.89% | 2 / 0 |
| beta_one_latest | 0 | -0.07% | -0.46% | -0.04% | 0 / 0 |
| beta_one_latest | 1 | -0.17% | -0.11% | +0.11% | 0 / 0 |
| beta_one_latest | 2 | -0.26% | -0.02% | +0.69% | 0 / 1 |
| beta_one_latest | 3 | -0.24% | -0.30% | +0.20% | 0 / 1 |
| beta_zero_rewrite1 | 0 | +0.35% | -2.71% | -2.91% | 2 / 2 |
| beta_zero_rewrite1 | 1 | +1.06% | -2.13% | -2.45% | 2 / 0 |
| beta_zero_rewrite1 | 2 | +1.76% | -1.02% | -1.80% | 2 / 0 |
| beta_zero_rewrite1 | 3 | +0.76% | -0.69% | -3.56% | 2 / 0 |
| key_align_latest | 0 | +1.69% | -7.38% | -3.19% | 6 / 1 |
| key_align_latest | 1 | +7.66% | -5.92% | -11.61% | 10 / 0 |
| key_align_latest | 2 | +5.36% | -3.28% | -5.51% | 7 / 0 |
| key_align_latest | 3 | +4.30% | -1.41% | -4.38% | 3 / 0 |

## Write strength beta at the value token of each target write (0 = initial, 3 = latest) (three rewrites, pooled over models)

| Layer | Outcome | 0 | 1 | 2 | 3 | unrelated_control |
|---|---|---:|---:|---:|---:|---:|
| 0 | correct | 0.585 | 0.583 | 0.584 | 0.583 | 0.583 |
| 0 | first_rewrite | 0.598 | 0.602 | 0.598 | 0.599 | 0.600 |
| 1 | correct | 0.568 | 0.571 | 0.568 | 0.568 | 0.568 |
| 1 | first_rewrite | 0.576 | 0.573 | 0.574 | 0.577 | 0.572 |
| 2 | correct | 0.567 | 0.566 | 0.572 | 0.576 | 0.583 |
| 2 | first_rewrite | 0.571 | 0.569 | 0.565 | 0.571 | 0.568 |
| 3 | correct | 0.582 | 0.580 | 0.579 | 0.582 | 0.585 |
| 3 | first_rewrite | 0.585 | 0.590 | 0.589 | 0.595 | 0.592 |

## Write strength beta at the key token of each target write (three rewrites, pooled over models)

| Layer | Outcome | 0 | 1 | 2 | 3 | unrelated_control |
|---|---|---:|---:|---:|---:|---:|
| 0 | correct | 0.646 | 0.646 | 0.646 | 0.646 | 0.647 |
| 0 | first_rewrite | 0.575 | 0.575 | 0.575 | 0.575 | 0.577 |
| 1 | correct | 0.683 | 0.624 | 0.607 | 0.599 | 0.592 |
| 1 | first_rewrite | 0.744 | 0.690 | 0.613 | 0.608 | 0.564 |
| 2 | correct | 0.617 | 0.660 | 0.709 | 0.725 | 0.752 |
| 2 | first_rewrite | 0.682 | 0.655 | 0.683 | 0.708 | 0.777 |
| 3 | correct | 0.693 | 0.751 | 0.782 | 0.793 | 0.787 |
| 3 | first_rewrite | 0.695 | 0.756 | 0.743 | 0.775 | 0.806 |

## Cosine(key at write key token, query at answer position) (three rewrites, pooled over models)

| Layer | Outcome | 0 | 1 | 2 | 3 |
|---|---|---:|---:|---:|---:|
| 0 | correct | -0.006 | -0.006 | -0.000 | -0.011 |
| 0 | first_rewrite | -0.004 | 0.115 | 0.141 | 0.141 |
| 1 | correct | 0.065 | 0.041 | 0.032 | 0.028 |
| 1 | first_rewrite | 0.041 | 0.029 | -0.021 | -0.017 |
| 2 | correct | -0.042 | -0.001 | 0.075 | 0.103 |
| 2 | first_rewrite | -0.063 | 0.024 | 0.221 | 0.101 |
| 3 | correct | -0.033 | -0.006 | 0.074 | 0.142 |
| 3 | first_rewrite | -0.026 | -0.020 | 0.217 | 0.187 |

## Cosine(key at write value token, query at answer position) (three rewrites, pooled over models)

| Layer | Outcome | 0 | 1 | 2 | 3 |
|---|---|---:|---:|---:|---:|
| 0 | correct | 0.069 | 0.135 | 0.135 | 0.135 |
| 0 | first_rewrite | 0.040 | 0.125 | 0.126 | 0.125 |
| 1 | correct | 0.042 | 0.013 | -0.003 | -0.001 |
| 1 | first_rewrite | 0.075 | 0.050 | 0.042 | -0.017 |
| 2 | correct | -0.039 | 0.004 | 0.043 | 0.026 |
| 2 | first_rewrite | -0.087 | 0.060 | -0.010 | -0.007 |
| 3 | correct | -0.015 | -0.016 | -0.006 | -0.002 |
| 3 | first_rewrite | 0.018 | 0.071 | -0.048 | -0.057 |

## Readout attribution share: key token of each target write (three rewrites, pooled over models)

| Layer | Outcome | 0 | 1 | 2 | 3 |
|---|---|---:|---:|---:|---:|
| 0 | correct | 0.016 | 0.015 | 0.014 | 0.028 |
| 0 | first_rewrite | 0.024 | 0.018 | 0.012 | 0.026 |
| 1 | correct | 0.020 | 0.020 | 0.020 | 0.031 |
| 1 | first_rewrite | 0.029 | 0.050 | 0.078 | 0.033 |
| 2 | correct | 0.017 | 0.024 | 0.033 | 0.031 |
| 2 | first_rewrite | 0.021 | 0.055 | 0.115 | 0.110 |
| 3 | correct | 0.019 | 0.022 | 0.029 | 0.026 |
| 3 | first_rewrite | 0.029 | 0.055 | 0.088 | 0.179 |

## Readout attribution share: value token of each target write (three rewrites, pooled over models)

| Layer | Outcome | 0 | 1 | 2 | 3 |
|---|---|---:|---:|---:|---:|
| 0 | correct | 0.016 | 0.012 | 0.021 | 0.042 |
| 0 | first_rewrite | 0.028 | 0.013 | 0.019 | 0.030 |
| 1 | correct | 0.017 | 0.016 | 0.013 | 0.020 |
| 1 | first_rewrite | 0.022 | 0.024 | 0.042 | 0.018 |
| 2 | correct | 0.019 | 0.027 | 0.032 | 0.025 |
| 2 | first_rewrite | 0.034 | 0.041 | 0.034 | 0.030 |
| 3 | correct | 0.016 | 0.026 | 0.021 | 0.019 |
| 3 | first_rewrite | 0.030 | 0.069 | 0.033 | 0.039 |

## Readout attribution share by token role (three rewrites, pooled over models)

| Layer | Outcome | role_target_writes | role_other_writes | role_query_tokens | role_filler_and_other_query |
|---|---|---:|---:|---:|---:|
| 0 | correct | 0.162 | 0.492 | 0.114 | 0.231 |
| 0 | first_rewrite | 0.170 | 0.504 | 0.169 | 0.158 |
| 1 | correct | 0.157 | 0.261 | 0.294 | 0.288 |
| 1 | first_rewrite | 0.296 | 0.248 | 0.300 | 0.156 |
| 2 | correct | 0.207 | 0.270 | 0.101 | 0.423 |
| 2 | first_rewrite | 0.440 | 0.237 | 0.140 | 0.182 |
| 3 | correct | 0.178 | 0.265 | 0.125 | 0.432 |
| 3 | first_rewrite | 0.523 | 0.198 | 0.071 | 0.208 |

## Beta at value token, one rewrite (trained distribution), pooled

| Layer | initial | rewrite |
|---|---:|---:|
| 0 | 0.594 | 0.593 |
| 1 | 0.571 | 0.572 |
| 2 | 0.569 | 0.568 |
| 3 | 0.583 | 0.583 |