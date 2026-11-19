# Assessment of seed-42 easy-task learning-rate sweeps

All six runs completed 2,000 steps: scalar/default, 3,329,784 parameters, two keys, length 128, vocabulary 8192, shuffled 10,000-example training pool (6.4 epochs), 128 repeatedly monitored development examples. These are not the four-condition paper experiment.

| Task | LR | Train accuracy | Development accuracy | Hours | Shortcut audit |
|---|---:|---:|---:|---:|---|
| mqar | 0.001 | 16.02% | 0.00% | 2.50 | True |
| mqar | 0.003 | 0.39% | 0.00% | 2.50 | True |
| mqar | 0.0003 | 30.08% | 0.00% | 2.50 | True |
| overwrite | 0.001 | 91.02% | 88.28% | 2.63 | False |
| overwrite | 0.003 | 59.77% | 36.33% | 2.56 | False |
| overwrite | 0.0003 | 28.91% | 0.00% | 2.65 | False |

## Interpretation

Overwrite at LR 0.001 learns after the earlier 200-step floor: development accuracy rises from 0% at step 600 to 5.47% at 800, 22.66% at 1000 and 88.28% at 2000. This supports budget/optimization sensitivity on this easier distribution. Shuffling and repeated epochs also changed; improvement cannot be attributed to step count alone.

Overwrite still fails a per-cell shortcut audit: last-value copying is perfect for the unrelated/zero-overwrite subgroup and scores 50% overall. Nevertheless 226/256 correct exceeds that baseline by 98 answers. Given one latest and one unrelated query per example, at least 98/128 latest queries must be correct. The shortcut warning does not make all observed learning meaningless, but prevents a clean memory-mechanism claim on this distribution.

MQAR development accuracy is exactly zero at all ten checkpoints for all three rates, including teacher-forced evaluation. Training accuracy rises to 30.08% at LR 0.0003, suggesting training-set-specific learning. This narrows the problem but does not isolate an implementation, optimization or capacity failure.

## Required next steps

1. Do not launch more full factorial seeds yet. Run a causal attention control alongside GDN on identical four-key short-context data. Require independent oracle and key-change audits, with complete key-free baselines, before training. Fix the baseline handling of later MQAR query ordinals (the final-two-values baseline currently produces incomplete checks beyond ordinal 1) before interpreting a four-key audit failure.
2. Report recent-training, fixed-training and separate development accuracy, first/later query accuracy, latest/unrelated overwrite accuracy, gradient norms, learning rates and step times. Attention is a positive control, not proof of a bug if GDN fails.
3. If clean overwrite learning is established, lock its generator, budget, LR 0.001 as the selected exploratory rate, vocabulary and real parameter counts. Compare C1/C3/C4/C6 on fresh paired seeds 43-45 and fresh confirmation data. Use identical budgets; retain seed 42 and these monitored data as development evidence.
4. Restore the intended retention-versus-overwrite claim only if MQAR also becomes learnable. Otherwise narrow the paper scope transparently or present the diagnostics as a preliminary study; do not claim a granularity effect from these scalar-only sweeps.
5. Do not increase to a nominal 10M model or blindly train longer before the attention control, since both change the diagnostic question and consume the deadline.

This review imports only the six diagnostic summaries, not the sweep branches' unrelated source/temporary-file changes. Original branches remain untouched.
