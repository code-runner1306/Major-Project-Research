# Kaggle corrected V4 seed 42: exploratory assessment

All eight runs finished 1,000 optimizer steps and their held-out evaluations; all four session summaries report success. This is one paired seed, not a replicated statistical result. Source revision `a7722bc9aac95b78804e79c79638f1b3e5e8ff5e`; learning rate 0.001; 512 validation and 512 held-out examples, V4 generator, float32 state.

## Held-out exact-query accuracy and final validation loss

| Condition | Family | Correct / queries | Accuracy | Final validation loss | Trainable parameters |
| --- | --- | ---: | ---: | ---: | ---: |
| C1 | mqar | 0/4772 | 0.0000% | 8.3287 | 3,329,784 |
| C1 | overwrite | 0/1024 | 0.0000% | 8.4735 | 3,329,784 |
| C3 | mqar | 1/4772 | 0.0210% | 8.3287 | 3,480,480 |
| C3 | overwrite | 0/1024 | 0.0000% | 8.4827 | 3,480,480 |
| C4 | mqar | 0/4772 | 0.0000% | 8.3279 | 3,329,784 |
| C4 | overwrite | 0/1024 | 0.0000% | 8.2803 | 3,329,784 |
| C6 | mqar | 1/4772 | 0.0210% | 8.3278 | 3,480,480 |
| C6 | overwrite | 2/1024 | 0.1953% | 8.2287 | 3,480,480 |

MQAR scores count 4,772 queries across 512 examples; overwrite scores count 1,024 queries across 512 examples. Comparisons share the same held-out generator hash within each family. Full metrics and validation histories are in the condition directories. The last validation loss is calculated on the validation set with ordinary teacher-forced inputs; held-out retrieval uses answer-free history.

## Interpretation

There is no useful held-out retrieval signal in this run: all scores are 0–0.20%, so these data do not establish an advantage for channel-wise versus scalar gating or a task-dependent interaction. This is a floor result under this training budget, not evidence that the approaches are equivalent. The nonzero counts are 1/4,772 (MQAR C3 and C6) and 2/1,024 (overwrite C6).

Overwrite validation loss is lower with calibrated initialization (C4 8.2803 versus C1 8.4735; C6 8.2287 versus C3 8.4827). This is an exploratory optimization signal, not demonstrated retrieval competence or a causal generalization across seeds. MQAR final validation losses cluster near 8.328. With 8,192 vocabulary symbols and 4,096 possible value symbols, ln(4,096) is approximately 8.318; this is a useful uniform-value reference, not an empirically measured baseline.

The requested posthoc grouped-gate diagnostics were computed for the channel-wise models, but the near-zero retrieval floor prevents a meaningful accuracy-based inference about the value of gate granularity. The study scale is labeled 10M but these artifacts record approximately 3.33M scalar and 3.48M channel-wise trainable parameters; compare capacity before claiming a parameter-matched effect.

Before funding additional full seeds, diagnose learnability with a smaller/easier controlled V4 task and measure an empirical random/shortcut baseline, training and held-out answer accuracy, and data/label alignment. Preserve the intended answer-free held-out evaluation. If a predeclared setup learns the task, lock it before collecting fresh paired seeds. Checkpoint weights were intentionally excluded from these branches, so re-evaluation or checkpoint inspection requires copies saved separately in Kaggle.

## Provenance and integrity

Each source branch is an independent orphan results branch; bundles were placed under condition-specific directories to avoid collisions. SHA-256 and size for every file listed in each export manifest matched after correcting Git-for-Windows CRLF conversion. The manifest excludes model checkpoints. Git history integration uses a consolidated commit with these source branch names:

- C1: `kaggle-results/seed42_mqar-C1_overwrite-C1_20261114T113610Z_f6306c3d` (`a7722bc9aac95b78804e79c79638f1b3e5e8ff5e`)
- C3: `kaggle-results/seed42_mqar-C3_overwrite-C3_20261114T114147Z_b8309441` (`a7722bc9aac95b78804e79c79638f1b3e5e8ff5e`)
- C4: `kaggle-results/seed42_mqar-C4_overwrite-C4_20261114T115029Z_71a436f3` (`a7722bc9aac95b78804e79c79638f1b3e5e8ff5e`)
- C6: `kaggle-results/seed42_mqar-C6_overwrite-C6_20261114T115438Z_7b86f275` (`a7722bc9aac95b78804e79c79638f1b3e5e8ff5e`)

All evaluation provenance records report no source drift. Each family uses the same generated held-out hash across conditions; this verifies the reported generator identity, not model success or the absence of all dataset shortcuts.
