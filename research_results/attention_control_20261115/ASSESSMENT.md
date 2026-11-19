# Attention-control assessment: 15 November 2026

## Decision

Do not launch the full C1/C3/C4/C6 multi-seed experiment yet. Freeze these results as exploratory evidence, not a final task qualification. Clean overwrite data is verified against the implemented audits; successful clean GDN learning is not established.

## Provenance and validity

Source revision: 658852d265a91e2be2121105a2e858965f16f6fe.
Remote branches:
- diagnostic/mqar/seed42_attention_control_mqar_20261115T065824Z
- diagnostic/overwrite/seed42_attention_control_overwrite_20261115T070019Z

Both sessions complete; all four workers accepted 2,000 optimizer steps, 8,000 microbatches, no rejected AMP attempts. Architectures have identical train/development/confirmation manifests within each family. Both development and confirmation datasets pass independent label oracles, query-key counterfactual checks, complete baseline checks, and the implemented shortcut threshold. These audits cover specified shortcuts, not every conceivable shortcut.

10,000 shuffled training examples, four keys, sequence length 128, vocabulary 8,192, one rewrite for overwrite, LR peak 0.001, effective batch32, development128 every200 steps, confirmation512 once. Actual GDN parameters3,329,784; attention3,174,560 with learned positions. One seed, one LR, no capacity matching.

## Fresh confirmation: answer-free accuracy

| Task | GDN | Attention |
|---|---:|---:|
| MQAR | 2/2048 = 0.098% | 4/2048 = 0.195% |
| Overwrite | 25/1024 = 2.441% | 649/1024 = 63.379% |

MQAR teacher-forced: GDN2/2048, attention1/2048. Thus answer-history errors alone cannot explain MQAR failure. Both score below tested copy baselines (~1.7-1.8% confirmation); no useful MQAR generalization established. Final recent-training scores12.30% GDN,1.37% attention.

Overwrite attention: latest-value98.44%, unrelated-retention28.32%, stale-value error0%; first query76.56%, later50.20%. Teacher-forced aggregate equals answer-free. Strong update performance but incomplete retention. Strongest tested key-free aggregate baseline20.31%; attention exceeds it substantially, but aggregate improvement does not establish general keyed retention.
Overwrite GDN: latest-value4.30%, unrelated-retention0.59%; teacher-forced aggregate4.69%. Recent-training28.52% vs development3.91% and confirmation2.44% indicates limited generalization. Earlier88.28% was a different two-key distribution with a shortcut; do not use it to qualify this task.

All GDN steps and overwrite attention steps clipped at norm1; average pre-clip norms MQAR-GDN1.88, overwrite-GDN2.87, overwrite-attention3.77. Clipping frequency is an observation, not evidence of a clipping defect (attention learns while also always clipped).

## Actual time

| Task | GDN including evaluations | Attention including evaluations | Whole session |
|---|---:|---:|---:|
| MQAR | 2h31m51s | 4m05s | 2h32m11s |
| Overwrite | 2h33m33s | 3m55s | 2h33m47s |

Mean optimizer step ~3.87/3.92s GDN vs0.098/0.094s attention. The attention worker completed its full budget; fast time is not premature exit. GDN reference recurrence is the main wall-time cost. Notebook installation/startup is outside coordinator timing.

## Next decision

The main granularity x initialization x task comparison is not informative when both MQAR models are at floor and scalar GDN also fails overwrite. The architecture difference on overwrite is diagnostic, not proof of a GDN implementation bug or incapacity. Position mechanism and parameter counts differ; optimization, data diversity, vocabulary burden and implementation remain competing explanations.

Recommended next bounded experiment: keep four-key V4 data, depth/width, 10k examples,2k steps, LR0.001 and splits fixed; reduce vocabulary to256 for both architectures and both tasks, regenerate and audit all data, and run matched pairs. This tests whether the high-cardinality symbol/output burden is blocking learnability. It changes embedding parameter count, so describe it as a separate diagnostic; it is not guaranteed to solve the problem. Use fresh development and untouched confirmation data; publish exact manifests. Current timing suggests roughly5-6h for two consecutive pairs on two T4s if step costs remain comparable, with existing cutoff retained. No new GPU run has been launched.

Qualify a setting only with strong fresh-confirmation performance in both overall and relevant subgroups (latest and unrelated for overwrite; first and later for MQAR), using a predeclared operational target such as80% each, plus audit passes. Then freeze all settings and compare C1/C3/C4/C6 on paired fresh seeds43-45 with fresh confirmation data. If attention learns but GDN remains poor in the easier setting, prioritize GDN optimization/recurrence diagnosis before spending a full factorial budget. If MQAR cannot be qualified by the deadline, narrow claims to a qualified overwrite study or report diagnostic failures transparently; do not claim retention-versus-overwrite granularity evidence.
