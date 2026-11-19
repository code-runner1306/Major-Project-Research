# V3 exploratory run review

Evidence: `output/granularity_exp_2/runs`, pulled from maink commit 57c6f12.
There are 21 completed training metrics: two complete eight-model seeds (42,43),
and five models for seed44 (MQAR C1,C3,C4; overwrite C1,C3). No complete third seed.
All 21 completed 1000 accepted optimizer steps. Some workers rejected 1..5 attempts.
Settings are V3 generator, LR .003, effective batch32, sequence lengths128/256/512,
50k training examples, 512 validation examples, legacy six-layer model. These are
not the new published-recipe experiments. The nominal10M model has approximately
3.33M scalar /3.48M channel parameters.

## Matched comparison: seeds42 and43 only

| Task | Condition | Mean final validation loss (step1000) | Mean saved answer accuracy (step650) |
|---|---|---:|---:|
| MQAR | C1 scalar default |8.1942|0.0524%|
| MQAR | C3 channel default |8.3286|0%|
| MQAR | C4 scalar calibrated |4.8914|8.8223%|
| MQAR | C6 channel calibrated |5.7421|5.2075%|
| Overwrite | C1 scalar default |7.4793|0.0977%|
| Overwrite | C3 channel default |4.9487|9.1309%|
| Overwrite | C4 scalar calibrated |4.6343|14.4043%|
| Overwrite | C6 channel calibrated |3.4399|25.2441%|

Paired data identities match across all four conditions for each complete seed
and task. Calibration improves loss in both seeds in both gate layouts/tasks.
The calibrated scalar condition has lower MQAR loss than calibrated channel;
the ranking reverses on overwrite. This is a hypothesis-generating task-dependent
pattern, not evidence of a confirmed interaction with only two complete seeds.

## Limits and usable conclusions

Only step300/650 answer accuracy is saved. Final step1000 artifacts contain loss,
not final retrieval accuracy. Lifecycle is trained, not a completed held-out
evaluation. Do not call the table final held-out accuracy. Source revision in the
manifest is unavailable; preserve original artifacts/provenance limitations.

V3 overwrite always positions the latest target write and one unrelated write as
the final two memory writes, then asks for the target and that unrelated key in
that order. A positional final-two-values rule can answer both without keyed
memory. MQAR V3 similarly puts the first queried association last in the initial
memory. Thus improvements cannot demonstrate the intended mechanism. The V4 and
published MQAR task changes address task construction, but are different settings;
do not pool their scores or claim direct causal recipe superiority from them.

Calibration is a package treatment: changed alpha biases/retention targets and
initialization; it is not isolated proof of memory lifetime. Derived alpha
half-lives are diagnostics, not measured full-state memory lifetimes. The model
also differs from the current published-style recipe.

Retain these runs as a clearly labeled exploratory appendix/pilot showing strong
initialization sensitivity under a fixed legacy budget, and as motivation to test
the task-dependent gating ranking prospectively. Do not use them as the main
retrieval/overwrite claim, combine incomplete seed44 selectively, or spend more
training budget extending V3. Finish clean-task learnability, then test calibrated
and default conditions with a common qualified budget and untouched confirmation.
If original checkpoints are available, final answer-free evaluation and per-query
shortcut comparisons are possible without retraining, but cannot repair V3's
structural task confound.
