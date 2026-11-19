# Published-recipe T4 development results

Fetched both diagnostic/published-recipe/{mqar,overwrite}/development branches
dated 20261116T002815Z and 20261116T002609Z. Evidence is preserved under
research_results/published_recipe_development_20261116. Both sessions used source
f12e991, seed42, chunked FP32, 100k training examples, effective batch512,
microbatch32, 512 development examples, LR .0021544346900318843, epoch cosine
schedule, 64 epochs/12544 accepted steps. Scalar and channel data manifests match
within each task; data audits pass and exact train/development overlap is zero.
Both workers exited successfully in each session. Confirmation was not evaluated.

| Task | Variant | Final correct answers | Accuracy | Latest | Unrelated retention | Stale error |
|---|---|---:|---:|---:|---:|---:|
| MQAR | scalar |2048/2048|100%|n/a|n/a|n/a|
| MQAR | channel |2047/2048|99.9512%|n/a|n/a|n/a|
| overwrite | scalar |1019/1024|99.5117%|100%|99.0234%|0%|
| overwrite | channel |1022/1024|99.8047%|100%|99.6094%|0%|

All scores are final epoch64 answer-free DEVELOPMENT metrics, not untouched
confirmation or multi-seed results. MQAR contains no gold answer tokens in inputs;
overwrite answer-free erases previous gold answer tokens. Teacher-forced overwrite
accuracy is99.7070% for both models, so final high scores do not require gold
answer history. Answer-free is the primary protocol, not autoregressive feedback.

## Learning behavior

MQAR scalar exceeds99% from epoch4 onward; channel from epoch3 onward. Both first
reach100% at epoch6. This easy condition has a ceiling and cannot resolve strong
final gate superiority. Overwrite latest-value accuracy is already99.8..100% at
epoch1, while unrelated retention is29.9..30.5%. The later gains therefore concern
preserving unrelated associations, a central question for this study.

Overwrite channel retention reaches98.05% at epoch5/980steps and99.61% at
epoch6/1176steps. Scalar retention is76.37% and84.18% at those epochs; it first
exceeds95% at epoch15/2940steps. This is a promising sample-efficiency observation,
not a replicated causal finding. Models have different parameter counts and a
single seed. Later curves occasionally fluctuate. Do not choose each model's best
checkpoint or declare significance from a few answers on the same development set.

## Measured runtime

MQAR scalar3.588h/channel3.896h; coordinator3h53m48s. Overwrite scalar3.641h/
channel3.955h; coordinator3h57m22s. Mean step times: MQAR1.026/1.114s;
overwrite1.036/1.126s. All64 validations total20.69/21.36s MQAR and41.10/41.65s
overwrite. Checkpoint writing totals2.54..2.69s per model. Two parallel workers
fit the12h limit with substantial headroom.

## Decision

Development learnability passes the prespecified80% primary and subgroup screen
for all four models. The source-pinned recipe is a viable development baseline;
the previous inability to learn is not inherent to these tasks/models. Several
factors changed jointly, so these runs cannot isolate whether training duration,
generator, architecture, normalization, or other recipe changes solved the old
failure.

Keep64epochs as the tested reference budget. A shorter schedule must be tested
because changing cosine T_max changes the whole trajectory; an early point on the
64epoch curve is not an exact prediction for an8/16epoch schedule. No need to
extend training. Next evaluate final checkpoints on a fresh, frozen confirmation
set using the same protocol, without tuning on it. Checkpoints were excluded from
GitHub and must be retrieved from Kaggle outputs. Preserve them promptly.

Before the main factorial, implement and qualify calibrated initialization for
the new decay parameterization (old sigmoid-gate calibration cannot be reused
unchanged). Freeze the common budget, task grid, evaluation and seed list. Then
test default/calibrated x scalar/channel across seeds and harder conditions to
avoid the current MQAR ceiling. Predeclare learning-curve/sample-efficiency and
unrelated-retention outcomes, not just saturated final accuracy. Do not claim the
full calibration study is ready solely from these default-initialization pilots.
