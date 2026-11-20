# Brief: Attention control (appendix)

Status: `results` (no attention checkpoint is saved in the repo; Kaggle checkpoints were not retained). The try-it slide shows `<Unavailable>` plus saved training curves.

## Question
Does a standard attention model do well on the same four-key MQAR and overwrite tests, as a comparison point for the recurrent (Gated DeltaNet) models?

## Sources and verified facts

### Same protocol: scalar Gated DeltaNet vs attention
- Source: `research_results/attention_control_20261115/{mqar,overwrite}/<family>/{gdn,attention}/summary.json`, `settings.json`, `validation_history.json`; audits in `<family>/<family>_audit.json`; design in `ATTENTION_DIAGNOSTIC.md`; generator in `attention_control_diagnostic.py` (`datasets`, `AttentionControl`, `compare`).
- Budget (from `manifests` in settings): 10,000 training examples, 4 keys, sequence length 128, distance 8, 1 overwrite (overwrite family), vocabulary 8,192, 512 held-out examples, 2,000 steps, learning rate 0.001.
- Parameters (settings.json `model_parameters`): scalar GDN 3,329,784; attention 3,174,560 (learned absolute positions). Backbone: 6 layers, width 160, 4 heads, head width 40 (both).
- Held-out answer-free (summary.json `confirmation.answer_free`):
  - MQAR: GDN 2/2048 (0.098%), attention 4/2048 (0.195%).
  - Overwrite: GDN 25/1024 (2.44%), attention 649/1024 (63.38%).
  - Overwrite detail, GDN vs attention: latest 4.30% vs 98.44%; unrelated retention 0.59% vs 28.32%; first query 3.71% vs 76.56%; later query 1.17% vs 50.20%; stale-value error 0.39% vs 0.00%.
- Key-free baselines (audit `confirmation.baselines`): MQAR best 1.76% (`last_value`); overwrite best 20.31% (`final_two_values`). Keyed last-write oracle 100% both.
- Development curves: `validation_history.json` `development_answer_accuracy` by `step` (every 200 steps).
- Clipping: all steps clipped in both GDN runs and overwrite attention (`clipped_step_fraction`); assessment says this is an observation, not a defect.
- Wall time (summary `wall_seconds`): attention about 4 min; GDN about 2.5 h (overwrite 9,213 s; MQAR 9,111 s). ASSESSMENT.md time table says 2h31m / 2h33m including evaluations.

### Scalar vs channel-wise on the same tests (different protocol)
- Source: `research_results/kaggle_seed42_20261114/C1` (scalar, default) and `C3` (channel-wise, default), `{mqar,overwrite}/<condition>/seed_42/metrics.json` (`retrieval`, `parameter.trainable_parameter_count`, `final_validation.loss`); `ASSESSMENT.md`.
- Budget: 1,000 steps, context length 512, 512 held-out examples, seed 42, learning rate 0.001. Not the same budget as the attention control (2,000 steps, length 128). Attention was NOT run on this protocol; channel-wise was NOT run on the attention protocol.
- Held-out exact-query accuracy: MQAR scalar 0/4772 (0%), channel-wise 1/4772 (0.02%); overwrite scalar 0/1024 (0%), channel-wise 0/1024 (0%).
- Parameters: scalar 3,329,784; channel-wise 3,480,480 (not capacity-matched).
- Final validation loss: MQAR 8.3287 vs 8.3287; overwrite 8.4735 vs 8.4827.
- Verdict in the saved report: floor results, no useful retrieval signal for either.

### Literature context (not one of our tests)
- `decay_design_results.md` section 4: channel-wise minus scalar validation loss on TinyStories, mean gap -0.0402 (10M) and -0.0389 (100M), all six seed pairs negative. Parsed from table rows 56 and 57. The file's authors are the paper's numbers, not our runs.

### Conflicts and notes
- `ASSESSMENT.md` (15 Nov) gives the same confirmation numbers as the summary.json files. Checked: they agree.
- ASSESSMENT.md says the Kaggle set had a "different two-key distribution" earlier; not used.
- The attention-control source revision (`658852d...`) and the Kaggle revision (`a7722bc...`) differ. Note in More detail only if asked.
- The worktree copy under `.claude/worktrees/` is not used; only the repo-root `research_results/` copy is read.

### Inside visual (schematic, no weights saved)
- Real example from the held-out generator (`retrieval.RetrievalDataset`, task overwrite, seed 42 + 300000, index 0). Tokens and roles come from `metadata` (`write_positions`, `all_write_positions`, `query_positions`).
- Recurrent memory size per layer: 4 heads x 40 x 40 = 6,400 numbers, fixed. Attention cache per layer: 2 x 4 x 40 = 320 numbers per token (grows with length). Derived from settings.json backbone.

### Not verified / left out
- No attention checkpoint exists locally, so no real attention weights and no live try-it.
- No channel-wise run on the attention-control protocol exists.
- The glossary has no term for "attention" or "attention control" (needed in report).
