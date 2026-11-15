# 07: Make retrieval evaluation causal and its metrics explicit

**What to build:** A completed evaluation tells the researcher exactly what was predicted, under what history, with which weighting and difficulty, without conflating individual query correctness with complete example success.

**Blocked by:** 05: Build an overwrite task that requires keyed retrieval; 06: Build an MQAR task with meaningful query and context difficulty.

**Status:** implementation committed; causal and metric regressions pending test-seam approval

- [ ] Freeze an answer-free primary query format for clean associative-recall measurement. Any retained oracle-history evaluation is separately labeled; generation claims have a separate autoregressive rollout.
- [ ] Prediction occurs before its own answer token. Future-suffix perturbations cannot change earlier scored predictions, and query-key counterfactuals select the expected target.
- [ ] Reports separately expose query accuracy, all-queries-correct example accuracy and prespecified macro/micro weighting.
- [ ] Latest-value accuracy is the primary overwrite endpoint; unrelated retention, stale-value errors and other errors have explicit denominators.
- [ ] Joint actual-distance/load breakdowns and first-versus-later query diagnostics accompany aggregate scores; requested and realized distances cannot be confused.
- [ ] Validation loss weighting and effective/supervised token counts are documented and stable across paired conditions.
- [ ] An offline evaluation workflow produces auditable example/case references and compact summaries with coherent metric units and split provenance.
