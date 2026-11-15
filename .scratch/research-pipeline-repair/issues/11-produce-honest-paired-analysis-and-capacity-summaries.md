# 11: Produce analysis that handles missing runs and uncertainty correctly

**What to build:** Paper-facing summaries describe the prespecified paired seed cohort honestly, including failed runs and non-achieved capacity, instead of silently summarizing whichever successful records are available.

**Blocked by:** 01: Protect legacy results and introduce corrected experiment identities; 07: Make retrieval evaluation causal and its metrics explicit.

**Status:** implementation committed; analysis regressions pending test-seam approval

- [ ] Primary granularity-by-initialization contrasts, secondary comparisons, metric weighting, uncertainty method and multiple-comparison treatment are prespecified and recorded.
- [ ] Training seeds are the unit of model replication; example-level evaluation precision is reported separately.
- [ ] Incomplete or failed records remain visible but cannot enter eligible primary aggregate estimates. Missing required cells or prespecified seeds make the primary paired result incomplete.
- [ ] Legacy and corrected protocols cannot be combined in one primary contrast; inspected seed-42 outcomes remain exploratory.
- [ ] Threshold failures remain explicit in every seed table. Conditional mean capacity among successful seeds cannot be labeled an unconditional capacity estimate.
- [ ] Known fixture outcomes verify interaction signs, paired counts, uncertainty units, duplicates/missing-cell handling and failure/capacity reporting through existing analysis entry points.
- [ ] Compact reports retain traceable artifact references and clearly separate validation selection from final held-out inference.
