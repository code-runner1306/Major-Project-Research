# 05: Build an overwrite task that requires keyed retrieval

**What to build:** Overwrite examples require looking up a queried key's latest value and retaining unrelated associations, rather than returning values from predictable final memory positions.

**Blocked by:** 01: Protect legacy results and introduce corrected experiment identities.

**Status:** implementation committed; dataset regressions pending test-seam approval

- [ ] A versioned generator randomizes target identity, relevant write locations and query ordering, with unrelated writes allowed after the target's latest write.
- [ ] An independent last-write-wins oracle derives targets from the causal memory prefix and solves every feasible case in the frozen grid and bounded randomized coverage.
- [ ] Query-key counterfactuals preserve memory while selecting the appropriate latest value; stale and unrelated targets remain distinguishable without fixed position or query-order cues.
- [ ] The audit's final-two-values baseline no longer solves the task. Recency, position, first/latest-value copying and no-memory baselines are measured against prespecified per-cell shortcut criteria and documented chance expectations.
- [ ] Actual query distance and distractor load control content before prediction. Nominal sequence length cannot increase solely through a suffix after all scored answers.
- [ ] Feasibility and balancing are audited jointly over key count, rewrites, actual distance, length and distractors; infeasible cases fail rather than being relabeled.
- [ ] Deterministic cases and content-level split checks are persisted; paired conditions receive identical cases. Padding, intentional filler and effective/supervised token counts are distinguishable.
- [ ] Dataset-facing checks and a small offline worker demonstrate valid batches, targets, masks, split provenance and new generator identity without modifying legacy data.
