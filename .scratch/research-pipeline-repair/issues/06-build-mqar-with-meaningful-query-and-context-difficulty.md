# 06: Build an MQAR task with meaningful query and context difficulty

**What to build:** MQAR asks for arbitrary stored associations and varies actual retrieval demand, so first-query recency and suffix padding cannot masquerade as associative recall or long-context behavior.

**Blocked by:** 01: Protect legacy results and introduce corrected experiment identities.

**Status:** implementation committed; dataset regressions pending test-seam approval

- [ ] A versioned generator chooses query order independently of memory order; the first query is not guaranteed to reference the last memory association.
- [ ] A causal keyed-lookup oracle solves the complete feasible grid and bounded randomized cases without trusting target-placement metadata.
- [ ] Query-key counterfactuals hold memory fixed and change the target appropriately. First-versus-later query and key-free baseline diagnostics expose positional advantages.
- [ ] The audit's last-value first-query shortcut no longer receives a guaranteed correct answer. Prespecified shortcut criteria and chance expectations are recorded before training.
- [ ] Distance and context sweeps change meaningful content before scored queries; per-answer realized distances, load and distractors match reported difficulty.
- [ ] Joint grids are balanced where feasible and impossible combinations fail explicitly rather than being silently adjusted.
- [ ] Deterministic train/validation/test case identities, content-level split checks, paired case equality and padding/effective-token accounting are auditable.
- [ ] Dataset-facing checks and a small offline worker verify target alignment, masking, genuine scored-prefix changes and new generator identity while preserving legacy examples.
