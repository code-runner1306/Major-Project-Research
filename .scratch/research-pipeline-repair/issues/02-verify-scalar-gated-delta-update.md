# 02: Implement and independently verify the scalar Gated DeltaNet update

**What to build:** The corrected scalar model trains and decodes using the published gated delta update, with mathematical correctness established independently of the implementation being tested.

**Blocked by:** 01: Protect legacy results and introduce corrected experiment identities.

**Status:** implementation committed; independent numerical and gradient verification pending test-seam approval

- [ ] The intended published equation and official reference revision are pinned, and state orientation is documented consistently.
- [ ] Decay is applied before evaluating the delta residual; the audit counterexample produces the published state of 0.75 rather than the legacy 0.5.
- [ ] Independent oracle cases cover nonzero initial state, multiple time steps, nonsymmetric states and limiting gate settings; the oracle does not call the corrected implementation.
- [ ] Output, final state and gradients match the independent oracle under justified tolerances.
- [ ] Full-sequence, split-sequence, prefill and cached continuation agree for the corrected scalar model through its existing public interfaces.
- [ ] Semantic versions and backend qualification identities change; an optimized recurrence or decode path is disabled until it independently qualifies for the corrected scalar semantics.
- [ ] A small CPU worker run emits corrected-version artifacts, and attempts to reuse legacy artifacts fail without modifying them.
