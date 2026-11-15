# 03: Implement key-axis channel decay and verify grouped endpoints

**What to build:** Scalar, grouped and channel-wise arms expose the intended number of independent key-axis decay controls while using the same corrected memory-update contract.

**Blocked by:** 02: Implement and independently verify the scalar Gated DeltaNet update.

**Status:** implementation committed; independent numerical and grouped-endpoint verification pending test-seam approval

- [ ] Channel-wise decay follows the pinned KDA equation on the key axis in the declared state orientation, before evaluating the residual.
- [ ] Nonsymmetric states, nonuniform channel gates and basis-vector probes distinguish correct key-axis decay from the legacy value-axis behavior.
- [ ] Forward output, state and gradients match an independent channel-wise oracle, including nonzero initial states and multiple time steps.
- [ ] Identical mapped gate functions make scalar, grouped and channel-wise endpoints agree where mathematically equivalent; group broadcasting preserves channel ordering.
- [ ] Cached and split-sequence execution agree with full-sequence execution for each arm.
- [ ] Small paired CPU worker runs expose the new semantics in artifacts. Every enabled optimized backend qualifies against the independent corrected oracle or remains unavailable.
