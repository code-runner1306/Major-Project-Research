# 09: Verify mixed precision and accepted optimizer-step accounting

**What to build:** A training run has an exact accepted-update budget and evidence that its chosen precision is suitable for the corrected recurrent dynamics, including recovery from occasional loss-scale overflow.

**Blocked by:** 03: Implement key-axis channel decay and verify grouped endpoints.

**Status:** implementation committed; numerical precision comparison and accounting regressions pending test-seam approval

- [ ] A bounded FP32 reference runs with autocast disabled; near-one retention, nonzero state and meaningful sequence lengths exercise output, state and gradient sensitivity.
- [ ] Precision-specific tolerances and failure criteria are justified and fixed before qualification; casts inside an active autocast region are not treated as proof of FP32 matmuls.
- [ ] Injected recoverable overflows retry the same batches after reducing scale, restoring relevant RNG state and clearing rejected gradients.
- [ ] Only accepted updates advance optimizer, scheduler and budget/data counters. Equivalent recoverable and overflow-free control runs agree under the test's declared precision contract.
- [ ] Persistent nonfinite loss/gradients fail loudly with retained diagnostics; retries are bounded and visible in artifacts.
- [ ] Worker-level checks verify step, microbatch and effective/supervised token accounting across accumulation and retry boundaries.
- [ ] CPU correctness, available precision checks and unavailable GPU-specific qualification are reported distinctly; GPU checks never fake-pass on a CPU host.
