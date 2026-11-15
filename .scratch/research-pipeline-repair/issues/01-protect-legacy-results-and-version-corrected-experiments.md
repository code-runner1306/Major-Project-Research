# 01: Protect legacy results and introduce corrected experiment identities

**What to build:** A researcher can preserve the exploratory pilot and start a corrected experiment without accidentally mixing artifacts from different scientific methods. Reuse decisions happen before stored task configurations can be overwritten.

**Blocked by:** None (can start immediately).

**Status:** implementation committed; regression verification pending test-seam approval

- [ ] Legacy runs, metrics, logs, manifests and available checkpoints remain unchanged; an inventory identifies VM-only artifacts and records checkpoint verification instructions.
- [ ] Corrected recurrence, model and generator versions produce a new experiment identity and output root, with explicit separation from legacy results.
- [ ] Identity covers the full effective model, data, calibration, training and evaluation protocol, relevant source-content hashes, and a reproducible source snapshot or patch when dirty.
- [ ] Reuse validates scientific compatibility before writing task configurations; changing any scientific setting rejects reuse or selects a fresh identity.
- [ ] Complete runs require compatible identity, lifecycle, checkpoint hash, accepted-step/token counts and mandatory evaluation artifacts. Metrics-only archives cannot masquerade as resumable training.
- [ ] Harmless operational changes have a documented compatibility policy; training/evaluation concurrency is recorded explicitly.
- [ ] Interrupted training fails with an honest recovery explanation unless complete optimizer, scheduler, scaler, RNG and data-position restoration is supported and verified.
- [ ] Small worker/launcher checks demonstrate compatible skipping, incompatible rejection and legacy preservation. Research training remains gated; no seed sweep starts as a side effect.
- [ ] Research guidance identifies legacy results as exploratory custom-method evidence and no longer recommends expanding that protocol as the next research step.
