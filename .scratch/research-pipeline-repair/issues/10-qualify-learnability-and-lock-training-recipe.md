# 10: Qualify learnability and lock a feasible training recipe

**What to build:** Research training is authorized by evidence that both corrected endpoints can learn valid tasks under an affordable validation-selected recipe, rather than by a successful plumbing or capacity smoke test.

**Blocked by:** 04: Establish a defensible architecture baseline and parameter accounting; 07: Make retrieval evaluation causal and its metrics explicit; 08: Make calibration and gate-sharing conclusions identifiable; 09: Verify mixed precision and accepted optimizer-step accounting.

**Status:** implementation committed; E0/E1 execution pending compute authorization

- [ ] Qualification freezes the easy-task criterion before execution, starting from at least 90% answer accuracy at two consecutive validations for both endpoints; any revision is prospective and documented.
- [ ] Recipe selection uses validation only and comparable tuning effort, with deterministic selection and explicit failures. Final held-out outcomes cannot choose the recipe.
- [ ] Curves versus accepted steps, effective tokens and wall time establish learnability and guide a feasible fixed budget; convergence or capacity claims require their own evidence.
- [ ] A protocol lock records mathematical/model/generator versions, initialization package, recipe, budgets, split/case identities, evaluation grids, metric weighting and stopping/exclusion rules.
- [ ] Offline success, failure and missing-lock cases demonstrate that production entry points enforce qualification rather than silently bypassing it.
- [ ] GPU qualification is an explicit bounded operation with logged hardware, numerical tolerances and complete artifacts; implementing this ticket does not automatically launch a research sweep.
- [ ] Reports distinguish easy-task learnability, final-difficulty qualification, CPU plumbing and GPU capacity outcomes.
