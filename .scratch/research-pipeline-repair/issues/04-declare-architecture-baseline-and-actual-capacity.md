# 04: Establish a defensible architecture baseline and parameter accounting

**What to build:** Every experiment identifies the architecture it actually runs, its reference-backed default memory behavior, and its actual trainable capacity, allowing a reviewer to understand precisely what varies between arms.

**Blocked by:** 03: Implement key-axis channel decay and verify grouped endpoints.

**Status:** implementation committed; parameter-count regressions pending test-seam approval

- [ ] The full token-mixer contract is reviewed against pinned references, including decay parameterization, convolution activations, normalization, output gating and hybrid layout.
- [ ] Intentional simplifications are explicitly declared; named-architecture reproduction claims are limited to what is implemented and verified.
- [ ] Default memory initialization is defensible and documented. A zero-bias sigmoid default is not presented as a faithful published default without evidence.
- [ ] Gate-function family and non-granularity settings are common across paired arms; corresponding non-gate backbone tensors initialize identically for a paired seed.
- [ ] Default and calibrated arms have explicitly accounted trainable biases and parameter differences.
- [ ] Artifacts and reports display actual trainable counts alongside nominal backbone labels and use consistent definitions of gate overhead, tied weights and recurrent state memory.
- [ ] Same-backbone and exact capacity-matched comparisons remain distinct. Claims independent of capacity require an appropriate useful-parameter control; unsupported exact matches fail explicitly rather than adding dummy parameters.
- [ ] Paired small-model worker checks verify construction, parameter accounting and the declared architectural differences end to end.
