# 12: Separate worker capacity testing from isolated efficiency measurement

**What to build:** The researcher can select a safe, useful worker count for the corrected workload and obtain publishable inference measurements without interference from concurrent model jobs.

**Blocked by:** 04: Establish a defensible architecture baseline and parameter accounting; 05: Build an overwrite task that requires keyed retrieval; 06: Build an MQAR task with meaningful query and context difficulty; 09: Verify mixed precision and accepted optimizer-step accounting.

**Status:** implementation committed; GPU capacity and isolated-efficiency qualification pending hardware authorization

- [ ] Training parallelism and inference measurement concurrency are separately configured and recorded; publishable inference benchmarks require isolated GPU access.
- [ ] A representative bounded pilot compares sequential and two-worker throughput, per-run slowdown, peak VRAM and visible contention under the same corrected workload.
- [ ] Worker qualification covers the relevant families and gate/initialization conditions, verifies current hardware/protocol identity and reports its actual limit without assuming a twofold speedup.
- [ ] Capacity-pilot evaluation remains labeled operational evidence; CPU smoke, independent numerical correctness, GPU precision, capacity and research qualification cannot authorize one another implicitly.
- [ ] Prefill, decode-only and end-to-end energy/latency regions have aligned boundaries and accurate names. Unavailable power/energy remains unavailable.
- [ ] Peak memory is reset/measured per phase and distinguishes allocated/reserved storage, recurrent state, convolution cache and attention KV cache.
- [ ] Backend, precision, hardware, workload, warmup, repetitions and concurrency accompany reported measurements; reference-loop timings are not generalized into optimized architectural claims.
- [ ] Controlled provider/clock checks verify boundaries, and offline launcher checks verify isolated benchmarking within a parallel-training workflow. Hardware qualification runs only when explicitly requested in an available GPU environment.
