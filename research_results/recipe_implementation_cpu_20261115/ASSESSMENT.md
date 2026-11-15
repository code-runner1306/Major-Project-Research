# Local implementation qualification

CPU-only evidence. These results do not qualify a CUDA backend, GPU concurrency,
MQAR/overwrite learnability, or the final training budget.

34 targeted tests passed, including real recurrence output/state/gradient
agreement, zero gates, cache continuation, five-step actual AdamW trajectories,
scored-token loss/all parameter gradients, and existing reference/cache checks.

The full orchestration was exercised with2/3/4 concurrent CPU processes in
.scratch/recipe_pilot_cpu_all_workers before scored-loss integration. The final
combined pilot completed its three two-worker implementation waves in
.scratch/recipe_pilot_cpu_scored. Both are implementation checks, not GPU speed
or capacity measurements. Startup and evaluation dominate these two-step CPU
smokes; do not extrapolate their phase throughput to Kaggle.

The saved numerical_qualification.json records stricter per-input recurrence
comparisons and the actual128-wide model trajectory for both gate types. The
pilot source revision is the parent commit because these checks preceded the
commit of the candidate files; the candidate source hashes identify tested
implementations. Future GPU results must identify their own committed source.

Next: run run_kaggle_recipe_pilot.sh on T4x2. Browser control could not determine
Brave's current URL reliably, so no Kaggle notebook was created or run. Existing
Kaggle sessions were not interrupted. The final published-recipe training
harness, real-task learnability and end-to-end12-hour budget remain outstanding.
