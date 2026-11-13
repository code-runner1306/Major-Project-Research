# Legacy artifact and protocol migration policy

The completed exploratory runs are legacy evidence. Their manifests,
checkpoints, logs, and metrics are retained in place and are never relabeled
as corrected-protocol results. A corrected run uses `corrected-v1`, a fresh
experiment identity, and a fresh output root. The runner refuses to write into
a legacy directory or into a non-empty directory without a compatible manifest.

Only operational changes are compatible with an existing artifact, and only
when they do not change the scientific protocol: log destination, reporting
format, and the number of idle workers used to schedule independent tasks.
Changes to model, recurrence, generator, data split, calibration, optimizer,
budget, evaluation, precision, or source content require a new protocol
identity and output root. Evaluation-only source drift must be explicitly
requested and does not authorize training or artifact mutation.

## VM checkpoint verification

For a checkpoint copied from the VM, retain the original run directory and
manifest. Verify the copied file before analysis:

1. Compute a SHA-256 digest (`Get-FileHash -Algorithm SHA256`) and compare it
   with `metrics.json.final_checkpoint.sha256`.
2. Load the checkpoint through `artifacts.validate_checkpoint` and verify the
   recorded accepted optimizer steps and processed-token total.
3. Confirm the manifest fingerprint, protocol version, source content hash,
   lifecycle status, and required metrics all match the run being analyzed.

If the VM retained only metrics or a checkpoint without its manifest and
lifecycle record, treat it as a non-resumable external result. Do not copy it
over a local run directory or count it as a complete locally validated run.

Corrected manifests also retain per-source content hashes and, when the
tracked source was dirty, the corresponding `git diff --binary` patch so the
source state can be reconstructed rather than identified only by a dirty bit.
