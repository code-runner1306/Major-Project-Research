# Gate diagnostics after the three-seed CPU confirmation

Status: complete,17November2026. All24trained final16epoch checkpoints were
probed on CPU, after the fresh confirmation evaluation completed. Checkpoint
hashes agree with confirmation. No weights were trained, refitted, or modified.
Recorded probe time:3.39minutes. Six source/selection/probe regression tests pass;
a separate shared-mean intervention test also passes.

## Inputs and limits

The first512rows of each EXACT full100k training pool were cached during
confirmation's overlap audit. Prefix equality matters: MQAR draws all keys, then
all values, then all gaps, so generating a smaller pool with the same seed would
not recreate that prefix. First256rows reproduce calibration inputs; next256rows
are independent of calibration fitting but were used for model training. Both
original training conditioning and overwrite answer-free conditioning are probed.
Confirmation examples are not used for gate diagnostics or fitting.

The initial model replay did not match the saved base-parameter fingerprints for
any of the24models. The cause is not established. We did NOT bypass that check
and call the replay exact. Initial full-model probes and unverified parameter
drifts were omitted. Instead we use original recorded calibration means/std/
offsets where available, and directly load the actual trained checkpoints for
all final probes. The original training-run hashes still match within paired
conditions; this limitation concerns local initial replay, not trained checkpoint
availability or their exact evaluation. Future runs should preserve an actual
initial checkpoint if bitwise initial replay is required.

Outputs retain240final layer/corpus measurements,12original initial calibration
records, and48exact calibrated-offset drift records. Large-tensor quantiles use
a fixed16384value sample; means/std/extrema/fractions use every probe value.
Correlation statistics use a fixed1024token sample. These are post-hoc
descriptive diagnostics, not preregistered primary outcomes or causal explanations
of the training trajectories.

## 1. Matched starting means did not remain matched

Example: layer2, on the original256calibration examples, averaged across3seeds:

| Task | Recorded starting mean, both gates | Trained scalar mean | Trained channel mean |
|---|---:|---:|---:|
| MQAR |0.978572|0.999513|0.946078|
| Overwrite |0.978572|0.989172|0.926394|

These are decay multipliers, not answer accuracies. Closer to1means less direct
decay in this operation; it is not a full-model memory lifetime. Every layer is
available in condition_layer_means.json and gates.csv. The comparison uses the
same initial calibration-input positions and recorded initial measurements,
not a freshly approximated initial model.

Calibration initialized the offsets; it did not constrain means throughout
training. Exact saved-offset-to-final RMS changes average0.258for MQAR scalar,
0.153for MQAR channel,0.171for overwrite scalar and0.118for overwrite channel.
Averages are across layers and seeds, taking the per-parameter RMS across its
coordinates first. Alpha changes cannot be attributed to offsets alone: rates,
projections and earlier-layer representations also trained.

## 2. Trained channel gates remain heterogeneous

On the calibration-input probe, mean within-token channel standard deviation is
approximately0.111for calibrated MQAR and0.103for calibrated overwrite, averaged
across layers/seeds. Scalar dispersion across channels is identically zero by
design. The channel outputs have not collapsed to the same multiplier.

Using recorded initial means/std for calibrated models, mean per-layer total
alpha standard deviation changes as follows (same input examples; then averaged
over layer/seed combinations):

| Task | Gate type | Recorded initial STD | Trained STD |
|---|---|---:|---:|
| MQAR | Scalar |0.00213|0.00144|
| MQAR | Channel |0.00261|0.11469|
| Overwrite | Scalar |0.00247|0.01965|
| Overwrite | Channel |0.00282|0.10899|

Thus initial per-gate mean matching did not prevent channel differentiation
during optimization. This does not establish that differentiation caused a
learning-speed improvement.

## 3. Response sensitivity differs after training

The local analytic sensitivity is
`abs(d_alpha/d_raw) = alpha * A * sigmoid(raw + dt_bias)`;
the offset has the same local derivative magnitude. It matches autograd in tests.
For calibrated MQAR layer2, mean sensitivity is0.000486scalar versus0.039707
channel. This is not the loss gradient and does not establish which model is
easier to optimize: it omits upstream derivatives, state interactions and readout.
We cannot compare it with an exact initial sensitivity because that was not
recorded and local initial replay failed verification.

## 4. Temporal variation is coordinated on this simple corpus

Across the independent probes, channel gate-variation covariance effective rank
averages about1.7..2.0depending on task/initialization, despite128output channels.
Mean absolute active-gate correlations are roughly0.53..0.59. This is not state
rank, useful memory capacity, or a count of necessary gates. Repeated filler
positions dominate the simple fixed-length corpus and can induce coordinated
variation. Do not infer that a two-gate model would be equivalent from this
statistic; that would require a separately controlled experiment.

## 5. Direct functional intervention

See ../three_seed_gate_ablation_20261117/REPORT.md. Temporarily replacing each
channel-decay vector by its own mean changes accuracy without touching weights.
All six MQAR channel models lose accuracy on the fixed training-side probe;
overwrite dependence varies strongly by seed. This verifies reliance on
channel-specific structure in the already-trained solutions, not superiority
over a separately trained scalar architecture or a cause of faster learning.

## Interpretation

The endpoints generalize to fresh same-distribution examples, while the models
develop different gate functions despite matched starting means. That supports
describing calibration as a limited initialization treatment, not a guarantee of
equal memory dynamics. The evidence still does not establish why the early MQAR
gap changes, nor a universal benefit of calibration on overwrite. Preserve those
distinctions in the paper. No additional64epoch training is needed to establish
these diagnostic observations.
