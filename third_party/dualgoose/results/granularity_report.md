# Scalar vs channel-wise decay in the Anatomy harness

Predictions frozen in PREDICTIONS_granularity.json before training.

## Verdicts

- **R0_reproduction**: `{"E1": true, "E4": true, "E5": true}`
- **G1_capacity**: `{"mean_difference": -0.016796874999999933, "p": 0.0005953798523457966, "confirmed": false}`
- **G2_distractors**: `{"lock_in": [6, 6], "fisher_p": 1.0, "direction_holds": true}`
- **G3_collision**: `{"lock_in": [3, 5], "fisher_p": 0.6499166468206716, "direction_holds": false, "significant": false}`
- **G4_mechanism_inputs**: `{"E1": {"original": 0.983, "fixed_channel_mean": 0.4518, "token_channel_average": 0.2619, "fixed_scalar_mean": 0.2466}, "E4": {"original": 0.6647, "fixed_channel_mean": 0.479, "token_channel_average": 0.0177, "fixed_scalar_mean": 0.0177}, "E5": {"original": 0.3857, "fixed_channel_mean": 0.2167, "token_channel_average": 0.0187, "fixed_scalar_mean": 0.0187}}`

## Cells

| Cell | Arm | n | mean | lock-in | rebind | per-seed |
|---|---|---:|---:|---:|---:|---|
| E1 | channel (armed_gdn) | 10 | 0.983 | 10/10 | 0.037 | [0.979, 1.0, 0.999, 0.917, 0.995, 1.0, 0.997, 0.968, 0.994, 0.983] |
| E1 | scalar (armed_gdn_scalar) | 10 | 1.000 | 10/10 | 0.036 | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] |
| E1 | published armed_gdn | 10 | 0.983 | 10/10 | 0.037 | [0.979, 1.0, 0.999, 0.917, 0.995, 1.0, 0.997, 0.968, 0.994, 0.983] |
| E4 | channel (armed_gdn) | 10 | 0.665 | 6/10 | 0.038 | [1.0, 0.995, 0.827, 0.986, 0.362, 0.419, 0.115, 0.037, 1.0, 0.905] |
| E4 | scalar (armed_gdn_scalar) | 10 | 0.591 | 6/10 | 0.020 | [0.019, 1.0, 0.018, 1.0, 0.969, 0.065, 0.82, 1.0, 0.017, 1.0] |
| E4 | published armed_gdn | 10 | 0.668 | 6/10 | - | [1.0, 0.866, 0.883, 0.989, 0.362, 0.447, 0.12, 0.041, 1.0, 0.971] |
| E5 | channel (armed_gdn) | 10 | 0.386 | 3/10 | 0.033 | [1.0, 0.973, 0.09, 0.137, 0.515, 0.048, 0.054, 0.023, 0.968, 0.048] |
| E5 | scalar (armed_gdn_scalar) | 10 | 0.499 | 5/10 | 0.021 | [0.019, 1.0, 0.018, 0.924, 0.083, 0.082, 0.847, 1.0, 0.018, 0.999] |
| E5 | published armed_gdn | 10 | 0.383 | 3/10 | 0.033 | [1.0, 0.932, 0.09, 0.149, 0.515, 0.047, 0.058, 0.023, 0.963, 0.05] |
