# KATA replication and rewrite-shortcut evaluation

Runs: lr_0.001, lr_0.003, lr_0.01. Wall time 29 s. Predictions frozen before training.

## Verdicts

- **R1_replication**: `{"selected_learning_rate": "lr_0.003", "ours": {"K64": 0.995, "K128": 0.765, "K256": 0.284, "K384": 0.119}, "ours_average": 0.541, "paper": {"model": "Gated DeltaNet, dh=64, 1,167,878 parameters", "K=64": 0.998, "K=128": 0.853, "K=256": 0.384, "K=384": 0.217, "average": 0.613}, "adequate": true}`
- **R2_rewrite_shortcut**: `{"selected_shows_it": false, "all_models": {"lr_0.001": {"two_pass": 1.0, "three_pass": 1.0, "four_pass": 1.0, "shortcut": false}, "lr_0.003": {"two_pass": 1.0, "three_pass": 1.0, "four_pass": 1.0, "shortcut": false}, "lr_0.01": {"two_pass": 1.0, "three_pass": 1.0, "four_pass": 1.0, "shortcut": false}}}`
- **R3_error_identity**: `{"second_write_share": 0.0, "error_counts": {"other_key_value": 13, "other": 1}, "confirmed": false}`
- **R4_partial_third_pass**: `{"lr_0.001": {"rewritten_keys": 1.0, "keys_left_at_pass_2": 0.999}, "lr_0.003": {"rewritten_keys": 1.0, "keys_left_at_pass_2": 1.0}, "lr_0.01": {"rewritten_keys": 1.0, "keys_left_at_pass_2": 1.0}}`
- **R5_single_pass**: `{"lr_0.001": 1.0, "lr_0.003": 1.0, "lr_0.01": 1.0}`

## Accuracy per cell

| Cell | lr_0.001 | lr_0.003 | lr_0.01 |
|---|---:|---:|---:|
| two_pass_K4 | 100.0% | 100.0% | 100.0% |
| two_pass_K8 | 100.0% | 100.0% | 100.0% |
| two_pass_K16 | 100.0% | 100.0% | 100.0% |
| two_pass_K32 | 99.9% | 100.0% | 100.0% |
| two_pass_K64 | 96.5% | 99.5% | 98.8% |
| two_pass_K128 | 58.9% | 76.5% | 71.7% |
| two_pass_K256 | 19.7% | 28.4% | 26.6% |
| two_pass_K384 | 8.3% | 11.9% | 11.7% |
| one_pass_K4 | 100.0% | 100.0% | 100.0% |
| three_pass_K4 | 100.0% | 100.0% | 100.0% |
| four_pass_K4 | 100.0% | 100.0% | 100.0% |
| partial_third_K4 | 100.0% | 100.0% | 100.0% |
| one_pass_K8 | 100.0% | 100.0% | 100.0% |
| three_pass_K8 | 100.0% | 100.0% | 100.0% |
| four_pass_K8 | 100.0% | 100.0% | 100.0% |
| partial_third_K8 | 100.0% | 100.0% | 100.0% |
| one_pass_K16 | 100.0% | 100.0% | 100.0% |
| three_pass_K16 | 100.0% | 100.0% | 100.0% |
| four_pass_K16 | 100.0% | 100.0% | 100.0% |
| partial_third_K16 | 100.0% | 100.0% | 100.0% |
| one_pass_K32 | 100.0% | 100.0% | 100.0% |
| three_pass_K32 | 99.9% | 100.0% | 100.0% |
| four_pass_K32 | 99.9% | 100.0% | 100.0% |
| partial_third_K32 | 99.9% | 100.0% | 100.0% |
