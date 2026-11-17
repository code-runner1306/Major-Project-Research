# Block-first layout confirmation

Models: 18. Wall time 178 s. Fresh data, seed offset 1300000. Hypotheses were frozen in protocol.json before evaluation.

## Verdicts

- **H1_shortcut**: `{"models_showing": 16, "of": 18, "confirmed": true, "models": ["calibrated/channel-wise/42", "calibrated/channel-wise/43", "calibrated/channel-wise/44", "calibrated/scalar/42", "calibrated/scalar/43", "calibrated/scalar/44", "default/channel-wise/42", "default/channel-wise/44", "default/scalar/42", "default/scalar/44", "joint/channel-wise/42", "joint/channel-wise/43", "joint/channel-wise/44", "joint/scalar/42", "joint/scalar/43", "joint/scalar/44"]}`
- **H2_error_identity**: `{"first_rewrite_share": 0.9712869594941036, "error_counts": {"first_rewrite": 11366, "second_rewrite": 7, "third_rewrite": 8, "initial": 281, "other": 40}, "confirmed": true}`
- **H3_trained_layout_sanity**: `{"models_at_least_95": 18, "of": 18, "confirmed": true}`
- **H4_rule_specificity**: `{"by_layout": {"block_after_one_unrelated": {"models_within_10_points": 13, "of": 16}, "run_broken_before_latest": {"models_within_10_points": 13, "of": 16}, "initial_then_run": {"models_within_10_points": 13, "of": 16}}, "confirmed": true}`
- **H5_gap_control**: `{"distance_12": {"deficit_persists": 14, "of": 16}, "distance_16": {"deficit_persists": 15, "of": 16}}`

## Latest-value accuracy by layout (mean over models)

| Cell | Layout | Latest | Unrelated | First-replacement answers |
|---|---|---:|---:|---:|
| distance12_r1 | block_first | 32.8% | 94.9% | 0.0% |
| distance12_r1 | general_nonblock | 95.7% | 66.6% | 0.0% |
| distance12_r3 | block_first | 31.9% | 94.3% | 1.6% |
| distance12_r3 | general_nonblock | 85.5% | 69.6% | 3.0% |
| distance16_r1 | block_first | 59.4% | 80.9% | 0.0% |
| distance16_r1 | general_nonblock | 96.2% | 68.8% | 0.0% |
| distance16_r3 | block_first | 48.4% | 80.9% | 2.8% |
| distance16_r3 | general_nonblock | 86.1% | 70.8% | 2.6% |
| main_r1 | block_after_one_unrelated | 99.4% | 98.4% | 0.0% |
| main_r1 | block_first | 100.0% | 98.0% | 0.0% |
| main_r1 | general_nonblock | 100.0% | 95.5% | 0.0% |
| main_r1 | run_broken_before_latest | 99.9% | 98.1% | 0.0% |
| main_r2 | block_after_one_unrelated | 86.4% | 97.5% | 13.1% |
| main_r2 | block_first | 17.0% | 96.3% | 82.9% |
| main_r2 | general_nonblock | 95.6% | 93.4% | 2.5% |
| main_r2 | initial_then_run | 86.8% | 96.4% | 13.1% |
| main_r2 | run_broken_before_latest | 91.9% | 91.3% | 3.0% |
| main_r3 | block_after_one_unrelated | 87.4% | 97.4% | 12.0% |
| main_r3 | block_first | 14.2% | 97.0% | 81.8% |
| main_r3 | general_nonblock | 95.9% | 91.8% | 1.5% |
| main_r3 | initial_then_run | 87.5% | 97.3% | 12.1% |
| main_r3 | run_broken_before_latest | 87.2% | 91.3% | 1.3% |
| main_r4 | block_after_one_unrelated | 86.3% | 97.3% | 12.9% |
| main_r4 | block_first | 14.9% | 97.1% | 81.9% |
| main_r4 | general_nonblock | 95.4% | 90.1% | 1.2% |
| main_r4 | initial_then_run | 87.2% | 97.3% | 11.9% |
| main_r4 | run_broken_before_latest | 87.1% | 90.0% | 0.9% |

## Per model, main_r3: block_first / general_nonblock / block_after_one / broken / initial_then_run

| Model | block_first | general | after_one | broken | initial_then_run |
|---|---:|---:|---:|---:|---:|
| calibrated/channel-wise/42 | 11.7% | 99.6% | 99.2% | 100.0% | 99.2% |
| calibrated/channel-wise/43 | 0.4% | 82.0% | 1.6% | 0.4% | 1.2% |
| calibrated/channel-wise/44 | 7.4% | 100.0% | 100.0% | 100.0% | 100.0% |
| calibrated/scalar/42 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| calibrated/scalar/43 | 10.2% | 100.0% | 100.0% | 100.0% | 100.0% |
| calibrated/scalar/44 | 0.0% | 50.4% | 15.2% | 9.4% | 10.2% |
| default/channel-wise/42 | 0.0% | 100.0% | 100.0% | 100.0% | 99.6% |
| default/channel-wise/43 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| default/channel-wise/44 | 9.0% | 100.0% | 100.0% | 99.6% | 99.6% |
| default/scalar/42 | 0.0% | 99.6% | 93.8% | 81.6% | 96.5% |
| default/scalar/43 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| default/scalar/44 | 6.6% | 100.0% | 100.0% | 100.0% | 100.0% |
| joint/channel-wise/42 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| joint/channel-wise/43 | 0.0% | 99.2% | 94.1% | 94.9% | 99.2% |
| joint/channel-wise/44 | 0.4% | 100.0% | 100.0% | 100.0% | 100.0% |
| joint/scalar/42 | 0.0% | 99.2% | 99.2% | 95.7% | 99.2% |
| joint/scalar/43 | 3.9% | 100.0% | 97.7% | 98.8% | 100.0% |
| joint/scalar/44 | 5.5% | 96.9% | 72.7% | 89.5% | 70.7% |

## Rule agreement on main_r2..r4, all layouts (share of latest-target answers equal to each rule)

| Model | latest | initial | second_occurrence | memory_slot_1 | opening_run_second | final_run_second |
|---|---:|---:|---:|---:|---:|---:|
| calibrated/channel-wise/42 | 81.5% | 0.0% | 18.1% | 18.0% | 97.6% | 59.1% |
| calibrated/channel-wise/43 | 19.6% | 0.0% | 60.0% | 20.3% | 39.5% | 53.9% |
| calibrated/channel-wise/44 | 84.0% | 0.0% | 15.9% | 15.9% | 95.9% | 56.8% |
| calibrated/scalar/42 | 80.1% | 0.0% | 19.9% | 19.9% | 99.9% | 61.3% |
| calibrated/scalar/43 | 82.3% | 7.2% | 9.6% | 9.6% | 89.6% | 50.8% |
| calibrated/scalar/44 | 16.6% | 0.7% | 63.3% | 23.3% | 36.6% | 50.6% |
| default/channel-wise/42 | 79.9% | 0.0% | 20.0% | 20.0% | 99.9% | 61.4% |
| default/channel-wise/43 | 100.0% | 0.0% | 0.0% | 0.0% | 80.0% | 41.2% |
| default/channel-wise/44 | 82.0% | 0.0% | 18.0% | 18.0% | 97.8% | 58.7% |
| default/scalar/42 | 74.0% | 0.1% | 23.5% | 22.7% | 93.9% | 58.9% |
| default/scalar/43 | 100.0% | 0.0% | 0.0% | 0.0% | 80.0% | 41.2% |
| default/scalar/44 | 80.9% | 0.0% | 18.9% | 18.9% | 98.9% | 59.8% |
| joint/channel-wise/42 | 80.0% | 0.0% | 20.0% | 20.0% | 100.0% | 61.3% |
| joint/channel-wise/43 | 77.4% | 0.3% | 20.9% | 21.4% | 97.4% | 59.8% |
| joint/channel-wise/44 | 81.4% | 0.0% | 18.6% | 18.6% | 98.6% | 59.4% |
| joint/scalar/42 | 78.8% | 0.2% | 20.8% | 21.2% | 98.8% | 60.5% |
| joint/scalar/43 | 80.4% | 0.5% | 18.8% | 19.2% | 98.0% | 59.7% |
| joint/scalar/44 | 66.1% | 0.0% | 32.1% | 18.6% | 83.3% | 61.9% |

Interpretation limits: discovery-confirmation on synthetic data for 1.6M-parameter models; block_first is also the layout with the longest leading filler, so absolute position is only partly separated by the distance controls.
