# Block-first layout confirmation

Models: 24. Wall time 1012 s. Fresh data, seed offset 1400000. Verdicts are reported for all models and separately per initialization or training arm. Hypotheses were frozen in protocol.json before evaluation.

## Verdicts

- **all_models**: `{"H1_shortcut": {"models_showing": 13, "of": 24, "confirmed": true, "models": ["A0_control/channel-wise/42", "A0_control/channel-wise/43", "A0_control/channel-wise/44", "A0_control/scalar/42", "A0_control/scalar/43", "A0_control/scalar/44", "A1_varied_distance/channel-wise/42", "A1_varied_distance/channel-wise/44", "A1_varied_distance/scalar/42", "A2_no_block_first/channel-wise/42", "A2_no_block_first/channel-wise/43", "A2_no_block_first/scalar/42", "A2_no_block_first/scalar/44"]}, "H2_error_identity": {"first_rewrite_share": 0.785982478097622, "error_counts": {"first_rewrite": 6280, "other": 714, "second_rewrite": 123, "initial": 847, "third_rewrite": 26}, "confirmed": false}, "H3_trained_layout_sanity": {"models_at_least_95": 19, "of": 24, "confirmed": false}, "H4_rule_specificity": {"by_layout": {"block_after_one_unrelated": {"models_within_10_points": 8, "of": 13}, "run_broken_before_latest": {"models_within_10_points": 8, "of": 13}, "initial_then_run": {"models_within_10_points": 10, "of": 13}}, "confirmed": true}, "H5_gap_control": {"distance_12": {"deficit_persists": 11, "of": 13}, "distance_16": {"deficit_persists": 11, "of": 13}}}`
- **group_A0_control**: `{"H1_shortcut": {"models_showing": 6, "of": 6, "confirmed": true, "models": ["A0_control/channel-wise/42", "A0_control/channel-wise/43", "A0_control/channel-wise/44", "A0_control/scalar/42", "A0_control/scalar/43", "A0_control/scalar/44"]}, "H2_error_identity": {"first_rewrite_share": 0.9575551782682513, "error_counts": {"first_rewrite": 3948, "other": 94, "second_rewrite": 39, "initial": 20, "third_rewrite": 22}, "confirmed": true}, "H3_trained_layout_sanity": {"models_at_least_95": 6, "of": 6, "confirmed": true}, "H4_rule_specificity": {"by_layout": {"block_after_one_unrelated": {"models_within_10_points": 4, "of": 6}, "run_broken_before_latest": {"models_within_10_points": 4, "of": 6}, "initial_then_run": {"models_within_10_points": 4, "of": 6}}, "confirmed": true}, "H5_gap_control": {"distance_12": {"deficit_persists": 4, "of": 6}, "distance_16": {"deficit_persists": 4, "of": 6}}}`
- **group_A1_varied_distance**: `{"H1_shortcut": {"models_showing": 3, "of": 6, "confirmed": false, "models": ["A1_varied_distance/channel-wise/42", "A1_varied_distance/channel-wise/44", "A1_varied_distance/scalar/42"]}, "H2_error_identity": {"first_rewrite_share": 0.9671848013816926, "error_counts": {"first_rewrite": 1680, "second_rewrite": 32, "third_rewrite": 4, "other": 9, "initial": 12}, "confirmed": true}, "H3_trained_layout_sanity": {"models_at_least_95": 6, "of": 6, "confirmed": true}, "H4_rule_specificity": {"by_layout": {"block_after_one_unrelated": {"models_within_10_points": 0, "of": 3}, "run_broken_before_latest": {"models_within_10_points": 0, "of": 3}, "initial_then_run": {"models_within_10_points": 2, "of": 3}}, "confirmed": false}, "H5_gap_control": {"distance_12": {"deficit_persists": 3, "of": 3}, "distance_16": {"deficit_persists": 3, "of": 3}}}`
- **group_A2_no_block_first**: `{"H1_shortcut": {"models_showing": 4, "of": 6, "confirmed": true, "models": ["A2_no_block_first/channel-wise/42", "A2_no_block_first/channel-wise/43", "A2_no_block_first/scalar/42", "A2_no_block_first/scalar/44"]}, "H2_error_identity": {"first_rewrite_share": 0.3066792097836312, "error_counts": {"first_rewrite": 652, "initial": 815, "other": 607, "second_rewrite": 52}, "confirmed": false}, "H3_trained_layout_sanity": {"models_at_least_95": 1, "of": 6, "confirmed": false}, "H4_rule_specificity": {"by_layout": {"block_after_one_unrelated": {"models_within_10_points": 4, "of": 4}, "run_broken_before_latest": {"models_within_10_points": 4, "of": 4}, "initial_then_run": {"models_within_10_points": 4, "of": 4}}, "confirmed": true}, "H5_gap_control": {"distance_12": {"deficit_persists": 4, "of": 4}, "distance_16": {"deficit_persists": 4, "of": 4}}}`
- **group_A3_mixed_rewrites**: `{"H1_shortcut": {"models_showing": 0, "of": 6, "confirmed": false, "models": []}, "H2_error_identity": {"first_rewrite_share": 0.0, "error_counts": {"other": 4}, "confirmed": false}, "H3_trained_layout_sanity": {"models_at_least_95": 6, "of": 6, "confirmed": true}, "H4_rule_specificity": {"by_layout": {"block_after_one_unrelated": {"models_within_10_points": 0, "of": 0}, "run_broken_before_latest": {"models_within_10_points": 0, "of": 0}, "initial_then_run": {"models_within_10_points": 0, "of": 0}}, "confirmed": false}, "H5_gap_control": {"distance_12": {"deficit_persists": 0, "of": 0}, "distance_16": {"deficit_persists": 0, "of": 0}}}`

## Latest-value accuracy by layout (mean over models)

| Cell | Layout | Latest | Unrelated | First-replacement answers |
|---|---|---:|---:|---:|
| distance12_r1 | block_first | 45.4% | 93.1% | 0.0% |
| distance12_r1 | general_nonblock | 92.3% | 75.5% | 0.0% |
| distance12_r3 | block_first | 38.9% | 92.3% | 9.3% |
| distance12_r3 | general_nonblock | 83.3% | 76.1% | 3.5% |
| distance16_r1 | block_first | 68.5% | 83.2% | 0.0% |
| distance16_r1 | general_nonblock | 92.7% | 75.1% | 0.0% |
| distance16_r3 | block_first | 57.4% | 81.2% | 9.8% |
| distance16_r3 | general_nonblock | 85.4% | 74.5% | 2.5% |
| distance20_r1 | block_first | 69.9% | 83.3% | 0.0% |
| distance20_r1 | general_nonblock | 93.0% | 75.8% | 0.0% |
| distance20_r3 | block_first | 60.3% | 81.6% | 7.8% |
| distance20_r3 | general_nonblock | 86.0% | 75.6% | 2.1% |
| distance24_r1 | block_first | 70.3% | 83.3% | 0.0% |
| distance24_r1 | general_nonblock | 91.3% | 74.1% | 0.0% |
| distance24_r3 | block_first | 60.9% | 83.2% | 7.2% |
| distance24_r3 | general_nonblock | 85.7% | 76.0% | 2.4% |
| main_r1 | block_after_one_unrelated | 99.4% | 98.9% | 0.0% |
| main_r1 | block_first | 89.0% | 94.8% | 0.0% |
| main_r1 | general_nonblock | 99.6% | 97.4% | 0.0% |
| main_r1 | run_broken_before_latest | 99.6% | 98.6% | 0.0% |
| main_r2 | block_after_one_unrelated | 85.4% | 98.6% | 13.7% |
| main_r2 | block_first | 59.8% | 94.4% | 32.8% |
| main_r2 | general_nonblock | 98.1% | 95.1% | 1.6% |
| main_r2 | initial_then_run | 89.7% | 98.4% | 8.5% |
| main_r2 | run_broken_before_latest | 84.0% | 95.6% | 9.9% |
| main_r3 | block_after_one_unrelated | 86.6% | 98.2% | 12.0% |
| main_r3 | block_first | 54.9% | 93.8% | 34.4% |
| main_r3 | general_nonblock | 95.8% | 93.8% | 2.8% |
| main_r3 | initial_then_run | 89.8% | 98.3% | 6.8% |
| main_r3 | run_broken_before_latest | 85.5% | 94.3% | 5.2% |
| main_r4 | block_after_one_unrelated | 86.8% | 98.2% | 11.8% |
| main_r4 | block_first | 55.3% | 93.3% | 35.0% |
| main_r4 | general_nonblock | 93.7% | 93.4% | 3.8% |
| main_r4 | initial_then_run | 89.8% | 98.3% | 6.8% |
| main_r4 | run_broken_before_latest | 86.2% | 93.7% | 7.7% |

## Per model, main_r3: block_first / general_nonblock / block_after_one / broken / initial_then_run

| Model | block_first | general | after_one | broken | initial_then_run |
|---|---:|---:|---:|---:|---:|
| A0_control/channel-wise/42 | 0.0% | 84.8% | 0.4% | 0.0% | 0.4% |
| A0_control/channel-wise/43 | 30.9% | 100.0% | 98.8% | 94.9% | 98.0% |
| A0_control/channel-wise/44 | 0.0% | 98.8% | 81.2% | 86.7% | 78.5% |
| A0_control/scalar/42 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| A0_control/scalar/43 | 34.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| A0_control/scalar/44 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| A1_varied_distance/channel-wise/42 | 37.9% | 82.0% | 56.6% | 57.0% | 73.4% |
| A1_varied_distance/channel-wise/43 | 99.2% | 100.0% | 100.0% | 99.6% | 100.0% |
| A1_varied_distance/channel-wise/44 | 15.6% | 66.8% | 55.1% | 43.8% | 74.2% |
| A1_varied_distance/scalar/42 | 16.0% | 82.0% | 0.0% | 0.8% | 41.8% |
| A1_varied_distance/scalar/43 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| A1_varied_distance/scalar/44 | 83.2% | 94.9% | 99.6% | 87.1% | 98.4% |
| A2_no_block_first/channel-wise/42 | 0.0% | 91.0% | 87.5% | 84.4% | 90.6% |
| A2_no_block_first/channel-wise/43 | 46.9% | 100.0% | 100.0% | 100.0% | 100.0% |
| A2_no_block_first/channel-wise/44 | 96.9% | 100.0% | 100.0% | 100.0% | 100.0% |
| A2_no_block_first/scalar/42 | 0.8% | 100.0% | 100.0% | 99.6% | 100.0% |
| A2_no_block_first/scalar/43 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| A2_no_block_first/scalar/44 | 55.1% | 100.0% | 100.0% | 100.0% | 100.0% |
| A3_mixed_rewrites/channel-wise/42 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| A3_mixed_rewrites/channel-wise/43 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| A3_mixed_rewrites/channel-wise/44 | 100.0% | 98.4% | 100.0% | 99.2% | 99.6% |
| A3_mixed_rewrites/scalar/42 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| A3_mixed_rewrites/scalar/43 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| A3_mixed_rewrites/scalar/44 | 100.0% | 99.6% | 100.0% | 100.0% | 100.0% |

## Rule agreement on main_r2..r4, all layouts (share of latest-target answers equal to each rule)

| Model | latest | initial | second_occurrence | memory_slot_1 | opening_run_second | final_run_second |
|---|---:|---:|---:|---:|---:|---:|
| A0_control/channel-wise/42 | 18.2% | 0.0% | 59.8% | 20.0% | 38.2% | 53.0% |
| A0_control/channel-wise/43 | 84.2% | 0.2% | 14.7% | 14.2% | 93.4% | 55.2% |
| A0_control/channel-wise/44 | 60.4% | 0.0% | 34.7% | 20.0% | 80.3% | 57.6% |
| A0_control/scalar/42 | 80.0% | 0.0% | 20.0% | 20.0% | 100.0% | 61.0% |
| A0_control/scalar/43 | 87.6% | 0.3% | 8.6% | 8.6% | 88.6% | 49.7% |
| A0_control/scalar/44 | 80.0% | 0.0% | 20.0% | 20.0% | 100.0% | 60.7% |
| A1_varied_distance/channel-wise/42 | 62.4% | 0.2% | 32.4% | 21.8% | 66.7% | 52.4% |
| A1_varied_distance/channel-wise/43 | 99.8% | 0.0% | 0.0% | 0.0% | 79.9% | 41.0% |
| A1_varied_distance/channel-wise/44 | 54.9% | 0.3% | 41.5% | 33.2% | 63.8% | 46.4% |
| A1_varied_distance/scalar/42 | 31.4% | 7.8% | 51.0% | 30.2% | 41.1% | 49.2% |
| A1_varied_distance/scalar/43 | 100.0% | 0.0% | 0.0% | 0.0% | 80.0% | 41.1% |
| A1_varied_distance/scalar/44 | 92.9% | 0.3% | 5.8% | 5.8% | 78.9% | 40.3% |
| A2_no_block_first/channel-wise/42 | 71.0% | 9.8% | 12.5% | 12.2% | 80.7% | 45.8% |
| A2_no_block_first/channel-wise/43 | 94.2% | 0.0% | 0.0% | 0.0% | 80.0% | 41.1% |
| A2_no_block_first/channel-wise/44 | 99.1% | 0.5% | 0.0% | 0.0% | 80.0% | 40.7% |
| A2_no_block_first/scalar/42 | 80.1% | 12.0% | 7.3% | 7.3% | 87.3% | 48.3% |
| A2_no_block_first/scalar/43 | 99.8% | 0.0% | 0.0% | 0.0% | 80.0% | 41.1% |
| A2_no_block_first/scalar/44 | 91.4% | 0.0% | 0.0% | 0.0% | 80.0% | 40.7% |
| A3_mixed_rewrites/channel-wise/42 | 100.0% | 0.0% | 0.0% | 0.0% | 80.0% | 41.0% |
| A3_mixed_rewrites/channel-wise/43 | 100.0% | 0.0% | 0.0% | 0.0% | 80.0% | 41.1% |
| A3_mixed_rewrites/channel-wise/44 | 99.5% | 0.0% | 0.1% | 0.1% | 79.5% | 40.3% |
| A3_mixed_rewrites/scalar/42 | 99.8% | 0.0% | 0.0% | 0.0% | 79.9% | 41.0% |
| A3_mixed_rewrites/scalar/43 | 100.0% | 0.0% | 0.0% | 0.0% | 80.0% | 41.1% |
| A3_mixed_rewrites/scalar/44 | 99.9% | 0.0% | 0.0% | 0.0% | 79.9% | 40.6% |

Interpretation limits: discovery-confirmation on synthetic data for 1.6M-parameter models; block_first is also the layout with the longest leading filler, so absolute position is only partly separated by the distance controls.
