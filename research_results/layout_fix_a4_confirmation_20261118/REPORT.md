# Block-first layout confirmation

Models: 30. Wall time 2981 s. Fresh data, seed offset 1500000. Verdicts are reported for all models and separately per initialization or training arm. Hypotheses were frozen in protocol.json before evaluation.

## Verdicts

- **all_models**: `{"H1_shortcut": {"models_showing": 13, "of": 30, "confirmed": false, "models": ["A0_control/channel-wise/42", "A0_control/channel-wise/43", "A0_control/channel-wise/44", "A0_control/scalar/42", "A0_control/scalar/43", "A0_control/scalar/44", "A1_varied_distance/channel-wise/42", "A1_varied_distance/channel-wise/44", "A1_varied_distance/scalar/42", "A2_no_block_first/channel-wise/42", "A2_no_block_first/channel-wise/43", "A2_no_block_first/scalar/42", "A2_no_block_first/scalar/44"]}, "H2_error_identity": {"first_rewrite_share": 0.7912225705329153, "error_counts": {"first_rewrite": 6310, "initial": 845, "second_rewrite": 101, "other": 694, "third_rewrite": 25}, "confirmed": false}, "H3_trained_layout_sanity": {"models_at_least_95": 26, "of": 30, "confirmed": false}, "H4_rule_specificity": {"by_layout": {"block_after_one_unrelated": {"models_within_10_points": 9, "of": 13}, "run_broken_before_latest": {"models_within_10_points": 9, "of": 13}, "initial_then_run": {"models_within_10_points": 10, "of": 13}}, "confirmed": true}, "H5_gap_control": {"distance_12": {"deficit_persists": 11, "of": 13}, "distance_16": {"deficit_persists": 10, "of": 13}}}`
- **group_A0_control**: `{"H1_shortcut": {"models_showing": 6, "of": 6, "confirmed": true, "models": ["A0_control/channel-wise/42", "A0_control/channel-wise/43", "A0_control/channel-wise/44", "A0_control/scalar/42", "A0_control/scalar/43", "A0_control/scalar/44"]}, "H2_error_identity": {"first_rewrite_share": 0.9593318809005084, "error_counts": {"first_rewrite": 3963, "initial": 20, "second_rewrite": 46, "other": 80, "third_rewrite": 22}, "confirmed": true}, "H3_trained_layout_sanity": {"models_at_least_95": 6, "of": 6, "confirmed": true}, "H4_rule_specificity": {"by_layout": {"block_after_one_unrelated": {"models_within_10_points": 4, "of": 6}, "run_broken_before_latest": {"models_within_10_points": 5, "of": 6}, "initial_then_run": {"models_within_10_points": 4, "of": 6}}, "confirmed": true}, "H5_gap_control": {"distance_12": {"deficit_persists": 4, "of": 6}, "distance_16": {"deficit_persists": 4, "of": 6}}}`
- **group_A1_varied_distance**: `{"H1_shortcut": {"models_showing": 3, "of": 6, "confirmed": false, "models": ["A1_varied_distance/channel-wise/42", "A1_varied_distance/channel-wise/44", "A1_varied_distance/scalar/42"]}, "H2_error_identity": {"first_rewrite_share": 0.9745075318655851, "error_counts": {"first_rewrite": 1682, "other": 13, "second_rewrite": 19, "third_rewrite": 2, "initial": 10}, "confirmed": true}, "H3_trained_layout_sanity": {"models_at_least_95": 6, "of": 6, "confirmed": true}, "H4_rule_specificity": {"by_layout": {"block_after_one_unrelated": {"models_within_10_points": 1, "of": 3}, "run_broken_before_latest": {"models_within_10_points": 0, "of": 3}, "initial_then_run": {"models_within_10_points": 2, "of": 3}}, "confirmed": false}, "H5_gap_control": {"distance_12": {"deficit_persists": 3, "of": 3}, "distance_16": {"deficit_persists": 2, "of": 3}}}`
- **group_A2_no_block_first**: `{"H1_shortcut": {"models_showing": 4, "of": 6, "confirmed": true, "models": ["A2_no_block_first/channel-wise/42", "A2_no_block_first/channel-wise/43", "A2_no_block_first/scalar/42", "A2_no_block_first/scalar/44"]}, "H2_error_identity": {"first_rewrite_share": 0.3159144893111639, "error_counts": {"first_rewrite": 665, "initial": 815, "other": 588, "third_rewrite": 1, "second_rewrite": 36}, "confirmed": false}, "H3_trained_layout_sanity": {"models_at_least_95": 2, "of": 6, "confirmed": false}, "H4_rule_specificity": {"by_layout": {"block_after_one_unrelated": {"models_within_10_points": 4, "of": 4}, "run_broken_before_latest": {"models_within_10_points": 4, "of": 4}, "initial_then_run": {"models_within_10_points": 4, "of": 4}}, "confirmed": true}, "H5_gap_control": {"distance_12": {"deficit_persists": 4, "of": 4}, "distance_16": {"deficit_persists": 4, "of": 4}}}`
- **group_A3_mixed_rewrites**: `{"H1_shortcut": {"models_showing": 0, "of": 6, "confirmed": false, "models": []}, "H2_error_identity": {"first_rewrite_share": 0.0, "error_counts": {"other": 4}, "confirmed": false}, "H3_trained_layout_sanity": {"models_at_least_95": 6, "of": 6, "confirmed": true}, "H4_rule_specificity": {"by_layout": {"block_after_one_unrelated": {"models_within_10_points": 0, "of": 0}, "run_broken_before_latest": {"models_within_10_points": 0, "of": 0}, "initial_then_run": {"models_within_10_points": 0, "of": 0}}, "confirmed": false}, "H5_gap_control": {"distance_12": {"deficit_persists": 0, "of": 0}, "distance_16": {"deficit_persists": 0, "of": 0}}}`
- **group_A4_distance_and_rewrites**: `{"H1_shortcut": {"models_showing": 0, "of": 6, "confirmed": false, "models": []}, "H2_error_identity": {"first_rewrite_share": 0.0, "error_counts": {"other": 9}, "confirmed": false}, "H3_trained_layout_sanity": {"models_at_least_95": 6, "of": 6, "confirmed": true}, "H4_rule_specificity": {"by_layout": {"block_after_one_unrelated": {"models_within_10_points": 0, "of": 0}, "run_broken_before_latest": {"models_within_10_points": 0, "of": 0}, "initial_then_run": {"models_within_10_points": 0, "of": 0}}, "confirmed": false}, "H5_gap_control": {"distance_12": {"deficit_persists": 0, "of": 0}, "distance_16": {"deficit_persists": 0, "of": 0}}}`

## Latest-value accuracy by layout (mean over models)

| Cell | Layout | Latest | Unrelated | First-replacement answers |
|---|---|---:|---:|---:|
| distance12_r1 | block_first | 56.5% | 94.3% | 0.0% |
| distance12_r1 | general_nonblock | 93.6% | 79.4% | 0.0% |
| distance12_r3 | block_first | 50.7% | 93.5% | 7.7% |
| distance12_r3 | general_nonblock | 87.2% | 80.2% | 2.5% |
| distance16_r1 | block_first | 75.0% | 86.8% | 0.0% |
| distance16_r1 | general_nonblock | 93.5% | 79.5% | 0.0% |
| distance16_r3 | block_first | 65.6% | 86.4% | 7.2% |
| distance16_r3 | general_nonblock | 88.4% | 81.2% | 2.2% |
| distance20_r1 | block_first | 74.9% | 87.2% | 0.0% |
| distance20_r1 | general_nonblock | 93.6% | 81.3% | 0.0% |
| distance20_r3 | block_first | 67.2% | 86.9% | 6.0% |
| distance20_r3 | general_nonblock | 89.1% | 79.7% | 1.9% |
| distance24_r1 | block_first | 74.7% | 86.8% | 0.0% |
| distance24_r1 | general_nonblock | 93.1% | 80.1% | 0.0% |
| distance24_r3 | block_first | 68.5% | 86.4% | 5.6% |
| distance24_r3 | general_nonblock | 88.1% | 81.6% | 1.8% |
| main_r1 | block_after_one_unrelated | 99.6% | 99.1% | 0.0% |
| main_r1 | block_first | 91.3% | 96.0% | 0.0% |
| main_r1 | general_nonblock | 99.7% | 97.9% | 0.0% |
| main_r1 | run_broken_before_latest | 99.6% | 99.0% | 0.0% |
| main_r2 | block_after_one_unrelated | 88.5% | 98.8% | 10.8% |
| main_r2 | block_first | 67.6% | 95.3% | 26.5% |
| main_r2 | general_nonblock | 98.6% | 96.5% | 1.2% |
| main_r2 | initial_then_run | 92.0% | 98.4% | 6.5% |
| main_r2 | run_broken_before_latest | 87.0% | 95.7% | 7.9% |
| main_r3 | block_after_one_unrelated | 89.5% | 98.7% | 9.5% |
| main_r3 | block_first | 64.5% | 95.0% | 27.6% |
| main_r3 | general_nonblock | 96.1% | 94.8% | 2.7% |
| main_r3 | initial_then_run | 92.0% | 98.5% | 5.4% |
| main_r3 | run_broken_before_latest | 88.0% | 95.0% | 4.6% |
| main_r4 | block_after_one_unrelated | 89.3% | 98.7% | 9.7% |
| main_r4 | block_first | 64.1% | 94.7% | 28.0% |
| main_r4 | general_nonblock | 94.8% | 94.1% | 3.2% |
| main_r4 | initial_then_run | 91.3% | 98.4% | 5.7% |
| main_r4 | run_broken_before_latest | 88.5% | 94.6% | 6.5% |

## Per model, main_r3: block_first / general_nonblock / block_after_one / broken / initial_then_run

| Model | block_first | general | after_one | broken | initial_then_run |
|---|---:|---:|---:|---:|---:|
| A0_control/channel-wise/42 | 0.0% | 80.9% | 0.0% | 0.0% | 0.8% |
| A0_control/channel-wise/43 | 30.9% | 100.0% | 98.8% | 96.5% | 97.7% |
| A0_control/channel-wise/44 | 0.4% | 98.4% | 83.6% | 89.8% | 82.4% |
| A0_control/scalar/42 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| A0_control/scalar/43 | 35.5% | 100.0% | 100.0% | 100.0% | 100.0% |
| A0_control/scalar/44 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| A1_varied_distance/channel-wise/42 | 40.2% | 83.6% | 60.5% | 53.9% | 75.8% |
| A1_varied_distance/channel-wise/43 | 100.0% | 100.0% | 100.0% | 100.0% | 99.6% |
| A1_varied_distance/channel-wise/44 | 17.6% | 65.2% | 56.2% | 36.7% | 75.0% |
| A1_varied_distance/scalar/42 | 16.8% | 77.7% | 0.0% | 1.6% | 45.3% |
| A1_varied_distance/scalar/43 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| A1_varied_distance/scalar/44 | 82.4% | 92.2% | 99.2% | 84.4% | 98.8% |
| A2_no_block_first/channel-wise/42 | 0.8% | 88.3% | 87.1% | 79.7% | 85.9% |
| A2_no_block_first/channel-wise/43 | 54.7% | 100.0% | 100.0% | 100.0% | 100.0% |
| A2_no_block_first/channel-wise/44 | 96.5% | 100.0% | 100.0% | 100.0% | 100.0% |
| A2_no_block_first/scalar/42 | 0.4% | 100.0% | 100.0% | 100.0% | 100.0% |
| A2_no_block_first/scalar/43 | 99.2% | 99.6% | 100.0% | 100.0% | 100.0% |
| A2_no_block_first/scalar/44 | 60.5% | 100.0% | 100.0% | 100.0% | 100.0% |
| A3_mixed_rewrites/channel-wise/42 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| A3_mixed_rewrites/channel-wise/43 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| A3_mixed_rewrites/channel-wise/44 | 100.0% | 98.8% | 99.6% | 99.6% | 99.6% |
| A3_mixed_rewrites/scalar/42 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| A3_mixed_rewrites/scalar/43 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| A3_mixed_rewrites/scalar/44 | 100.0% | 99.2% | 100.0% | 100.0% | 100.0% |
| A4_distance_and_rewrites/channel-wise/42 | 100.0% | 100.0% | 100.0% | 99.6% | 100.0% |
| A4_distance_and_rewrites/channel-wise/43 | 100.0% | 100.0% | 100.0% | 100.0% | 99.6% |
| A4_distance_and_rewrites/channel-wise/44 | 98.8% | 100.0% | 100.0% | 99.6% | 100.0% |
| A4_distance_and_rewrites/scalar/42 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| A4_distance_and_rewrites/scalar/43 | 99.6% | 100.0% | 100.0% | 100.0% | 100.0% |
| A4_distance_and_rewrites/scalar/44 | 99.6% | 100.0% | 100.0% | 99.6% | 100.0% |

## Rule agreement on main_r2..r4, all layouts (share of latest-target answers equal to each rule)

| Model | latest | initial | second_occurrence | memory_slot_1 | opening_run_second | final_run_second |
|---|---:|---:|---:|---:|---:|---:|
| A0_control/channel-wise/42 | 17.6% | 0.0% | 59.9% | 20.0% | 37.6% | 53.0% |
| A0_control/channel-wise/43 | 84.0% | 0.1% | 14.8% | 14.5% | 93.8% | 55.6% |
| A0_control/channel-wise/44 | 61.2% | 0.0% | 34.0% | 19.9% | 81.1% | 58.4% |
| A0_control/scalar/42 | 80.0% | 0.0% | 20.0% | 20.0% | 100.0% | 61.2% |
| A0_control/scalar/43 | 87.6% | 0.4% | 8.9% | 8.9% | 88.9% | 50.1% |
| A0_control/scalar/44 | 80.0% | 0.0% | 20.0% | 20.0% | 99.9% | 61.1% |
| A1_varied_distance/channel-wise/42 | 64.0% | 0.1% | 31.1% | 21.6% | 67.4% | 51.1% |
| A1_varied_distance/channel-wise/43 | 99.8% | 0.0% | 0.0% | 0.0% | 79.9% | 41.2% |
| A1_varied_distance/channel-wise/44 | 53.6% | 0.1% | 43.2% | 33.6% | 62.0% | 46.2% |
| A1_varied_distance/scalar/42 | 31.0% | 7.0% | 52.3% | 30.4% | 40.8% | 49.5% |
| A1_varied_distance/scalar/43 | 100.0% | 0.0% | 0.0% | 0.0% | 80.0% | 41.3% |
| A1_varied_distance/scalar/44 | 91.6% | 0.4% | 6.8% | 6.8% | 78.6% | 40.7% |
| A2_no_block_first/channel-wise/42 | 69.6% | 9.7% | 13.4% | 13.3% | 79.6% | 45.3% |
| A2_no_block_first/channel-wise/43 | 94.2% | 0.0% | 0.0% | 0.0% | 80.0% | 41.3% |
| A2_no_block_first/channel-wise/44 | 99.3% | 0.5% | 0.0% | 0.0% | 80.0% | 41.1% |
| A2_no_block_first/scalar/42 | 80.1% | 12.2% | 7.3% | 7.3% | 87.3% | 48.5% |
| A2_no_block_first/scalar/43 | 99.7% | 0.0% | 0.0% | 0.0% | 80.0% | 41.3% |
| A2_no_block_first/scalar/44 | 91.7% | 0.0% | 0.0% | 0.0% | 80.0% | 41.1% |
| A3_mixed_rewrites/channel-wise/42 | 100.0% | 0.0% | 0.0% | 0.0% | 80.0% | 41.2% |
| A3_mixed_rewrites/channel-wise/43 | 100.0% | 0.0% | 0.0% | 0.0% | 80.0% | 41.3% |
| A3_mixed_rewrites/channel-wise/44 | 99.5% | 0.0% | 0.1% | 0.0% | 79.5% | 40.7% |
| A3_mixed_rewrites/scalar/42 | 99.9% | 0.0% | 0.0% | 0.0% | 79.9% | 41.1% |
| A3_mixed_rewrites/scalar/43 | 100.0% | 0.0% | 0.0% | 0.0% | 80.0% | 41.3% |
| A3_mixed_rewrites/scalar/44 | 99.9% | 0.0% | 0.0% | 0.0% | 79.9% | 41.1% |
| A4_distance_and_rewrites/channel-wise/42 | 99.9% | 0.0% | 0.0% | 0.0% | 79.9% | 41.1% |
| A4_distance_and_rewrites/channel-wise/43 | 99.9% | 0.0% | 0.0% | 0.0% | 79.9% | 41.2% |
| A4_distance_and_rewrites/channel-wise/44 | 99.8% | 0.0% | 0.0% | 0.0% | 79.9% | 41.1% |
| A4_distance_and_rewrites/scalar/42 | 100.0% | 0.0% | 0.0% | 0.0% | 80.0% | 41.2% |
| A4_distance_and_rewrites/scalar/43 | 99.9% | 0.0% | 0.0% | 0.0% | 79.9% | 41.2% |
| A4_distance_and_rewrites/scalar/44 | 99.9% | 0.0% | 0.0% | 0.0% | 79.9% | 41.1% |

Interpretation limits: discovery-confirmation on synthetic data for 1.6M-parameter models; block_first is also the layout with the longest leading filler, so absolute position is only partly separated by the distance controls.
