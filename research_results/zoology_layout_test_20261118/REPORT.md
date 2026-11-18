# Zoology MQAR layout test

Models: 14. Wall time 1265 s. Seed offset 1600000. Hypotheses frozen in protocol.json before evaluation.

## Verdicts

- **Z1_sanity**: `{"models_at_least_95": 14, "of": 14}`
- **Z2_position_shortcut**: `{"models_showing": 1, "of": 14, "models": ["default/scalar/44"], "majority": false}`
- **Z3_anchored_errors**: `{"anchored_share": 0.34183673469387754, "errors": 392, "confirmed": false}`
- **Z4_query_layout**: `{"near": {"models_within_10_points": 14, "of": 14}, "far": {"models_within_10_points": 14, "of": 14}, "memory_order": {"models_within_10_points": 14, "of": 14}, "reverse_order": {"models_within_10_points": 14, "of": 14}}`

## Accuracy per model and cell

| Model | zoology_baseline | shift_0 | shift_1 | shift_2 | shift_4 | shift_8 | shift_16 | near | far | memory_order | reverse_order | shift_8_far |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calibrated/channel-wise/42 | 100.0% | 100.0% | 99.5% | 99.8% | 99.8% | 99.7% | 99.6% | 100.0% | 100.0% | 100.0% | 100.0% | 99.7% |
| calibrated/channel-wise/43 | 99.9% | 100.0% | 96.6% | 93.3% | 88.5% | 90.2% | 91.1% | 100.0% | 100.0% | 100.0% | 100.0% | 91.0% |
| calibrated/channel-wise/44 | 100.0% | 99.9% | 99.8% | 99.9% | 100.0% | 99.9% | 99.9% | 100.0% | 100.0% | 99.9% | 100.0% | 100.0% |
| calibrated/scalar/42 | 100.0% | 100.0% | 99.5% | 99.3% | 97.3% | 94.1% | 94.6% | 99.8% | 99.9% | 100.0% | 100.0% | 93.3% |
| calibrated/scalar/43 | 100.0% | 100.0% | 99.9% | 100.0% | 99.9% | 99.9% | 100.0% | 100.0% | 100.0% | 99.9% | 100.0% | 99.8% |
| calibrated/scalar/44 | 99.9% | 99.9% | 97.1% | 99.9% | 98.5% | 97.5% | 95.0% | 99.9% | 100.0% | 99.9% | 100.0% | 97.3% |
| default/channel-wise/42 | 99.9% | 100.0% | 99.8% | 99.9% | 99.9% | 99.9% | 99.9% | 100.0% | 100.0% | 100.0% | 99.9% | 100.0% |
| default/channel-wise/43 | 100.0% | 100.0% | 99.4% | 100.0% | 99.9% | 99.6% | 99.8% | 100.0% | 100.0% | 100.0% | 100.0% | 99.6% |
| default/channel-wise/44 | 100.0% | 100.0% | 99.7% | 99.5% | 99.8% | 99.6% | 99.4% | 100.0% | 100.0% | 100.0% | 100.0% | 99.6% |
| default/scalar/42 | 99.8% | 100.0% | 99.9% | 99.8% | 99.3% | 99.6% | 99.2% | 100.0% | 99.9% | 99.9% | 100.0% | 99.4% |
| default/scalar/43 | 99.8% | 99.9% | 99.8% | 99.8% | 99.7% | 99.6% | 99.7% | 99.9% | 100.0% | 99.8% | 100.0% | 99.6% |
| default/scalar/44 | 100.0% | 100.0% | 75.3% | 89.5% | 91.4% | 90.6% | 89.4% | 100.0% | 100.0% | 99.9% | 100.0% | 88.9% |
| published64/channel-wise/42 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| published64/scalar/42 | 100.0% | 99.9% | 99.9% | 100.0% | 99.9% | 99.9% | 99.9% | 100.0% | 100.0% | 99.9% | 100.0% | 99.9% |

## Error attribution (all models)

| Cell | Errors | Another memory value | Position-anchored |
|---|---:|---:|---:|
| zoology_baseline | 18 | 18 | 0 |
| shift_0 | 12 | 12 | 0 |
| shift_1 | 696 | 664 | 1 |
| shift_2 | 403 | 362 | 136 |
| shift_4 | 536 | 467 | 0 |
| shift_8 | 615 | 535 | 0 |
| shift_16 | 670 | 595 | 0 |
| near | 15 | 15 | 0 |
| far | 8 | 8 | 0 |
| memory_order | 24 | 24 | 0 |
| reverse_order | 7 | 7 | 0 |
| shift_8_far | 661 | 557 | 0 |
