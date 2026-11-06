# V100 training pilot

This is a profiling artifact, not a research training result.

- Backend: `reference`
- Device: `NVIDIA RTXA6000-12Q` / `(8, 6)`
- Warmup / measured steps: 1 / 2

| Scale | Variant | GPU p50 s/step | Wall p50 s/step | Tokens/sec | Peak GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 10M | scalar | 8.1198 | 8.1208 | 2017.5 | 4.00 |
| 10M | channel-wise | 7.9142 | 7.9152 | 2069.9 | 4.03 |
| 30M | scalar | 14.8104 | 14.8116 | 1106.2 | 7.68 |
| 30M | channel-wise | 13.8756 | 13.8768 | 1180.7 | 7.76 |
| 55M | scalar | 34.8826 | 34.8837 | 469.7 | 13.41 |
| 55M | channel-wise | 43.1917 | 43.1928 | 379.3 | 13.54 |

Interpret wall-minus-GPU time as host/data/launch overhead. Compare scalar and channel-wise only on this same artifact/device.

See `training_time_estimate.md` for per-dataset extrapolations and 1/3/5-seed totals.
