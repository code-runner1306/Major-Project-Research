# Pilot-based training time estimate

Planned budget: **1,000 optimizer steps per variant/seed**.
Times below are training-only and include scalar plus channel-wise variants.

## Per model, dataset, and seed

| Model | Dataset | Scalar step p50 (s) | Channel-wise step p50 (s) | Scalar hours/seed | Channel-wise hours/seed | Both variants hours/seed | Timing source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 10M | tinystories | 8.11 | 8.03 | 2.25 | 2.23 | 4.48 | data_fed_p50 |
| 10M | mqar | 9.51 | 9.65 | 2.64 | 2.68 | 5.32 | data_fed_p50 |
| 10M | overwrite | 9.45 | 9.50 | 2.63 | 2.64 | 5.27 | data_fed_p50 |
| **10M total** | all datasets | — | — | — | — | 15.07 h/seed | 5 configured seed(s): 75.36 h |
| 30M | tinystories | 13.62 | 13.61 | 3.78 | 3.78 | 7.56 | data_fed_p50 |
| 30M | mqar | 17.44 | 18.92 | 4.84 | 5.26 | 10.10 | data_fed_p50 |
| 30M | overwrite | 16.75 | 17.10 | 4.65 | 4.75 | 9.40 | data_fed_p50 |
| **30M total** | all datasets | — | — | — | — | 27.07 h/seed | 5 configured seed(s): 135.34 h |
| 55M | tinystories | 34.77 | 43.76 | 9.66 | 12.16 | 21.82 | data_fed_p50 |
| 55M | mqar | 71.70 | 84.42 | 19.92 | 23.45 | 43.37 | data_fed_p50 |
| 55M | overwrite | 73.40 | 84.56 | 20.39 | 23.49 | 43.88 | data_fed_p50 |
| **55M total** | all datasets | — | — | — | — | 109.06 h/seed | 1 configured seed(s): 109.06 h |

## Overall totals across selected model sizes

| Seeds per model size | Sequential training hours | Estimated hours using measured parallel speedups | Sequential days, one GPU |
| ---: | ---: | ---: | ---: |
| 1 | 151.2 | 145.4 | 6.3 |
| 3 | 453.6 | 436.1 | 18.9 |
| 5 | 756.0 | 726.8 | 31.5 |

## Configured seed plan

| Model | Seeds | Estimated training hours |
| --- | ---: | ---: |
| 10M | 5 | 75.4 |
| 30M | 5 | 135.3 |
| 55M | 1 | 109.1 |
| **Total** | — | **319.8** |

Using measured pilot speedups as an idealized schedule gives **290.6 hours** for the configured seed plan (about **29.2 hours saved**). This excludes evaluation and assumes the pilot speedups hold across full runs.

## Low-tier parallel-training pilot

These trials use data-fed short optimizer workloads. VRAM-unsafe worker counts are skipped before model allocation.

| Model | Dataset | Recommended workers | Workers tested (status / speedup / process CPU / system CPU / GPU utilization) |
| --- | --- | ---: | --- |
| 10M | tinystories | 1 | 2: skipped_vram_capacity (9.4 GiB estimated)<br>3: skipped_vram_capacity (14.2 GiB estimated) |
| 10M | mqar | 3 | 2: speedup 1.90x; CPU proc/system 181.37560055610044%/64.96363636363637%; CPU clock 4200.0 MHz; GPU 68.5%<br>3: speedup 2.21x; CPU proc/system 241.38620629338203%/90.09285714285714%; CPU clock 4200.0 MHz; GPU 70.28571428571429% |
| 10M | overwrite | 3 | 2: speedup 1.83x; CPU proc/system 181.09666106855326%/65.05652173913043%; CPU clock 4200.0 MHz; GPU 65.82608695652173%<br>3: speedup 2.25x; CPU proc/system 241.81777556477437%/93.05714285714285%; CPU clock 4200.0 MHz; GPU 67.0% |
| 30M | tinystories | 1 | 2: skipped_vram_capacity (16.6 GiB estimated)<br>3: skipped_vram_capacity (25.0 GiB estimated) |
| 30M | mqar | 1 | 2: skipped_vram_capacity (13.6 GiB estimated)<br>3: skipped_vram_capacity (20.4 GiB estimated) |
| 30M | overwrite | 1 | 2: skipped_vram_capacity (13.6 GiB estimated)<br>3: skipped_vram_capacity (20.4 GiB estimated) |

CPU process utilization may exceed 100% when multiple cores are busy. High CPU/system utilization with weak throughput gain indicates likely CPU/input contention.
Current CPU frequency is sampled when psutil exposes it, but thermal throttle flags/frequency caps are not directly read; this is a utilization-based contention check, not a definitive thermal-throttling diagnosis. Recommendations apply only to the tested GPU and data pipeline.

## Estimate limits

- Times extrapolate short pilot p50 optimizer-step measurements to the planned fixed step budget.
- Training totals include both configured gate variants for every dataset and seed.
- Measured data-fed timings include input fetch and host-to-device work for each family when available.
- Resident-compute fallbacks omit dataset input overhead and are labeled as proxies.
- Pilot measurements use very few steps; treat estimates as planning ranges, not guarantees.
- Startup, checkpoint writes, final evaluation, efficiency benchmarks, and report generation are excluded.
- Seed-count totals are sequential training totals; parallel wall-time savings are reported separately from the bounded concurrency pilot.
