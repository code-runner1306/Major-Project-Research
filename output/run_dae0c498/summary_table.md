# Gated DeltaNet Gating Granularity Study Summary

## Execution Run ID: dae0c498
*Device: `cpu`*

### Aggregated Performance (Mean ± Std Dev)

| Scale   | Variant      |   Parameters | Val Perplexity     | Throughput (tok/s)   | Retrieval Acc (%)   |
|:--------|:-------------|-------------:|:-------------------|:---------------------|:--------------------|
| 100M    | channel-wise |  100,817,280 | 21043.87 ± 780.99  | 19.90 ± 0.53         | 0.0% ± 0.0%         |
| 100M    | scalar       |   96,918,912 | 23711.71 ± 951.49  | 20.46 ± 1.81         | 0.0% ± 0.0%         |
| 10M     | channel-wise |   10,155,680 | 42085.04 ± 898.29  | 90.35 ± 2.68         | 0.0% ± 0.0%         |
| 10M     | scalar       |   10,030,880 | 41893.78 ± 876.71  | 79.33 ± 16.75        | 0.0% ± 0.0%         |
| 30M     | channel-wise |   30,071,040 | 31802.25 ± 1251.39 | 37.14 ± 2.65         | 0.0% ± 0.0%         |
| 30M     | scalar       |   29,272,320 | 33202.08 ± 950.05  | 42.36 ± 2.22         | 0.0% ± 0.0%         |
| 55M     | channel-wise |   55,122,368 | 27178.99 ± 2078.69 | 26.08 ± 7.04         | 0.0% ± 0.0%         |
| 55M     | scalar       |   53,348,288 | 25642.62 ± 2962.26 | 27.38 ± 4.31         | 0.0% ± 0.0%         |

---
### Scaling Charts Generated:
1. `perplexity_scaling.png` (Validation Perplexity vs. Parameters)
2. `throughput_scaling.png` (Tokens per second vs. Parameters)
3. `retrieval_scaling.png` (Needle retrieval accuracy vs. Parameters)
