# Gated DeltaNet Gating Granularity Study Summary

## Execution Run ID: 86ee2ecb
*Device: `cpu`*

### Aggregated Performance (Mean ± Std Dev)

| Scale   | Variant      |   Parameters | Val Perplexity     | Throughput (tok/s)   | Retrieval Acc (%)   | Peak Memory (MB)   | MQAR Acc (%)   |
|:--------|:-------------|-------------:|:-------------------|:---------------------|:--------------------|:-------------------|:---------------|
| 100M    | channel-wise |  100,817,280 | 21043.87 ± 780.99  | 20.41 ± 2.47         | 0.0% ± 0.0%         | 3614.8 ± 81.3      | 0.0% ± 0.0%    |
| 100M    | scalar       |   96,918,912 | 23711.71 ± 951.49  | 20.02 ± 1.53         | 0.0% ± 0.0%         | 3406.0 ± 219.0     | 0.0% ± 0.0%    |
| 10M     | channel-wise |   10,155,680 | 42085.04 ± 898.29  | 71.98 ± 14.44        | 0.0% ± 0.0%         | 1135.9 ± 31.8      | 0.0% ± 0.0%    |
| 10M     | scalar       |   10,030,880 | 41893.78 ± 876.71  | 68.48 ± 20.62        | 0.0% ± 0.0%         | 1074.9 ± 87.6      | 0.0% ± 0.0%    |
| 30M     | channel-wise |   30,071,040 | 31802.25 ± 1251.39 | 43.13 ± 3.26         | 0.0% ± 0.0%         | 1759.7 ± 108.3     | 0.0% ± 0.0%    |
| 30M     | scalar       |   29,272,320 | 33202.08 ± 950.05  | 42.23 ± 2.87         | 0.0% ± 0.0%         | 1662.8 ± 98.1      | 0.0% ± 0.0%    |
| 55M     | channel-wise |   55,122,368 | 27178.99 ± 2078.69 | 28.78 ± 5.76         | 0.0% ± 0.0%         | 2425.0 ± 69.2      | 0.0% ± 0.0%    |
| 55M     | scalar       |   53,348,288 | 25642.62 ± 2962.26 | 26.34 ± 0.69         | 0.0% ± 0.0%         | 2267.7 ± 91.5      | 0.0% ± 0.0%    |

---
### Scaling Charts Generated:
1. `perplexity_scaling.png` (Validation Perplexity vs. Parameters)
2. `throughput_scaling.png` (Tokens per second vs. Parameters)
3. `retrieval_scaling.png` (Needle retrieval accuracy vs. Parameters)
4. `memory_scaling.png` (Peak Memory in MB vs. Parameters)
5. `mqar_scaling.png` (MQAR accuracy vs. Parameters)
