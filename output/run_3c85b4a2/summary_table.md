# Gated DeltaNet Gating Granularity Study Summary

## Execution Run ID: 3c85b4a2
*Device: `cpu`*

### Aggregated Performance (Mean ± Std Dev)

| Scale   | Variant      |   Parameters | Val Perplexity     | Throughput (tok/s)   | Retrieval Acc (%)   | Peak Memory (MB)   | MQAR Acc (%)   |
|:--------|:-------------|-------------:|:-------------------|:---------------------|:--------------------|:-------------------|:---------------|
| 100M    | channel-wise |  100,817,280 | 21172.48 ± 678.60  | 20.68 ± 3.07         | 0.0% ± 0.0%         | 3141.74 ± 50.68    | 0.0% ± 0.0%    |
| 100M    | scalar       |   96,918,912 | 23836.29 ± 1032.43 | 20.47 ± 2.46         | 0.0% ± 0.0%         | 3066.39 ± 44.44    | 0.0% ± 0.0%    |
| 10M     | channel-wise |   10,155,680 | 42528.78 ± 805.44  | 75.30 ± 20.09        | 0.0% ± 0.0%         | 1324.35 ± 198.89   | 0.0% ± 0.0%    |
| 10M     | scalar       |   10,030,880 | 42311.19 ± 740.81  | 77.95 ± 19.84        | 0.0% ± 0.0%         | 1172.04 ± 27.62    | 0.0% ± 0.0%    |
| 30M     | channel-wise |   30,071,040 | 32147.10 ± 1319.35 | 38.95 ± 7.48         | 0.0% ± 0.0%         | 1722.30 ± 113.62   | 0.0% ± 0.0%    |
| 30M     | scalar       |   29,272,320 | 33604.37 ± 1008.54 | 49.36 ± 6.11         | 0.0% ± 0.0%         | 1773.90 ± 195.30   | 0.0% ± 0.0%    |
| 55M     | channel-wise |   55,122,368 | 27386.95 ± 2314.18 | 32.98 ± 2.05         | 0.0% ± 0.0%         | 2439.93 ± 111.58   | 0.0% ± 0.0%    |
| 55M     | scalar       |   53,348,288 | 26003.60 ± 2995.53 | 21.48 ± 12.00        | 0.0% ± 0.0%         | 2395.29 ± 23.44    | 0.0% ± 0.0%    |

---
### Scaling Charts Generated:
1. `perplexity_scaling.png` (Validation Perplexity vs. Parameters)
2. `throughput_scaling.png` (Tokens per second vs. Parameters)
3. `retrieval_scaling.png` (Needle retrieval accuracy vs. Parameters)
4. `memory_scaling.png` (Peak Memory in MB vs. Parameters)
5. `mqar_scaling.png` (MQAR accuracy vs. Parameters)
