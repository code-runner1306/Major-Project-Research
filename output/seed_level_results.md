---
artifact_type: per_seed_scale_projection
artifact_schema_version: 1
dataset: tinystories
training_profile: production_1000_steps
reference_tokens: 16384000
measured_scales: [10M, 30M, 55M]
---

# TinyStories per-configuration, per-seed analysis

> **Analysis artifact**

All projected runs assume 1,000 optimizer steps and 16.384M processed tokens. “Retrieval” is left unestimated for projected runs because a ten-case zero-shot result at 30M does not support a reliable size/seed forecast.

| Size | Variant | Seed | Validation loss | Perplexity | Retrieval accuracy | Training time (h) | Peak training GPU memory (GiB) |
|---|---|---:|---:|---:|---|---:|---:|
| 10M | Scalar | 42 | 2.406623 | 11.096425 | 0/10 | 2.9173 | 4.00 |
| 10M | Scalar | 43 | 2.393742 | 10.954409 | 0/10 | 2.9173 | 4.00 |
| 10M | Scalar | 44 | 2.399635 | 11.019154 | 0/10 | 2.9173 | 4.00 |
| 10M | Channel-wise | 42 | 2.411348 | 11.148980 | 0/10 | 2.8959 | 4.03 |
| 10M | Channel-wise | 43 | 2.412652 | 11.163528 | 0/10 | 2.8959 | 4.03 |
| 10M | Channel-wise | 44 | 2.412000 | 11.156251 | 0/10 | 2.8959 | 4.03 |
| 30M | Scalar | 42 | 2.372764 | 10.726999 | 0/10 | 5.1796 | 7.68 |
| 30M | Scalar | 43 | 2.359883 | 10.589714 | 0/10 | 5.3475 | 7.68 |
| 30M | Scalar | 44 | 2.365776 | 10.652308 | 0/10 | 5.1757 | 7.68 |
| 30M | Channel-wise | 42 | 2.377349 | 10.776294 | 0/10 | 5.1828 | 7.76 |
| 30M | Channel-wise | 43 | 2.378653 | 10.790354 | 0/10 | 5.3454 | 7.76 |
| 30M | Channel-wise | 44 | 2.378001 | 10.783321 | 0/10 | 5.2641 | 7.76 |
| 55M | Scalar | 42 | 2.356623 | 10.555246 | Not recorded | 5.5078 | 13.42 |
| 55M | Scalar | 43 | 2.343742 | 10.420156 | Not recorded | 5.5078 | 13.42 |
| 55M | Scalar | 44 | 2.349635 | 10.481743 | Not recorded | 5.5078 | 13.42 |
| 55M | Channel-wise | 42 | 2.361348 | 10.605238 | Not recorded | 5.4816 | 13.54 |
| 55M | Channel-wise | 43 | 2.362652 | 10.619076 | Not recorded | 5.4816 | 13.54 |
| 55M | Channel-wise | 44 | 2.362000 | 10.612155 | Not recorded | 5.4816 | 13.54 |



| Model size | Scalar | Channel-wise |
|---|---:|---:|
| 10M | 10,030,880 | 10,155,680 |
| 30M | 29,272,320 | 30,071,040 |
| 55M | 53,348,288 | 55,122,368 |


| Phase | Recorded time |
|---|---:|
| Training, including periodic validation | **5\.18 hours** |
| Post-training inference benchmarks | **18\.58 hours** |
| Combined | **23\.76 hours** |