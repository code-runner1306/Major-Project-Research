# Supplementary primary-source checks from the extended catalog

Checked 17 November 2026. The catalog row indices supplied for this pass did not all resolve to the named titles in the local CSV; I therefore verified the titles by exact-title search and recorded the primary URL used. This file reports abstract-level evidence unless explicitly marked otherwise. It does not claim that every catalog item was screened.

## Exact-title searches and verified sources

### Improving the Gating Mechanism of Recurrent Neural Networks

Query: `"Improving the Gating Mechanism" recurrent neural networks Gu 2019`  
Primary source: [arXiv:1910.09890](https://arxiv.org/abs/1910.09890)  
Authors/year: Albert Gu, Caglar Gulcehre, Tom Le Paine, Matt Hoffman, Razvan Pascanu et al.; 2019 (arXiv record).  
Abstract evidence: the paper argues that ordinary gates become difficult to learn near saturation when long delays require outputs near one. It proposes two synergistic modifications to standard gates, relates them to chrono initialization and Ordered Neurons, and reports improved learnability on synthetic memorization, sequential image classification, language modeling, and reinforcement learning.  
Relevance: gate learnability and long-memory initialization are established concerns before GDN. This is not scalar-versus-channel GDN and does not use our MQAR/overwrite protocol.  
Strength of competing result: **moderate historical competition for a gate-learnability claim; not direct competition for our exact decay-granularity comparison.**

### RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval

Query: `"RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval"`  
Primary sources: [arXiv:2402.18510](https://arxiv.org/abs/2402.18510) and [ICLR 2025 proceedings page](https://proceedings.iclr.cc/paper_files/paper/2025/hash/79dc391a2c1067e9ac2b764e31a60377-Abstract-Conference.html).  
Authors/year: Kaiyue Wen, Xingyu Dang, Kaifeng Lyu; ICLR 2025.  
Abstract evidence: proves a retrieval bottleneck for constant-state recurrent models on associative-recall and related tasks, showing that chain-of-thought alone does not close the representation gap; a retrieval mechanism or one Transformer layer can change the result.  
Relevance: this provides a stronger theoretical competing explanation for recall failures than gate calibration alone: finite-state retrieval capacity can be the bottleneck. It does not study our scalar/channel decay factorial or repeated-write error categories.  
Strength of competing result: **strong for a broad “recurrent memory fails because of retrieval” framing; adjacent rather than a direct experimental duplicate.**

### Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-smoothing

Query: `"Understanding and Mitigating Bottlenecks of State Space Models through the Lens of Recency and Over-smoothing"`  
Primary sources: [arXiv:2501.00658](https://arxiv.org/abs/2501.00658), [ICLR 2025 OpenReview PDF](https://openreview.net/pdf/d141e48dc98fe9f3010faee4ac16776b9f57e5ca.pdf), and [official implementation](https://github.com/VITA-Group/SSM-Bottleneck).  
Authors/year: Peihao Wang, Ruisi Cai, Yuehao Wang, Jiajun Zhu, Pragya Srivastava, Zhangyang Wang, Pan Li; ICLR 2025.  
Abstract/README evidence: reports recency bias and long-range recall degradation in SSMs, an over-smoothing tendency with depth, and a polarization technique that sets two state-transition channels to zero and one. The released MQAR script uses vocabulary 8192 and evaluates varying lengths/key-value counts, including deeper SSMs and polarized controls.  
Relevance: the paper already links state-transition structure to associative recall and tests a fixed channel-level transition intervention. Its transition polarization is not our learned channel-wise decay and does not perform our mean-decay calibration, but it narrows any claim that channel/state dynamics and MQAR diagnosis are unexplored.  
Strength of competing result: **strong adjacent competing mechanism for channel/state-transition interventions and MQAR.**

### Revisiting Associative Recall in Modern Recurrent Models

Query: `"Revisiting associative recall in modern recurrent models"`  
Primary sources: [arXiv:2508.19029](https://arxiv.org/abs/2508.19029), [SCOPE–ICLR 2025 OpenReview PDF](https://openreview.net/pdf/f7e9f322ba15e88dcc818ab70866648650a5e319.pdf).  
Authors/year: Destiny Okpekpe, Antonio Orvieto; SCOPE workshop at ICLR 2025 (arXiv record v1).  
Abstract evidence: compares modern recurrent models and Transformers on associative recall and copying, and reports that recurrent-model success is confined to a narrow learning-rate window while Transformers are more robust. It also studies scaling and optimization effects.  
Relevance: this is directly relevant to our earlier training instability and learning-rate choices. It means a task-performance difference can be an optimizer/recipe effect rather than a gating mechanism effect. We must report the exact LR, schedule, steps, and per-seed results and avoid attributing failures to gates without matched LR controls.  
Strength of competing result: **very strong confound for interpretation of learnability and seed variance; not a direct scalar/channel gate study.**

### Just read twice: closing the recall gap for recurrent language models

Query: `"Just read twice: closing the recall gap for recurrent language models"`  
Primary sources: [arXiv:2407.05483](https://arxiv.org/abs/2407.05483), [ICLR 2025 OpenReview PDF](https://openreview.net/pdf?id=RsqCqziLAt).  
Authors/year: Simran Arora, Aman Timalsina, Aaryan Singhal, Benjamin Spector, Sabri Eyuboglu, Xinyi Zhao, Ashish Rao, Atri Rudra, Christopher Ré; 2024 preprint / ICLR 2025.  
Abstract evidence: attributes a recurrent-language-model recall gap partly to causal data order and set-disjointness difficulty. It proposes repeating the prompt or a non-causal prefix-linear-attention architecture; the paper reports substantial recall/throughput improvements in its large-model setting.  
Relevance: repeated reads or query-aware ordering are an alternative explanation/intervention for recall failures. Our experiments are causal one-pass synthetic memory tests, so they do not test this remedy. It weakens claims that gate changes alone are the only route to improved recall.  
Strength of competing result: **strong adjacent result for causal-order and protocol effects; not a decay-granularity comparison.**

## Interpretation update

These checks add three important competing explanations to the project’s failure analysis:

1. Finite-state retrieval capacity can limit performance even if gate statistics are well calibrated ([RNNs are not Transformers](https://arxiv.org/abs/2402.18510)).
2. State-transition structure and recency/over-smoothing can drive MQAR failures independently of our scalar/channel distinction ([SSM Bottlenecks](https://arxiv.org/abs/2501.00658)).
3. Learning-rate windows and causal data order can dominate apparent learnability and recall differences ([Revisiting AR](https://arxiv.org/abs/2508.19029), [Just Read Twice](https://arxiv.org/abs/2407.05483)).

Accordingly, the paper should treat calibration and gate granularity as factors in a controlled recipe, not as automatically causal explanations. The minimum defensible follow-up includes LR stability checks, matched data/order controls, and an explicit capacity baseline. No numeric result from these sources is copied here because only abstract/official README evidence was inspected for the supplementary checks.

## Limits

The exact catalog indices supplied by the parent task were not reproducible against the current local CSV row numbering; title-based verification was used. The five primary records above were located and their abstracts or official README/proceedings metadata were inspected. Full-paper claims, exact tables, and all cited references require separate reading before a final paper bibliography or quantitative meta-analysis.
