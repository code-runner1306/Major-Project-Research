# Manuscript review and resolutions

| Issue | Verdict | Resolution |
|---|---|---|
| Fresh overwrite versus stress result | Valid clarity issue. The two results are compatible. | Table I covers fresh single-rewrite examples and scores both queries. The stress suite uses six baseline and shift conditions plus three paired update diagnostics, and includes an additional initialization. Channel-wise leads in 8/9 pairs on latest-value accuracy and 5/9 if both answers are scored on those same cells. |
| Retention timescales | Valid scope issue. The original suite already reached 128 tokens, but did not extend far beyond training. | Added a paired 64, 256, and 1,024 token MQAR evaluation of 12 frozen checkpoints. The sweep exposes sharp accuracy losses in one default channel-wise checkpoint and two mean-decay scalar checkpoints. It fixes task load and leaves mixed retention under continuing writes open. |
| Resource section | Valid presentation issue. Parameter count and timing remain useful within their scope. | Kept parameter count, state size, and sequential A6000 step timing. Removed single-session T4 values, unavailable rows, and NVML discussion. Moved resource reporting out of the contribution list. |
| Kernel efficiency | Valid missing context. A blanket claim that vector gates require secondary tiling or that scalar GDN is always faster goes beyond the evidence. | Explained Mamba-2's scalar transition and SSD factorization, GLA's use of secondary tiling, and the distinction between decay algebra and GDN's delta correction. |
| Unrun model size | Valid editorial issue. | Removed the mentions from the manuscript and figures. |
| RWKV-7, GLA, and Mamba-2 | Valid related-work gap. RWKV-7 is a precedent with a different update, so calling it the identical arm would be inaccurate. | Added RWKV-7 and framed the scalar/vector contrast explicitly. |
| E1 practical effect | Valid interpretation issue. | Reported the 1.68-point mean gap alongside the 10/10 lock-in tie and the test used. |
| Missing joint condition in Table I | Valid scope ambiguity. | Explained that joint initialization entered the stress suite but was outside the fresh confirmation matrix. |
| Fig. 1 notation | Valid technical error. | Matched the decay matrix and right multiplication to Eq. (1). |
| Parameter formula | Valid notation issue. The original count was numerically correct for equal widths. | Distinguished residual input width from key output width and retained the 66,040 difference. |
| Training details and update count | Valid reproducibility issue. | Added optimizer settings, FP32, batch and microbatch sizes, learning rate, cosine horizon, no warmup, decay exemptions, and what counts as an update. |
| Table III chance level | Valid interpretation issue. Identical rounded scores alone do not establish chance. | Added the harness's 1/54 value-marginal reference for E4 and E5. |
| Overwrite layout paragraph | Valid placement issue. | Moved it to a short Results subsection. |
| TinyStories | Valid concern about inferential weight. | Kept one descriptive sentence and its sample sizes. |
| Fig. 2 | Valid space concern. | Reduced the workflow to four compact evaluation cohorts and made their sample scopes explicit. |
| Fig. 3 | Valid design concern. | Replaced aggregate charts and table text with paired points for the synthetic stress suite and Anatomy cells. |
| Checkpoint-intervention flow | Valid editorial issue. | Removed the disconnected initialization sentence and clarified the contrast between intervention types. |
| Abstract | Valid editorial issue. | Condensed the findings, omitted unavailable measurements, and ended with the central conclusion. |
| Zoology venue | Valid metadata correction. | The [published paper](https://openreview.net/attachment?id=LY3ukUANko&name=pdf) identifies ICLR 2024. |
| Das reference | The paper is real. The URL concern is valid. | The [TMLR digest](https://groups.google.com/g/tmlr-announce-weekly/c/Q1idbmozPVM) identifies the same OpenReview submission under an earlier title. The indexed [TMLR listing](https://tmlr.infinite-conf.org/paper_pages/FdXB4u27LD.html) confirms the cited title and author. The bibliography uses the canonical OpenReview forum URL. |
| 2026 references | Verification request, with no ID or author mismatch found. | Checked [GDN-2](https://arxiv.org/abs/2605.22791), [Anatomy](https://arxiv.org/abs/2609.16183), and [KATA](https://arxiv.org/abs/2607.17419) against their arXiv records. |

The length-extension evaluator and raw measurements are in the repository. It reuses the saved 16-epoch synthetic checkpoints and audits every generated query against the stored associations.
