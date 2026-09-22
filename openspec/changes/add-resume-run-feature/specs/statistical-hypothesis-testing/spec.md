## ADDED Requirements

### Requirement: Wilcoxon signed-rank and parametric hypothesis testing
The pipeline SHALL compute non-parametric Wilcoxon signed-rank tests (and paired t-tests) comparing Scalar Gating (Variant A) vs. Channel-wise Gating (Variant B) across seeds for each parameter scale (10M, 30M, 55M, 100M).

#### Scenario: Aggregating multi-seed metrics for statistical testing
- **WHEN** all seeds for Variant A and Variant B at a specific scale finish execution
- **THEN** the pipeline runs `scipy.stats.wilcoxon` and `scipy.stats.ttest_rel` on validation perplexity, throughput, needle retrieval accuracy, and MQAR accuracy, outputting $p$-values, test statistics, and 95% confidence intervals to `statistical_analysis.json`.

### Requirement: Effect size calculation (Cohen's d)
The pipeline SHALL calculate Cohen's $d$ effect sizes for all metric comparisons between Variant A and Variant B at each parameter scale to quantify the magnitude of difference independently of sample size.

#### Scenario: Reporting effect sizes in statistical analysis
- **WHEN** statistical evaluation runs for a given metric across variants
- **THEN** the output report includes Cohen's $d$ along with mean difference and percentage relative gain/loss.

### Requirement: Comprehensive statistical summary artifact
The orchestrator SHALL generate a dedicated markdown document `statistical_report.md` detailing all hypothesis test results, $p$-values, confidence intervals, and statistical significance flags ($p < 0.05$).

#### Scenario: End-of-grid statistical reporting
- **WHEN** all 24 grid search runs complete or are restored from cache
- **THEN** `statistical_report.md` is compiled and written to the output directory containing a comprehensive summary table with significance indicators.
