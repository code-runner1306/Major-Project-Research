## ADDED Requirements

### Requirement: Error-bar scaling charts with statistical significance markers
The plotting module SHALL generate scaling plots (Perplexity, Throughput, Retrieval Accuracy, Peak Memory, MQAR) with standard deviation error bars (or 95% confidence intervals) and statistical significance annotations ($*, **, \text{ns}$) for comparisons between Variant A and Variant B.

#### Scenario: Rendering scaling plots with error bars
- **WHEN** `generate_plots` is called in `run_all.py`
- **THEN** line charts include `yerr` error bars showing multi-seed variation and visual markers indicating statistically significant differences between gating variants.

### Requirement: Step-wise training loss convergence trajectories
The system SHALL aggregate step-by-step training loss histories across seeds for each configuration and render step-wise loss convergence plots comparing Variant A vs. Variant B across training steps.

#### Scenario: Visualizing convergence dynamics
- **WHEN** training completes across seeds for a scale
- **THEN** a `loss_convergence_<scale>.png` figure is rendered displaying the mean loss curve with shaded standard deviation bands across training steps.
