# Gating experiment teaching resources

## Knowledge

- [Our calibration implementation](recipe_calibration.py)
  Primary source for what parameters we fit, which training inputs we use, and
  what a passed calibration check means.
- [Our model implementation](recipe_model.py)
  Primary source for the actual decay formula and scalar/channel parameter shapes.
- [Current methodology](RESEARCH_METHODOLOGY.md)
  Human-readable parameter counts, target choices and limitations.
- [Two-seed overwrite review](research_notes/calibrated_overwrite_two_seed_review_2026-11-16.md)
  Checked development results; use for separating fitted initialization from task outcomes.
- [Yang et al., Gated Delta Networks](https://arxiv.org/abs/2412.06464)
  Original architecture paper explaining the complementary roles of forgetting
  gates and delta-rule updates. Our matched-mean calibration targets are an
  experimental extension, not a result established by this paper.

## Wisdom (Communities)

No community activity is needed to answer the current implementation question.

## Gaps

The mechanism behind the seed-dependent learning changes remains unestablished.
The learner has requested an explanation but has not yet demonstrated mastery.
