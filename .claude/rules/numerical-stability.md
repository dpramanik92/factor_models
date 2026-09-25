---
scope: src/factor_model/**
---

# Numerical stability rules

- Always winsorize before z-scoring (see `.claude/skills/statistics/SKILL.md`) - z-scoring first
  lets outliers distort the mean/std used for scoring, defeating the point.
- Guard every ratio denominator that can be zero or near-zero: P/E, P/B (price ratios), D/E
  (equity can be ~0 or negative for distressed companies), ROA/ROE (assets/equity denominators).
  Prefer producing `NaN` over raising, and prefer producing `NaN` over silently clipping to a
  large sentinel value.
- Cross-sectional aggregations (mean, std, z-score, winsorize) must be NaN-safe: missing inputs
  stay missing in the output, they are never silently imputed to 0 or the cross-sectional mean at
  the factor-construction stage. Imputation, if ever needed, is a validation-stage decision, not
  a factor-construction one.
- When forward-filling annual fundamentals to a daily panel, forward-fill only - never
  interpolate or backfill, since that would leak future information into past dates.
