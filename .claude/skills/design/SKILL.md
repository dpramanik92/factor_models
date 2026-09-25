---
name: design
description: Output/report design conventions for this repo's generated artifacts - CSV/parquet schema naming, validation_report.md structure, and diagnostic plot style. Use when writing anything under output/ (factor returns, summaries, the validation report, plots). Not related to chat-facing visualizations - see the dataviz skill for those.
---

# Output and report design conventions

This governs `output/` artifacts only - internal engineering/research deliverables, not
chat-facing visualizations (use the `dataviz` skill for anything rendered in an artifact or chat).

## File/schema naming

Match the naming conventions already established by prior work found in `data/fama_macbeth_*`
and `data/cross_sectional_regression_*` for continuity (not reuse of the files themselves):

- `output/factor_returns_daily.csv`: `Date, N, <industry_name>_coef..., <style_factor>_coef...`
- `output/r2_daily.csv`: `Date, N, R2, Adj_R2`
- `output/residuals_long.parquet`: `Date, Symbol, ActualReturn, Fitted, Residual`
- `output/summary.csv`: `Factor, Type, Mean_coef, Naive_t, NW_SE, NW_t, Pct_days_positive, N_days`
  (`Type` is `Industry` or `Style`)
- `output/factor_covariance_latest.csv`: K x K matrix, factor names as both row and column index
- `output/validation_report.md`: see structure below

## `validation_report.md` structure

1. **Executive summary** - a single pass/fail table, one row per test, at the very top. A reader
   should be able to tell if the model is sound from this table alone.
2. One section per test category (regression fit, multicollinearity, residual diagnostics,
   look-ahead-bias check, factor stationarity, size-quintile spread), each with: the statistic
   computed, the threshold/criterion, the verdict, and a one-line interpretation.
3. Supporting plots referenced inline, saved to `output/plots/`.

## Plot style conventions

Plain `matplotlib` (these are internal diagnostics, not polished chat artifacts). Keep it
consistent and readable:
- Same color per factor across every plot it appears in.
- Visually distinguish industry factors from style factors (e.g. different marker/linestyle),
  since they play different roles in the regression (industry = intercept substitute, style =
  actual descriptors).
- Dates on the x-axis in ISO format; coefficients/returns as percentages or bps, not raw
  decimals, with 2 decimal places on t-stats.

## Numbers in chat

When reporting cross-sectional regression results or validation findings back to the user
directly in chat (not just writing the files), lead with the headline numbers (mean R²,
how many factors are significant at the NW 5% level, any failed validation checks) rather than
dumping full tables - point to the `output/` file for the detail.
