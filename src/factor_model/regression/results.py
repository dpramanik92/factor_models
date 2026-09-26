"""Save cross-sectional regression outputs to output/ (mirrored into data/ - io/writers.py)."""

from __future__ import annotations

import logging

import pandas as pd

from factor_model import config
from factor_model.io.writers import save_csv, save_parquet
from factor_model.regression.cross_sectional import DESIGN_COLS, CrossSectionalResults
from factor_model.regression.newey_west import summarize_factor_returns

logger = logging.getLogger(__name__)


def save_regression_results(
    results: CrossSectionalResults, paths: config.OutputPaths = config.DEFAULT_OUTPUT_PATHS
) -> pd.DataFrame:
    """Writes factor_returns_daily.csv, r2_daily.csv, residuals_long.parquet, summary.csv under
    paths.output_dir (default output/; pass a period-scoped OutputPaths to avoid overwriting the
    full-window result - see config.MODEL_PERIODS).

    Returns the summary DataFrame (Factor, Type, Mean_coef, Naive_t, NW_SE, NW_t,
    Pct_days_positive, N_days) - the headline result printed to chat.
    """
    factor_returns_out = results.factor_returns.reset_index().rename(columns={"index": "Date"})
    factor_returns_out = results.r2_daily[["Date", "N"]].merge(factor_returns_out, on="Date")
    save_csv(factor_returns_out, paths.factor_returns_file, index=False)
    save_csv(results.r2_daily, paths.r2_daily_file, index=False)
    save_parquet(results.residuals_long, paths.residuals_long_file)

    if not results.flagged_corporate_actions.empty:
        save_csv(results.flagged_corporate_actions, paths.flagged_corporate_actions_file, index=False)

    summary = summarize_factor_returns(results.factor_returns)
    def _factor_type(f: str) -> str:
        if f in config.INDUSTRY_FACTORS:
            return "Industry"
        if f in config.INDICATOR_FACTORS:
            return "Indicator"
        return "Style"

    summary["Type"] = summary["Factor"].apply(_factor_type)
    summary = summary[["Factor", "Type", "Mean_coef", "Naive_t", "NW_SE", "NW_t", "Pct_days_positive", "N_days"]]
    summary["Factor"] = pd.Categorical(summary["Factor"], categories=DESIGN_COLS, ordered=True)
    summary = summary.sort_values("Factor").reset_index(drop=True)
    save_csv(summary, paths.summary_file, index=False)

    logger.info("Saved regression outputs to %s", paths.output_dir)
    return summary
