"""Save cross-sectional regression outputs to output/ (mirrored into data/ - io/writers.py)."""

from __future__ import annotations

import logging

import pandas as pd

from factor_model import config
from factor_model.io.writers import save_csv, save_parquet
from factor_model.regression.cross_sectional import DESIGN_COLS, CrossSectionalResults
from factor_model.regression.newey_west import summarize_factor_returns

logger = logging.getLogger(__name__)


def save_regression_results(results: CrossSectionalResults) -> pd.DataFrame:
    """Writes factor_returns_daily.csv, r2_daily.csv, residuals_long.parquet, summary.csv.

    Returns the summary DataFrame (Factor, Type, Mean_coef, Naive_t, NW_SE, NW_t,
    Pct_days_positive, N_days) - the headline result printed to chat.
    """
    factor_returns_out = results.factor_returns.reset_index().rename(columns={"index": "Date"})
    factor_returns_out = results.r2_daily[["Date", "N"]].merge(factor_returns_out, on="Date")
    save_csv(factor_returns_out, config.FACTOR_RETURNS_FILE, index=False)
    save_csv(results.r2_daily, config.R2_DAILY_FILE, index=False)
    save_parquet(results.residuals_long, config.RESIDUALS_LONG_FILE)

    if not results.flagged_corporate_actions.empty:
        save_csv(results.flagged_corporate_actions, config.OUTPUT_DIR / "flagged_corporate_actions.csv", index=False)

    summary = summarize_factor_returns(results.factor_returns)
    summary["Type"] = summary["Factor"].apply(lambda f: "Industry" if f in config.INDUSTRY_FACTORS else "Style")
    summary = summary[["Factor", "Type", "Mean_coef", "Naive_t", "NW_SE", "NW_t", "Pct_days_positive", "N_days"]]
    summary["Factor"] = pd.Categorical(summary["Factor"], categories=DESIGN_COLS, ordered=True)
    summary = summary.sort_values("Factor").reset_index(drop=True)
    save_csv(summary, config.SUMMARY_FILE, index=False)

    logger.info("Saved regression outputs to %s", config.OUTPUT_DIR)
    return summary
