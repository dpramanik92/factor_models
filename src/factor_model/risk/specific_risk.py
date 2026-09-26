"""Per-stock EWMA specific (idiosyncratic) risk from regression residuals."""

from __future__ import annotations

import logging

import pandas as pd

from factor_model import config
from factor_model.io.writers import save_csv

logger = logging.getLogger(__name__)


def compute_ewma_specific_risk(
    residuals_long: pd.DataFrame, halflife: int = config.EWMA_HALFLIFE_SPECIFIC
) -> pd.DataFrame:
    """Latest EWMA specific-return std dev per symbol, annualized."""
    wide = residuals_long.pivot(index="Date", columns="Symbol", values="Residual").sort_index()
    min_periods = min(2 * halflife, max(len(wide) - 1, 1))
    ewma_var = wide.ewm(halflife=halflife, min_periods=min_periods).var()
    latest_var = ewma_var.iloc[-1]
    latest_std_annualized = (latest_var * config.TRADING_DAYS_PER_YEAR) ** 0.5
    out = latest_std_annualized.rename("specific_risk_annualized").reset_index()
    out.columns = ["Symbol", "specific_risk_annualized"]
    return out.sort_values("Symbol").reset_index(drop=True)


def save_specific_risk(specific_risk: pd.DataFrame, paths: config.OutputPaths = config.DEFAULT_OUTPUT_PATHS) -> None:
    save_csv(specific_risk, paths.specific_risk_file, index=False)
    logger.info("Saved specific risk (%d symbols) to %s", len(specific_risk), paths.specific_risk_file)
