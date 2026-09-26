"""Barra-style stock-level risk model: Sigma = X F X' + D.

X = latest-date factor exposures (Symbol x DESIGN_COLS, the same z-scored/industry-weight
panel the regression itself used), F = the EWMA factor covariance matrix
(output/factor_covariance_latest.csv), D = diagonal specific-risk variance
(output/specific_risk_latest.csv, converted from its annualized units back to the daily units F
is in). Everything here is in daily-return units for consistency with F and the expected
returns (portfolio/expected_returns.py) - annualization, where useful, happens only at the
reporting/analytics layer, never inside the optimizer.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from factor_model import config
from factor_model.regression.cross_sectional import DESIGN_COLS

logger = logging.getLogger(__name__)


def get_latest_exposures(exposure_panel: pd.DataFrame) -> pd.DataFrame:
    """Latest-date factor exposures, Symbol-indexed, restricted to symbols with complete
    (non-NaN) exposure on every one of the 20 design factors - a stock missing any (e.g. a very
    recent IPO whose beta/idio_vol rolling window hasn't cleared yet) can't be risk-modeled and
    is excluded from the optimizable universe, logged for visibility.
    """
    latest_date = exposure_panel["Date"].max()
    latest = exposure_panel.loc[exposure_panel["Date"] == latest_date].set_index("Symbol")[DESIGN_COLS]

    complete = latest.dropna()
    excluded = sorted(set(latest.index) - set(complete.index))
    if excluded:
        logger.warning(
            "Excluding %d symbols from the optimizable universe (incomplete factor exposure "
            "on the latest date %s, e.g. a too-recent IPO): %s",
            len(excluded),
            latest_date.date(),
            excluded,
        )
    return complete


def build_stock_covariance(
    exposures: pd.DataFrame, factor_covariance: pd.DataFrame, specific_risk: pd.DataFrame
) -> pd.DataFrame:
    """Sigma (daily units), Symbol x Symbol, for the symbols in `exposures`.

    Stocks in `exposures` without a specific-risk estimate (e.g. insufficient residual history
    for the EWMA specific-variance calculation) are dropped rather than given a
    fabricated/imputed value - see portfolio/validation.py's coverage check.
    """
    spec = specific_risk.set_index("Symbol")["specific_risk_annualized"]
    symbols = [s for s in exposures.index if s in spec.index and pd.notna(spec.loc[s])]
    missing_spec = sorted(set(exposures.index) - set(symbols))
    if missing_spec:
        logger.warning("Excluding %d symbols with no specific-risk estimate: %s", len(missing_spec), missing_spec)

    X = exposures.loc[symbols, DESIGN_COLS].to_numpy()
    F = factor_covariance.loc[DESIGN_COLS, DESIGN_COLS].to_numpy()
    factor_risk = X @ F @ X.T

    daily_specific_var = (spec.loc[symbols].to_numpy() ** 2) / config.TRADING_DAYS_PER_YEAR
    Sigma = factor_risk + np.diag(daily_specific_var)

    # Symmetrize away any floating-point asymmetry from the matrix products above.
    Sigma = (Sigma + Sigma.T) / 2
    return pd.DataFrame(Sigma, index=symbols, columns=symbols)
