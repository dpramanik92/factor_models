"""Macro-momentum expected-return tilts for two specific stock groups, found via alpha research
(CLAUDE.md's "Alpha research" section): Oil-sector returns' sensitivity to oil-price momentum,
and IT-exporter returns' sensitivity to USD/INR momentum. Both are additive daily-return tilts
applied only to their own group's member stocks in portfolio/pipeline.py and portfolio/backtest.py
- everything else in mu_stock is untouched.

Point-in-time discipline: the regression slope behind each tilt is *never* a fixed, pre-calibrated
constant - `compute_macro_timing_tilt` re-estimates it fresh from whatever history is available up
to `as_of` every time it's called (including at every walk-forward-backtest rebalance date), the
same discipline portfolio/expected_returns.py's own EWMA/shrinkage estimates already follow. Before
config.MACRO_TIMING_MIN_MONTHS of overlapping monthly data exist, a group's tilt is simply 0.0 (not
an error) - there isn't enough history yet to estimate a slope.
"""

from __future__ import annotations

import logging

import pandas as pd

from factor_model import config
from factor_model.regression.macro_predictability import (
    macro_predictive_regression,
    monthly_factor_returns,
    monthly_macro_change,
)

logger = logging.getLogger(__name__)


def fetch_oil_price(start: str) -> pd.Series:
    """Brent crude futures daily close (config.OIL_TICKER) via yfinance."""
    import yfinance as yf

    price = yf.Ticker(config.OIL_TICKER).history(start=start, auto_adjust=True)["Close"]
    price.index = price.index.tz_localize(None)
    return price.rename("OilPrice")


def fetch_usdinr(start: str) -> pd.Series:
    """USD/INR spot daily close (config.USDINR_TICKER) via yfinance."""
    import yfinance as yf

    price = yf.Ticker(config.USDINR_TICKER).history(start=start, auto_adjust=True)["Close"]
    price.index = price.index.tz_localize(None)
    return price.rename("USDINR")


def _momentum(monthly_change: pd.Series, lookback_months: int) -> pd.Series:
    """Trailing `lookback_months`-month compounded return of a monthly change series."""
    return (1.0 + monthly_change).rolling(lookback_months).apply(lambda x: x.prod() - 1.0, raw=True)


def _group_tilt(
    returns_wide: pd.DataFrame,
    group_symbols: list[str],
    rest_symbols: list[str],
    macro_price: pd.Series,
    lookback_months: int,
) -> tuple[float, float]:
    """Fits group_spread_monthly ~ lagged macro momentum on all available history, returns
    (slope, latest_momentum_reading) - or (0.0, 0.0) if there isn't enough overlapping history
    yet (macro_predictive_regression's own >=24-month requirement) or momentum isn't computable.
    """
    spread_daily = returns_wide[group_symbols].mean(axis=1) - returns_wide[rest_symbols].mean(axis=1)
    spread_monthly = monthly_factor_returns(spread_daily.to_frame("spread"))

    momentum = _momentum(monthly_macro_change(macro_price), lookback_months)
    momentum_valid = momentum.dropna()
    if momentum_valid.empty:
        return 0.0, 0.0

    reg = macro_predictive_regression(momentum, spread_monthly, lag=1)
    if reg.empty:
        return 0.0, 0.0

    slope = float(reg.iloc[0]["Slope"])
    latest_momentum = float(momentum_valid.iloc[-1])
    return slope, latest_momentum


def compute_macro_timing_tilt(
    returns_wide: pd.DataFrame,
    universe: list[str],
    oil_price: pd.Series,
    usdinr: pd.Series,
    as_of: pd.Timestamp | None = None,
) -> pd.Series:
    """Returns a Symbol-indexed daily expected-return tilt (0.0 for every symbol outside both
    groups) to add to mu_stock. `as_of` truncates all history used (returns, oil price, USD/INR)
    to that date for point-in-time-correct backtest use - omit it (or pass the latest available
    date) for a live/current portfolio construction.
    """
    if as_of is not None:
        returns_wide = returns_wide.loc[:as_of]
        oil_price = oil_price.loc[:as_of]
        usdinr = usdinr.loc[:as_of]

    universe_syms = [s for s in universe if s in returns_wide.columns]
    tilt = pd.Series(0.0, index=universe_syms)

    for group_symbols, macro_price, lookback_months in [
        (config.OIL_GROUP_SYMBOLS, oil_price, config.OIL_MOMENTUM_LOOKBACK_MONTHS),
        (config.IT_EXPORTER_SYMBOLS, usdinr, config.USDINR_MOMENTUM_LOOKBACK_MONTHS),
    ]:
        group = [s for s in group_symbols if s in universe_syms]
        if len(group) < 2:
            continue
        rest = [s for s in universe_syms if s not in group]

        slope, latest_momentum = _group_tilt(returns_wide, group, rest, macro_price, lookback_months)
        if slope == 0.0 and latest_momentum == 0.0:
            continue

        monthly_tilt = slope * latest_momentum
        daily_tilt = monthly_tilt / config.TRADING_DAYS_PER_MONTH
        tilt.loc[group] += daily_tilt
        logger.info(
            "Macro timing tilt: %d symbols, slope=%.4f, latest_momentum=%.4f, daily_tilt=%.6f",
            len(group), slope, latest_momentum, daily_tilt,
        )

    return tilt
