"""Significance-shrunk, recency-weighted expected factor returns: the expected-return input to
the portfolio optimizer.

Convention: expected_daily_return_k = ewma_mean_k * shrinkage_k.
  - ewma_mean_k: factor k's EWMA-recency-weighted mean daily return (half-life
    config.EXPECTED_RETURN_EWMA_HALFLIFE_DAYS, most recent value of
    factor_returns_daily[k].ewm(halflife=...).mean()) - "what is the premium worth right now",
    not a flat average over the full ~8-year history.
  - shrinkage_k in [0, 1]: ramps linearly with |NW_t_k| (the full-sample Newey-West HAC t-stat,
    regression/newey_west.py) between config.EXPECTED_RETURN_SHRINKAGE_T_MIN (0 weight at or
    below) and config.EXPECTED_RETURN_SHRINKAGE_T_FULL (full weight at or above) - "was this
    ever a real signal at all". Deliberately still computed on the full history rather than a
    recency-weighted variant: a short recent window is too noisy to judge statistical
    significance from, but is exactly what we want for the *magnitude* once a factor has cleared
    the significance bar.

This two-part design replaced two earlier, narrower fixes in sequence:
  1. An original `historical_Sharpe * current_EWMA_vol` formula, dropped after Quant_analyst
     review surfaced no significance filter and risk/return double-counting.
  2. A `flat_full_sample_mean * significance_shrinkage` formula (no recency), dropped after
     portfolio/validation.py's test_expected_return_vs_realized_stock_returns showed it was
     dominated by stale multi-year rallies: Communication Services and Industrials cleared the
     significance bar on their full-sample mean, but the Max-Sharpe portfolio's resulting
     largest holdings (e.g. BHARTIARTL, CIPLA, DRREDDY) had *negative* realized returns over the
     trailing 252 days - rho=0.046 (p=0.67, not significant) between projected and trailing-
     window realized returns, versus rho=0.295 (p=0.005) full-sample.

factor_returns_daily is already in excess-return space (the regression's y is excess return over
the risk-free rate - see .claude/skills/statistics/SKILL.md), so no further risk-free adjustment
is needed here.
"""

from __future__ import annotations

import pandas as pd

from factor_model import config
from factor_model.regression.cross_sectional import DESIGN_COLS
from factor_model.regression.newey_west import summarize_factor_returns


def compute_significance_shrunk_expected_returns(
    factor_returns_daily: pd.DataFrame,
    t_min: float = config.EXPECTED_RETURN_SHRINKAGE_T_MIN,
    t_full: float = config.EXPECTED_RETURN_SHRINKAGE_T_FULL,
    ewma_halflife: int = config.EXPECTED_RETURN_EWMA_HALFLIFE_DAYS,
    design_cols: list[str] = DESIGN_COLS,
) -> tuple[pd.Series, pd.DataFrame]:
    """Returns (expected_daily_return per `design_cols` factor, a detail DataFrame with the
    full-sample mean/NW t-stat, the EWMA recent mean, and the shrinkage weight behind each one,
    for transparency/logging). `design_cols` defaults to the pooled model's DESIGN_COLS;
    portfolio/segmented_risk_model.py passes config.SEGMENT_DESIGN_COLS instead, once per segment.
    """
    factor_returns_daily = factor_returns_daily[design_cols]
    summary = summarize_factor_returns(factor_returns_daily).set_index("Factor").reindex(design_cols)

    ewma_mean = factor_returns_daily.ewm(halflife=ewma_halflife, adjust=True).mean().iloc[-1].reindex(design_cols)

    abs_t = summary["NW_t"].abs()
    shrinkage = ((abs_t - t_min) / (t_full - t_min)).clip(lower=0.0, upper=1.0)
    expected = ewma_mean * shrinkage

    detail = summary.assign(EWMA_recent_mean=ewma_mean, shrinkage_weight=shrinkage, expected_daily_return=expected)
    return expected.rename("expected_daily_return"), detail


def project_to_stock_returns(
    expected_factor_returns: pd.Series, exposures: pd.DataFrame, design_cols: list[str] = DESIGN_COLS
) -> pd.Series:
    """mu_stock = X_stock @ mu_factor - each stock's expected daily excess return, from its
    factor exposures and the expected factor returns above. This is the actual per-stock
    expected-return input the optimizer uses (mu_factor alone is only 20 numbers, one per
    factor - not directly a stock-level return). `design_cols` defaults to DESIGN_COLS;
    portfolio/segmented_risk_model.py passes its joint (segment-suffixed) column list instead.
    """
    X = exposures[design_cols]
    mu_stock = X @ expected_factor_returns.reindex(design_cols)
    return mu_stock.rename("expected_return")
