"""Does a macro variable's prior-month return/change predict a factor's return this month?

One lagged-predictor time-series regression per (macro variable, factor) pair:
    factor_return_monthly_t = alpha + beta * macro_change_monthly_{t-1} + error
with Newey-West HAC standard errors on beta (monthly data is far more autocorrelation-prone
than the daily factor-return series elsewhere in this model, so HAC is not optional here).

This is a pure research/diagnostic module - unlike portfolio/timing_signals.py's within-factor
technical signals, it isn't wired into the expected-return construction or the backtest;
whether any of its findings get used there is a separate decision, made only after reviewing
actual results (see the multiple-testing caveat on summarize_macro_predictability).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from factor_model.regression.newey_west import newey_west_lag


def monthly_factor_returns(factor_returns_daily: pd.DataFrame) -> pd.DataFrame:
    """Compounds daily factor returns to calendar-month returns: (1+r).prod() - 1 per month."""
    return (1.0 + factor_returns_daily).groupby(factor_returns_daily.index.to_period("M")).prod() - 1.0


def monthly_macro_change(prices: pd.Series) -> pd.Series:
    """Month-end-close simple return (pct change), indexed by calendar month period (matching
    monthly_factor_returns' PeriodIndex, not resample's native DatetimeIndex - a real bug caught
    only by running this against actual fetched data, not the synthetic-PeriodIndex unit tests:
    concatenating a PeriodIndex-keyed and a DatetimeIndex-keyed series silently aligns to nothing
    and drops every row). The same pct-change convention is applied uniformly to every macro
    series here (including the 10Y yield level, so a +5% "return" there means the yield itself
    rose 5% relative to its prior month-end level, not a 5-percentage-point move - see the
    caveat in summarize_macro_predictability).
    """
    month_end = prices.resample("ME").last()
    change = month_end.pct_change().rename(prices.name)
    change.index = change.index.to_period("M")
    return change


def macro_predictive_regression(macro_change_monthly: pd.Series, factor_returns_monthly: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """One row per factor: OLS of factor_return_t on macro_change_{t-lag} (single predictor plus
    intercept), Newey-West HAC t-stat/SE on the slope, R^2, N obs. Factors with fewer than 24
    overlapping monthly observations are skipped (too little data for a meaningful HAC estimate).
    """
    macro_lagged = macro_change_monthly.shift(lag)
    rows = []
    for factor in factor_returns_monthly.columns:
        df = pd.concat([factor_returns_monthly[factor], macro_lagged], axis=1).dropna()
        df.columns = ["y", "x"]
        n = len(df)
        if n < 24:
            continue

        X = sm.add_constant(df["x"].to_numpy())
        y = df["y"].to_numpy()
        hac_lag = newey_west_lag(n)
        ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lag})

        rows.append(
            {
                "Factor": factor,
                "Slope": ols.params[1],
                "NW_t": ols.tvalues[1],
                "NW_p": ols.pvalues[1],
                "R2": ols.rsquared,
                "N": n,
            }
        )
    return pd.DataFrame(rows)


def summarize_macro_predictability(
    macro_changes_monthly: dict[str, pd.Series], factor_returns_monthly: pd.DataFrame, lag: int = 1, bonferroni_alpha: float = 0.05
) -> pd.DataFrame:
    """Runs macro_predictive_regression for every (macro variable, factor) pair and stacks the
    results, with a Bonferroni-corrected significance flag - testing N_macros x N_factors pairs
    at the usual p<0.05 threshold would expect several false positives by chance alone (e.g. ~4
    out of 80 pairs), so `Significant_bonferroni` requires p below alpha/(n_macros*n_factors),
    a much higher bar appropriate for a multi-way fishing expedition like this one.
    """
    all_rows = []
    n_tests = len(macro_changes_monthly) * len(factor_returns_monthly.columns)
    bonferroni_threshold = bonferroni_alpha / n_tests
    for macro_name, macro_series in macro_changes_monthly.items():
        result = macro_predictive_regression(macro_series, factor_returns_monthly, lag=lag)
        result.insert(0, "Macro", macro_name)
        all_rows.append(result)
    combined = pd.concat(all_rows, ignore_index=True)
    combined["Significant_uncorrected_p05"] = combined["NW_p"] < 0.05
    combined["Significant_bonferroni"] = combined["NW_p"] < bonferroni_threshold
    return combined.sort_values("NW_p").reset_index(drop=True)
