"""Newey-West HAC t-stats on daily factor-return time series.

Convention (.claude/skills/statistics/SKILL.md): fit an intercept-only OLS on
each factor's daily coefficient series with HAC standard errors, lag chosen
by the Newey-West (1994) plug-in rule. Report both naive and NW t-stats side
by side - they diverge exactly when factor returns are autocorrelated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def newey_west_lag(n_obs: int) -> int:
    """Newey-West (1994) plug-in lag rule: L = floor(4*(T/100)^(2/9))."""
    return max(1, int(np.floor(4 * (n_obs / 100) ** (2 / 9))))


def summarize_factor_returns(factor_returns: pd.DataFrame) -> pd.DataFrame:
    """One row per factor: mean coefficient, naive t-stat, NW SE/t-stat, % positive days, N days."""
    rows = []
    for factor in factor_returns.columns:
        series = factor_returns[factor].dropna()
        n = len(series)
        if n < 3:
            continue

        mean_coef = series.mean()
        naive_se = series.std(ddof=1) / np.sqrt(n)
        naive_t = mean_coef / naive_se if naive_se > 0 else np.nan

        X = np.ones((n, 1))
        lag = newey_west_lag(n)
        ols = sm.OLS(series.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
        nw_se = ols.bse[0]
        nw_t = ols.tvalues[0]

        rows.append(
            {
                "Factor": factor,
                "Mean_coef": mean_coef,
                "Naive_t": naive_t,
                "NW_SE": nw_se,
                "NW_t": nw_t,
                "Pct_days_positive": float((series > 0).mean()),
                "N_days": n,
            }
        )
    return pd.DataFrame(rows)
