"""Technical-timing-signal research on a factor's own daily return series - built to answer one
question empirically before touching any production code: does trading the `beta` factor's
premium on a MACD or 30-day moving-average crossover add genuine, statistically significant alpha
over simply holding it (buy-and-hold), or is it noise? See portfolio/expected_returns.py for
whether/how a signal that clears this bar gets used.

A factor's daily return series (output/factor_returns_daily.csv) is exactly the daily P&L of
holding a unit of that factor's cross-sectional exposure, so cumulating it (cumprod(1+r)) gives a
well-defined synthetic "price" index to run ordinary technical signals against - the same
construction a MACD/moving-average signal would use on any tradeable price series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from factor_model.regression.newey_west import newey_west_lag


def factor_price_index(factor_returns: pd.Series) -> pd.Series:
    """Synthetic price index (starts at 1.0) from a factor's daily return series."""
    return (1.0 + factor_returns.fillna(0.0)).cumprod()


def macd_position(price: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """+1 when the MACD line (fast EMA - slow EMA) is above its signal line (EMA of the MACD
    line), -1 otherwise - standard MACD crossover convention.
    """
    fast_ema = price.ewm(span=fast, adjust=False).mean()
    slow_ema = price.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return np.sign(macd_line - signal_line).replace(0.0, 1.0)


def moving_average_position(price: pd.Series, window: int = 30) -> pd.Series:
    """+1 when price is above its trailing `window`-day simple moving average, -1 otherwise."""
    ma = price.rolling(window, min_periods=window).mean()
    return np.sign(price - ma).replace(0.0, 1.0)


def momentum_position(price: pd.Series, lookback: int, skip: int = 0) -> pd.Series:
    """Time-series momentum: +1 when the trailing `lookback`-day return (measured as of `skip`
    days ago, so e.g. lookback=252, skip=21 reproduces this model's own 12-month-ex-1-month
    momentum factor convention - factors/momentum.py) is positive, -1 otherwise.
    """
    past = price.shift(skip + lookback)
    recent = price.shift(skip)
    trailing_return = recent / past - 1.0
    return np.sign(trailing_return).replace(0.0, 1.0)


def test_timing_alpha(factor_returns: pd.Series, position: pd.Series, strategy_name: str) -> dict:
    """Does timing this factor with `position` (computed from information available up to and
    including day t, so shifted by 1 before trading day t+1's return - no look-ahead) add alpha
    over simply buying and holding the factor? Tests the NW-HAC significance of the *excess* of
    the timed strategy's daily return over buy-and-hold's, plus each strategy's own stats.
    """
    position = position.reindex(factor_returns.index)
    traded_position = position.shift(1)  # yesterday's signal trades today's return
    valid = traded_position.notna() & factor_returns.notna()

    r = factor_returns[valid]
    pos = traded_position[valid]
    strategy_returns = pos * r
    excess = (strategy_returns - r).dropna()

    n = len(excess)
    lag = newey_west_lag(n)
    ols = sm.OLS(excess.to_numpy(), np.ones((n, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lag})

    def _ann_stats(series: pd.Series) -> dict:
        mean = float(series.mean())
        std = float(series.std(ddof=1))
        return {
            "ann_return": mean * 252,
            "ann_vol": std * (252 ** 0.5),
            "sharpe": (mean / std) * (252 ** 0.5) if std > 0 else np.nan,
        }

    return {
        "strategy": strategy_name,
        "buy_and_hold": _ann_stats(r),
        "timed": _ann_stats(strategy_returns),
        "excess_mean_daily": float(excess.mean()),
        "excess_nw_t": float(ols.tvalues[0]),
        "excess_nw_se": float(ols.bse[0]),
        "pct_days_long": float((pos > 0).mean()),
        "n_days": n,
    }
