"""Rolling market-model regression: shared basis for both beta and idiosyncratic volatility.

Computed in-house from price data (not loaded from a precomputed data/ file, unlike most other
factors) so that extending the underlying price history actually extends these factors' usable
history too - see CLAUDE.md "History length" for why this matters.

For each date t, beta_t and idio_vol_t come from the same trailing LOOKBACK_DAYS market-model
regression: r_stock = alpha + beta * r_market + eps. Computed in closed form via rolling
covariance/variance (exact for a univariate OLS regression) rather than a per-window explicit
fit, which is what makes this vectorizable across the whole panel instead of a per-date Python
loop:
    beta_t        = Cov(r_stock, r_market)_t / Var(r_market)_t
    Var(eps)_t     = Var(r_stock)_t - beta_t^2 * Var(r_market)_t
    idio_vol_t     = sqrt(Var(eps)_t)          (daily scale, not annualized - irrelevant once
                                                 z-scored, see .claude/skills/statistics/SKILL.md)
"""

from __future__ import annotations

import pandas as pd

from factor_model import config
from factor_model.io.loaders import load_benchmark_returns, load_returns


def compute_rolling_beta_and_idio_vol(
    universe: list[str], lookback_days: int = config.BETA_IDIO_VOL_LOOKBACK_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (beta_wide, idio_vol_wide), both (Date index, Symbol columns)."""
    returns = load_returns(universe)
    market_returns = load_benchmark_returns()
    market_returns = market_returns.reindex(returns.index)

    rolling_cov = returns.rolling(lookback_days, min_periods=lookback_days).cov(market_returns)
    rolling_var_market = market_returns.rolling(lookback_days, min_periods=lookback_days).var()
    rolling_var_stock = returns.rolling(lookback_days, min_periods=lookback_days).var()

    beta = rolling_cov.div(rolling_var_market, axis=0)
    residual_var = rolling_var_stock.sub(beta.pow(2).multiply(rolling_var_market, axis=0))
    idio_vol = residual_var.clip(lower=0) ** 0.5

    beta.index.name = "Date"
    idio_vol.index.name = "Date"
    return beta, idio_vol
