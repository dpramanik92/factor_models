"""1-year momentum factor, computed fresh from daily close prices.

mom(t) = P(t - MOMENTUM_LAG_DAYS) / P(t - MOMENTUM_LOOKBACK_DAYS) - 1

Trailing ~252-trading-day return excluding the most recent ~21 trading days,
the standard Barra/academic convention that avoids short-term-reversal
contamination (see .claude/skills/statistics/SKILL.md). Not the same as
data/nse200_momentum_6m, which is a different (6-month) window.
"""

from __future__ import annotations

import pandas as pd

from factor_model import config
from factor_model.factors._util import wide_to_long
from factor_model.io.loaders import load_prices


def compute_momentum(universe: list[str]) -> pd.DataFrame:
    """Returns long (Date, Symbol, momentum) for the given universe."""
    prices = load_prices(universe)
    lagged_start = prices.shift(config.MOMENTUM_LOOKBACK_DAYS)
    lagged_end = prices.shift(config.MOMENTUM_LAG_DAYS)
    momentum = lagged_end / lagged_start - 1.0
    return wide_to_long(momentum, "momentum")
