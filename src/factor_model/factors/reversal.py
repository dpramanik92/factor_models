"""Short-term reversal factor, computed fresh from daily close prices.

reversal(t) = -(P(t - REVERSAL_LAG_DAYS) / P(t - REVERSAL_LOOKBACK_DAYS) - 1)

The negative of the trailing ~1-month (21-trading-day) return - exactly the window
factors/momentum.py's 12-1 convention deliberately excludes, to avoid conflating momentum with
short-term reversal (Jegadeesh 1990 / Lehmann 1990: recent losers tend to bounce, recent winners
tend to give some back, over a ~1-month horizon). Sign convention matches momentum.py's (higher
factor value -> expect higher future return): negating the raw recent return means a stock that
fell over the last month gets a positive reversal exposure. Exploratory - not yet wired into
config.STYLE_FACTORS/factors/panel.py pending an alpha check (see CLAUDE.md's "Alpha research"
notes).
"""

from __future__ import annotations

import pandas as pd

from factor_model import config
from factor_model.factors._util import wide_to_long
from factor_model.io.loaders import load_prices

#: Reuses momentum's own lag window (its exclusion window is this factor's lookback window).
REVERSAL_LOOKBACK_DAYS = config.MOMENTUM_LAG_DAYS
REVERSAL_LAG_DAYS = 0


def compute_reversal(universe: list[str]) -> pd.DataFrame:
    """Returns long (Date, Symbol, reversal) for the given universe."""
    prices = load_prices(universe)
    lagged_start = prices.shift(REVERSAL_LOOKBACK_DAYS)
    lagged_end = prices.shift(REVERSAL_LAG_DAYS)
    reversal = -(lagged_end / lagged_start - 1.0)
    return wide_to_long(reversal, "reversal")
