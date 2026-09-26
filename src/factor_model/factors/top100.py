"""Top-100-by-market-cap indicator factor: a per-date 0/1 flag for whether a stock ranks in the
top `config.TOP100_N` names by market cap that day, among the stocks with a valid market cap.
Reuses the market-cap panel factors/market_cap.py already builds (shared with the size factor
and the WLS regression weight) rather than recomputing it.

Added because the factors driving the largest-cap segment of this universe were observed to
behave differently from the rest - this indicator tests for a *level* effect (a return premium
or discount specific to large-cap membership) on top of the existing continuous, smooth
log(market_cap) size factor, not a full interaction between cap segment and every other factor's
coefficient (that would be a much larger model - this is deliberately just the two requested
indicator variables, not a segment-interaction model).
"""

from __future__ import annotations

import pandas as pd

from factor_model import config
from factor_model.factors._util import wide_to_long


def compute_top100_flag(market_cap: pd.DataFrame, n: int = config.TOP100_N) -> pd.DataFrame:
    """Returns long (Date, Symbol, top100_flag). Ranked per date among symbols with a valid
    (non-null, positive) market cap that day; NaN market cap -> NaN flag (missing, not 0 - a
    stock with no market cap that day isn't "not in the top N", it's simply not ranked).
    """
    valid = market_cap.where(market_cap > 0)
    rank = valid.rank(axis=1, method="first", ascending=False)
    flag = (rank <= n).astype(float)
    flag = flag.where(valid.notna())  # keep NaN where the underlying market cap was itself NaN
    flag.index.name = "Date"
    return wide_to_long(flag, "top100_flag")
