"""Liquidity factor: log of trailing-252-day average daily share turnover (volume divided by
point-in-time shares outstanding), computed in-house from raw daily volume
(data/nse198_daily_volume.parquet - backfilled 2017-01-02 onward for all universe symbols, see
CLAUDE.md's "History extension") and the same point-in-time shares-outstanding panel
factors/market_cap.py already builds for market cap.

Higher value = more liquid (heavily traded relative to shares outstanding). Barra-style liquidity
factors are conventionally built from multiple horizons (STOM/STOQ/STOA - 1-month/1-quarter/
1-year turnover); this uses a single 252-trading-day window instead, consistent with every other
single-lookback factor already in this model (beta/idio_vol/momentum all use 252 days too), not
a Barra multi-horizon blend. The regression itself reveals the empirical sign - a common finding
in equity markets is an *illiquidity premium* (less liquid stocks earn higher average returns),
which would show up here as a negative coefficient on this liquidity measure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_model import config
from factor_model.factors._util import wide_to_long
from factor_model.factors.market_cap import compute_shares_outstanding_daily
from factor_model.io.loaders import load_volume


def _forward_fill_unflagged_holidays(
    volume: pd.DataFrame, threshold: float = config.LIQUIDITY_HOLIDAY_ZERO_FRACTION_THRESHOLD
) -> pd.DataFrame:
    """Some exchange holidays aren't flagged as such in the raw data - the whole cross-section
    shows zero or NaN volume that day (verified against real data: 100% zero/NaN across all 209
    symbols on known unflagged-holiday dates, vs. 0% on an ordinary trading day - the same
    signature regression/cross_sectional.py's degenerate-return-day check already independently
    detects for these same dates). Left unfixed, one such day poisons the entire trailing
    LIQUIDITY_LOOKBACK_DAYS rolling window that contains it (a min_periods == window rolling
    calculation requires zero nulls/zeros-treated-as-missing anywhere in the window) - not just
    that day's factor value but up to a year of subsequent ones. Forward-filled here, the same
    remedy already used for the analogous NIFTY50 benchmark price gaps (see CLAUDE.md).
    """
    is_holiday = (volume.isna() | (volume == 0)).mean(axis=1) >= threshold
    filled = volume.copy()
    filled.loc[is_holiday] = np.nan
    return filled.ffill()


def compute_liquidity(
    universe: list[str],
    annual_ratios: pd.DataFrame,
    business_day_index: pd.DatetimeIndex,
    lookback_days: int = config.LIQUIDITY_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Returns long (Date, Symbol, liquidity) - raw log(average daily turnover), not yet
    z-scored (that happens uniformly in factors/panel.py).
    """
    volume = load_volume(universe).reindex(index=business_day_index, columns=universe)
    volume = _forward_fill_unflagged_holidays(volume)
    shares_daily = compute_shares_outstanding_daily(annual_ratios, business_day_index).reindex(columns=universe)

    turnover = volume / shares_daily.where(shares_daily > 0)
    avg_turnover = turnover.rolling(lookback_days, min_periods=lookback_days).mean()
    liquidity = np.log(avg_turnover.where(avg_turnover > 0))
    liquidity.index.name = "Date"

    return wide_to_long(liquidity, "liquidity")
