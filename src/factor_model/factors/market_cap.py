"""Market cap: price x point-in-time shares outstanding, computed in-house from price data and
fundamentals (not loaded from the precomputed data/nse198_size_full file, which only covers
2020-09-22 onward regardless of how far back the price panel itself goes - see CLAUDE.md
"History extension"). Used both as the size factor's raw input and as the WLS regression weight
(sqrt(market_cap)) - see factors/size.py and factors/panel.py.
"""

from __future__ import annotations

import pandas as pd

from factor_model.factors.pit import expand_frame_to_daily
from factor_model.io.loaders import load_prices


def compute_shares_outstanding_daily(
    annual_ratios: pd.DataFrame, business_day_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Point-in-time daily shares outstanding (Date index, Symbol columns), expanded from annual
    filings with the standard reporting lag (factors/pit.py). Shared by compute_market_cap (this
    file) and factors/liquidity.py's turnover calculation, so both use the same PIT shares figure.
    """
    universe = sorted(annual_ratios["Symbol"].unique())
    shares_by_symbol = {
        s: g.set_index("FiscalYearEnd")["No. of Equity Shares"] for s, g in annual_ratios.groupby("Symbol")
    }
    return expand_frame_to_daily(shares_by_symbol, business_day_index).reindex(columns=universe)


def compute_market_cap(
    annual_ratios: pd.DataFrame, business_day_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Returns (Date index, Symbol columns) raw market cap in rupees."""
    universe = sorted(annual_ratios["Symbol"].unique())
    prices = load_prices(universe).reindex(index=business_day_index, columns=universe)
    shares_daily = compute_shares_outstanding_daily(annual_ratios, business_day_index)

    market_cap = prices * shares_daily
    market_cap.index.name = "Date"
    return market_cap
