"""Value factor: composite of earnings yield, book yield, and dividend yield.

E/P and B/P are computed in-house from daily price x point-in-time per-share fundamentals
(EPS, book value per share, dividend per share - factors/fundamentals_ratios.py), not loaded
from the precomputed data/nse198_{pe,pb,divyield}_full files, which only cover 2020-09-22
onward regardless of how far back the price panel itself goes (see CLAUDE.md "History
extension"). E/P = EPS/Price and B/P = BookValuePerShare/Price are used instead of raw P/E or
P/B to avoid the sign flips and blow-ups those ratios produce near zero or negative
earnings/book value. Each subcomponent is winsorized and z-scored per date, then averaged
(requiring at least 2 of 3 non-missing) and re-z-scored into a single Value factor - see
.claude/skills/statistics/SKILL.md.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from factor_model.factors._util import wide_to_long
from factor_model.factors.pit import expand_frame_to_daily
from factor_model.io.loaders import load_prices
from factor_model.standardize import winsorize_zscore


def compute_value(annual_ratios: pd.DataFrame, business_day_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Returns long (Date, Symbol, value) - the raw (pre-final-zscore) composite of the three
    standardized subcomponents (E/P, B/P, dividend yield). Subcomponent standardization happens
    here (needed to combine incommensurable ratios); the final re-standardization of the
    composite happens uniformly in factors/panel.py.
    """
    universe = sorted(annual_ratios["Symbol"].unique())
    prices = load_prices(universe).reindex(index=business_day_index, columns=universe)

    eps_by_symbol = {s: g.set_index("FiscalYearEnd")["EPS"] for s, g in annual_ratios.groupby("Symbol")}
    bvps_by_symbol = {
        s: g.set_index("FiscalYearEnd")["BookValuePerShare"] for s, g in annual_ratios.groupby("Symbol")
    }
    dps_by_symbol = {
        s: g.set_index("FiscalYearEnd")["DividendPerShare"] for s, g in annual_ratios.groupby("Symbol")
    }
    eps_daily = expand_frame_to_daily(eps_by_symbol, business_day_index).reindex(columns=universe)
    bvps_daily = expand_frame_to_daily(bvps_by_symbol, business_day_index).reindex(columns=universe)
    dps_daily = expand_frame_to_daily(dps_by_symbol, business_day_index).reindex(columns=universe)

    price_positive = prices.where(prices > 0)
    ep = eps_daily.where(eps_daily > 0) / price_positive
    bp = bvps_daily.where(bvps_daily > 0) / price_positive
    dy = dps_daily.where(dps_daily >= 0) / price_positive

    ep_z = ep.apply(lambda row: winsorize_zscore(row), axis=1)
    bp_z = bp.apply(lambda row: winsorize_zscore(row), axis=1)
    dy_z = dy.apply(lambda row: winsorize_zscore(row), axis=1)

    stacked = np.stack([ep_z.to_numpy(), bp_z.to_numpy(), dy_z.to_numpy()])
    n_valid = np.sum(~np.isnan(stacked), axis=0)
    with warnings.catch_warnings():
        # All-NaN slices are expected wherever a (date, symbol) has no valid subcomponents;
        # they're masked to NaN by the n_valid check below regardless of nanmean's own value.
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        composite = np.nanmean(stacked, axis=0)
    composite = np.where(n_valid >= 2, composite, np.nan)

    # Composite is an average of already-standardized subcomponents; the final
    # re-standardization happens uniformly in factors/panel.py alongside every
    # other style factor, so we return the raw composite here.
    composite_df = pd.DataFrame(composite, index=business_day_index, columns=universe)
    composite_df.index.name = "Date"

    return wide_to_long(composite_df, "value")
