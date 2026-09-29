"""Quarterly earnings-surprise (SUE) for a Post-Earnings-Announcement-Drift (PEAD) check -
exploratory, not wired into config.STYLE_FACTORS/factors/panel.py. See CLAUDE.md's "Alpha
research" notes for why this exists: the earlier annual-EPS-frequency PEAD test found no drift,
likely because annual results are stale news for a company that already reports quarterly - this
quarterly version (io/fundamentals.py's `Quarters` block, factors/fundamentals_ratios.py's
`build_fundamentals_quarterly_long`) tests the actual quarterly announcement events.

SUE (Foster/Olsen/Shevlin's seasonal-random-walk convention): the YoY change in quarterly Net
profit (same quarter one year ago, not the immediately preceding quarter - avoiding conflating
genuine seasonality, e.g. a festive-quarter spike, with a real surprise), standardized by that
company's own trailing rolling std of the same YoY-change series. Raw Net profit is used (not
EPS) since Screener's Quarters block doesn't carry a quarterly shares-outstanding figure.
"""

from __future__ import annotations

import pandas as pd

from factor_model import config

#: Same-quarter-one-year-ago lag (4 quarters), not a simple 1-period lag - see module docstring.
SUE_LAG_QUARTERS = 4
#: Trailing window (in quarters) for the rolling std that standardizes the YoY change - kept
#: short since Screener only exports the last ~10 trailing quarters per company.
SUE_ROLLING_WINDOW = 8
SUE_MIN_PERIODS = 3


def compute_quarterly_sue(
    quarterly_long: pd.DataFrame,
    lag_quarters: int = SUE_LAG_QUARTERS,
    rolling_window: int = SUE_ROLLING_WINDOW,
    min_periods: int = SUE_MIN_PERIODS,
) -> pd.DataFrame:
    """Returns (Symbol, QuarterEnd, SUE, AnnouncementDate) for every quarter with a computable
    surprise. AnnouncementDate = QuarterEnd + config.QUARTERLY_REPORTING_LAG_DAYS, the
    point-in-time date the result actually becomes public (SEBI LODR's quarterly filing
    deadline) - mirrors factors/pit.py's annual-reporting-lag convention.
    """
    df = quarterly_long.sort_values(["Symbol", "QuarterEnd"]).copy()
    grp = df.groupby("Symbol")

    yoy_diff = df["Net profit"] - grp["Net profit"].shift(lag_quarters)
    rolling_std = yoy_diff.groupby(df["Symbol"]).transform(
        lambda s: s.rolling(rolling_window, min_periods=min_periods).std()
    )
    df["SUE"] = yoy_diff / rolling_std.where(rolling_std > 0)
    df["AnnouncementDate"] = df["QuarterEnd"] + pd.Timedelta(days=config.QUARTERLY_REPORTING_LAG_DAYS)

    return df[["Symbol", "QuarterEnd", "SUE", "AnnouncementDate"]]
