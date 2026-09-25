"""Point-in-time utility: expand annual (FY-end) fundamentals to a daily panel
without look-ahead bias.

An FY's fundamentals aren't public until REPORTING_LAG_DAYS after the FY-end
(SEBI LODR's annual-results filing deadline) - see
.claude/skills/fundamental_analysis/SKILL.md. Forward-filling before that date
would leak future information into past dates, which model_reviewer's
look-ahead-bias check is specifically designed to catch.
"""

from __future__ import annotations

import pandas as pd

from factor_model import config


def apply_reporting_lag(fy_end_dates: pd.Series, lag_days: int = config.REPORTING_LAG_DAYS) -> pd.Series:
    """Shift each FY-end date forward by lag_days to get the date the value becomes available."""
    return fy_end_dates + pd.Timedelta(days=lag_days)


def expand_to_daily(
    annual_series: pd.Series,
    business_day_index: pd.DatetimeIndex,
    lag_days: int = config.REPORTING_LAG_DAYS,
) -> pd.Series:
    """Forward-fill an FY-indexed series onto a daily business-day index, respecting the
    reporting lag: a given FY's value is NaN until FY_end + lag_days, then holds until the
    next FY's value becomes available. Forward-fill only, never backfill or interpolate.
    """
    annual_series = annual_series.dropna()
    if annual_series.empty:
        return pd.Series(index=business_day_index, dtype=float)

    available_dates = apply_reporting_lag(pd.Series(annual_series.index), lag_days)
    staged = pd.Series(annual_series.values, index=pd.DatetimeIndex(available_dates))
    staged = staged.sort_index()

    combined_index = business_day_index.union(staged.index).sort_values()
    daily = staged.reindex(combined_index).ffill()
    daily = daily.reindex(business_day_index)
    return daily


def expand_frame_to_daily(
    annual_by_symbol: dict[str, pd.Series],
    business_day_index: pd.DatetimeIndex,
    lag_days: int = config.REPORTING_LAG_DAYS,
) -> pd.DataFrame:
    """Apply expand_to_daily per symbol and assemble into a wide (Date x Symbol) DataFrame."""
    cols = {
        symbol: expand_to_daily(series, business_day_index, lag_days)
        for symbol, series in annual_by_symbol.items()
    }
    df = pd.DataFrame(cols)
    df.index.name = "Date"
    return df
