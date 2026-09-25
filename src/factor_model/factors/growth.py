"""Growth factor: YoY revenue (Sales) growth, computed from fundamentals.

RevenueGrowth = Sales_t / Sales_{t-1} - 1, FY-over-FY. See
.claude/skills/fundamental_analysis/SKILL.md.
"""

from __future__ import annotations

import pandas as pd

from factor_model.factors._util import wide_to_long
from factor_model.factors.pit import expand_frame_to_daily


def compute_growth(annual_ratios: pd.DataFrame, business_day_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Returns long (Date, Symbol, growth) for every symbol present in annual_ratios."""
    by_symbol = {
        symbol: group.set_index("FiscalYearEnd")["RevenueGrowth"]
        for symbol, group in annual_ratios.groupby("Symbol")
    }
    daily = expand_frame_to_daily(by_symbol, business_day_index)
    return wide_to_long(daily, "growth")
