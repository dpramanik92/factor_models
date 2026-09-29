"""Accruals factor (Sloan 1996 earnings-quality anomaly): Accruals = (Net profit - Cash from
Operating Activity) / Total Assets, computed from fundamentals and expanded to daily via the
point-in-time reporting lag (factors/pit.py). See .claude/skills/fundamental_analysis/SKILL.md.

Higher accruals means reported profit is less backed by actual operating cash flow - the
anomaly's standard finding is that high-accrual firms subsequently *underperform* low-accrual
firms (an earnings-quality signal, not a growth or value one). This is exploratory - not yet
wired into config.STYLE_FACTORS/factors/panel.py pending an alpha check (see CLAUDE.md's
"Alpha research" notes).
"""

from __future__ import annotations

import pandas as pd

from factor_model.factors._util import wide_to_long
from factor_model.factors.pit import expand_frame_to_daily


def compute_accruals(annual_ratios: pd.DataFrame, business_day_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Returns long (Date, Symbol, accruals) for every symbol present in annual_ratios."""
    by_symbol = {
        symbol: group.set_index("FiscalYearEnd")["Accruals"]
        for symbol, group in annual_ratios.groupby("Symbol")
    }
    daily = expand_frame_to_daily(by_symbol, business_day_index)
    return wide_to_long(daily, "accruals")
