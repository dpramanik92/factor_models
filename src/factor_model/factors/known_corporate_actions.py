"""Hand-maintained list of fundamentals observations known to be distorted by a
merger/demerger/restatement rather than reflecting organic business performance.

YoY-delta ratios (revenue growth, the capex proxy) assume the entity being
compared is the same business a year earlier - a merger breaks that
assumption even though the underlying financial-statement numbers are
completely genuine. Levels-based ratios (leverage, ROA, ROE, net margin) are
left alone: they're legitimate as-of-FY snapshots of the (now-merged)
entity's actual balance sheet, not something to hide.

Maintained by the equity_researcher agent. Add an entry here (rather than a
blanket statistical outlier threshold) whenever a real, identifiable
corporate action is found to have produced a non-organic jump - a generic
threshold can't reliably distinguish e.g. HDFC Bank's merger-driven +66% FY24
revenue "growth" from Zomato's genuinely organic +168% growth in the same
sample.
"""

from __future__ import annotations

import pandas as pd

# (Symbol, FiscalYearEnd) -> reason. FiscalYearEnd is the FY whose YoY-delta
# ratios (growth, capex) should be excluded because the prior-year comparison
# no longer reflects the same business.
KNOWN_CORPORATE_ACTIONS: dict[tuple[str, str], str] = {
    ("HDFCBANK", "2024-03-31"): (
        "HDFC Ltd merged into HDFC Bank effective 2023-07-01 (reverse merger). "
        "FY2024 Sales +66.1% and Total Assets +59.3% YoY reflect absorbing HDFC "
        "Ltd's balance sheet, not organic growth or capex."
    ),
}


def exclude_known_corporate_actions(annual_ratios: pd.DataFrame) -> pd.DataFrame:
    """NaN out RevenueGrowth/Capex for (Symbol, FiscalYearEnd) pairs in the known list."""
    df = annual_ratios.copy()
    fy_str = df["FiscalYearEnd"].dt.strftime("%Y-%m-%d")
    key = list(zip(df["Symbol"], fy_str))
    mask = pd.Series(key, index=df.index).isin(KNOWN_CORPORATE_ACTIONS.keys())
    df.loc[mask, ["RevenueGrowth", "Capex"]] = pd.NA
    return df
