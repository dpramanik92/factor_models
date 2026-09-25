"""Assemble the mapped, symbol-keyed annual fundamentals table and derive ratios.

Combines the universe mapping (mapping/ticker_mapping.py) with the parsed
per-file fundamentals (io/fundamentals.py) into one long (Symbol,
FiscalYearEnd, ...) table - this becomes model_data/fundamentals_annual_long
- and computes the ratios needed by leverage/quality/capex/growth/
profitability. See .claude/skills/fundamental_analysis/SKILL.md for formulas.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from factor_model.factors.known_corporate_actions import exclude_known_corporate_actions
from factor_model.io.fundamentals import ParsedFundamentals

logger = logging.getLogger(__name__)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.where(denominator != 0)
    return numerator / denom


def build_fundamentals_annual_long(
    mapping_df: pd.DataFrame, parsed: dict[str, ParsedFundamentals]
) -> pd.DataFrame:
    """One row per (Symbol, FiscalYearEnd) for every included, successfully-parsed file."""
    included = mapping_df[mapping_df["included"]]
    frames: list[pd.DataFrame] = []
    for _, row in included.iterrows():
        stem = Path(row["fundamentals_filename"]).stem
        item = parsed.get(stem)
        if item is None:
            logger.warning("No parsed fundamentals for included file %s; skipping", row["fundamentals_filename"])
            continue
        annual = item.annual.reset_index()
        annual["Symbol"] = row["matched_symbol"]
        frames.append(annual)

    if not frames:
        return pd.DataFrame()

    long_df = pd.concat(frames, ignore_index=True)
    cols = ["Symbol", "FiscalYearEnd"] + [c for c in long_df.columns if c not in ("Symbol", "FiscalYearEnd")]
    return long_df[cols].sort_values(["Symbol", "FiscalYearEnd"]).reset_index(drop=True)


def compute_annual_ratios(fundamentals_annual_long: pd.DataFrame) -> pd.DataFrame:
    """Add Leverage, ROA, ROE, Capex (proxy), RevenueGrowth, NetMargin columns."""
    df = fundamentals_annual_long.sort_values(["Symbol", "FiscalYearEnd"]).copy()

    equity = df["Equity Share Capital"] + df["Reserves"]

    df["Leverage"] = _safe_divide(df["Borrowings"], equity)
    df["ROA"] = _safe_divide(df["Net profit"], df["Total Assets"])
    df["ROE"] = _safe_divide(df["Net profit"], equity)
    df["NetMargin"] = _safe_divide(df["Net profit"], df["Sales"])

    grp = df.groupby("Symbol")
    prior_net_block = grp["Net Block"].shift(1)
    prior_total_assets = grp["Total Assets"].shift(1)
    prior_sales = grp["Sales"].shift(1)

    net_block_delta = df["Net Block"] - prior_net_block
    df["Capex"] = _safe_divide(net_block_delta + df["Depreciation"], prior_total_assets)
    df["RevenueGrowth"] = _safe_divide(df["Sales"], prior_sales) - 1.0

    # A ratio built off a partial/TTM latest FY (or a prior-year gap) is not comparable;
    # NaN it out rather than let it silently look like a normal YoY figure.
    fy_gap_days = grp["FiscalYearEnd"].diff().dt.days
    non_annual_gap = (fy_gap_days < 300) | (fy_gap_days > 430)
    df.loc[non_annual_gap.fillna(False), ["Capex", "RevenueGrowth"]] = np.nan

    df = exclude_known_corporate_actions(df)

    return df
