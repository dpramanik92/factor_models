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


def build_fundamentals_quarterly_long(
    mapping_df: pd.DataFrame, parsed: dict[str, ParsedFundamentals]
) -> pd.DataFrame:
    """One row per (Symbol, QuarterEnd) for every included, successfully-parsed file with a
    non-empty quarterly block (io/fundamentals.py's `Quarters` section - only the last ~10
    trailing quarters Screener exports, not full history). Exploratory - not part of the main
    annual-frequency pipeline; see factors/pead.py / CLAUDE.md's "Alpha research" notes.
    """
    included = mapping_df[mapping_df["included"]]
    frames: list[pd.DataFrame] = []
    for _, row in included.iterrows():
        stem = Path(row["fundamentals_filename"]).stem
        item = parsed.get(stem)
        if item is None or item.quarterly is None or item.quarterly.empty:
            continue
        quarterly = item.quarterly.reset_index()
        quarterly["Symbol"] = row["matched_symbol"]
        frames.append(quarterly)

    if not frames:
        return pd.DataFrame()

    long_df = pd.concat(frames, ignore_index=True)
    cols = ["Symbol", "QuarterEnd"] + [c for c in long_df.columns if c not in ("Symbol", "QuarterEnd")]
    return long_df[cols].sort_values(["Symbol", "QuarterEnd"]).reset_index(drop=True)


def compute_annual_ratios(fundamentals_annual_long: pd.DataFrame) -> pd.DataFrame:
    """Add Leverage, ROA, ROE, Capex (proxy), RevenueGrowth, NetMargin columns."""
    df = fundamentals_annual_long.sort_values(["Symbol", "FiscalYearEnd"]).copy()

    equity = df["Equity Share Capital"] + df["Reserves"]

    df["Leverage"] = _safe_divide(df["Borrowings"], equity)
    df["ROA"] = _safe_divide(df["Net profit"], df["Total Assets"])
    df["ROE"] = _safe_divide(df["Net profit"], equity)
    df["NetMargin"] = _safe_divide(df["Net profit"], df["Sales"])

    # Per-share figures, used by factors/value.py to compute PE/PB/dividend yield in-house
    # (instead of the precomputed data/nse198_{pe,pb,divyield}_full files, which only cover
    # 2020-09-22 onward regardless of how far back the price panel itself goes).
    # "No. of Equity Shares" is Screener's raw absolute share count, while every numerator here
    # (Net profit, equity, Dividend Amount) is in Rs. Crore - divide shares by 1e7 too (crore =
    # 1e7) so the per-share result is in plain rupees, not off by ~1e7. Found by equity_researcher
    # review (2026-09): this was previously numerically inert (the ~1e7 mis-scale is a uniform
    # positive constant across every company/date, so it cancelled out exactly under factors/
    # value.py's per-date winsorize+z-score), but left uncorrected it's a landmine for any future
    # consumer of these columns as raw diagnostic numbers.
    shares = (df["No. of Equity Shares"] / 1e7).where(df["No. of Equity Shares"] > 0)
    df["EPS"] = _safe_divide(df["Net profit"], shares)
    df["BookValuePerShare"] = _safe_divide(equity, shares)
    df["DividendPerShare"] = _safe_divide(df["Dividend Amount"], shares)

    grp = df.groupby("Symbol")
    prior_net_block = grp["Net Block"].shift(1)
    prior_total_assets = grp["Total Assets"].shift(1)
    prior_sales = grp["Sales"].shift(1)

    net_block_delta = df["Net Block"] - prior_net_block
    df["Capex"] = _safe_divide(net_block_delta + df["Depreciation"], prior_total_assets)
    df["RevenueGrowth"] = _safe_divide(df["Sales"], prior_sales) - 1.0

    # Accruals (Sloan 1996 earnings-quality anomaly): the portion of reported profit not backed
    # by operating cash flow, scaled by total assets. Same-period Total Assets (not prior-period),
    # matching the ROA/NetMargin convention above rather than Capex's prior-period one, since this
    # is a same-period ratio, not a YoY comparison - see factors/accruals.py.
    df["Accruals"] = _safe_divide(df["Net profit"] - df["Cash from Operating Activity"], df["Total Assets"])

    # A ratio built off a partial/TTM latest FY (or a prior-year gap) is not comparable;
    # NaN it out rather than let it silently look like a normal YoY figure.
    fy_gap_days = grp["FiscalYearEnd"].diff().dt.days
    non_annual_gap = (fy_gap_days < 300) | (fy_gap_days > 430)
    df.loc[non_annual_gap.fillna(False), ["Capex", "RevenueGrowth"]] = np.nan

    df = exclude_known_corporate_actions(df)

    return df
