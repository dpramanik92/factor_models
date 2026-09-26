"""Label-anchor parser for the Screener.in "Data Sheet" fundamentals exports.

Row positions are consistent across the 99 files we've inspected, but per
.claude/skills/fundamental_analysis/SKILL.md we deliberately don't hardcode
row indices: we scan column A for known labels within each section's row
range, so the parser survives any row-count differences between companies
(e.g. banks/NBFCs with blank sector-specific rows).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from factor_model import config

logger = logging.getLogger(__name__)

SECTION_HEADERS = ["META", "PROFIT & LOSS", "Quarters", "BALANCE SHEET", "CASH FLOW", "PRICE", "DERIVED"]

# Labels pulled from the annual PROFIT & LOSS block.
PNL_LABELS = [
    "Sales",
    "Other Income",
    "Depreciation",
    "Interest",
    "Profit before tax",
    "Tax",
    "Net profit",
    "Dividend Amount",
]

# Labels pulled from the annual BALANCE SHEET block (excluding "Total", handled separately).
BS_LABELS = [
    "Equity Share Capital",
    "Reserves",
    "Borrowings",
    "Other Liabilities",
    "Net Block",
    "Investments",
    "Other Assets",
    "Receivables",
    "Inventory",
    "Cash & Bank",
    "No. of Equity Shares",
]

CASHFLOW_LABELS = [
    "Cash from Operating Activity",
    "Cash from Investing Activity",
    "Cash from Financing Activity",
    "Net Cash Flow",
]


def _norm(s: object) -> str:
    return str(s).strip().rstrip(":").upper() if pd.notna(s) else ""


def _find_section_bounds(col0: pd.Series) -> dict[str, tuple[int, int]]:
    """Map each known section header to its (start_row, end_row) exclusive-of-header range."""
    header_rows: dict[str, int] = {}
    for i, val in col0.items():
        norm = _norm(val)
        for header in SECTION_HEADERS:
            if norm == header.upper() and header not in header_rows:
                header_rows[header] = i
    ordered = sorted(header_rows.items(), key=lambda kv: kv[1])
    bounds: dict[str, tuple[int, int]] = {}
    for idx, (name, row) in enumerate(ordered):
        end = ordered[idx + 1][1] if idx + 1 < len(ordered) else len(col0)
        bounds[name] = (row + 1, end)
    return bounds


def _find_label_row(col0: pd.Series, label: str, start: int, end: int) -> int | None:
    target = _norm(label)
    window = col0.iloc[start:end]
    for i, val in window.items():
        if _norm(val) == target:
            return i
    return None


def _find_all_label_rows(col0: pd.Series, label: str, start: int, end: int) -> list[int]:
    target = _norm(label)
    window = col0.iloc[start:end]
    return [i for i, val in window.items() if _norm(val) == target]


def _row_as_numeric_series(df: pd.DataFrame, row: int, ncols: int) -> np.ndarray:
    vals = df.iloc[row, 1 : 1 + ncols]
    return pd.to_numeric(vals, errors="coerce").to_numpy()


@dataclass
class ParsedFundamentals:
    company_name: str | None
    annual: pd.DataFrame  # index = FY end date, columns = line items


def parse_fundamentals_file(path: Path) -> ParsedFundamentals:
    """Parse one Screener.in xlsx's Data Sheet into an annual (FY-indexed) DataFrame."""
    raw = pd.read_excel(path, sheet_name="Data Sheet", header=None)
    col0 = raw[0]

    company_name = None
    name_row = _find_label_row(col0, "COMPANY NAME", 0, len(col0))
    if name_row is not None:
        company_name = raw.iloc[name_row, 1]

    bounds = _find_section_bounds(col0)
    if "PROFIT & LOSS" not in bounds or "BALANCE SHEET" not in bounds:
        raise ValueError(f"{path.name}: could not locate PROFIT & LOSS / BALANCE SHEET sections")

    pnl_start, pnl_end = bounds["PROFIT & LOSS"]
    date_row = _find_label_row(col0, "Report Date", pnl_start, pnl_end)
    if date_row is None:
        raise ValueError(f"{path.name}: no Report Date row in PROFIT & LOSS block")
    dates_raw = raw.iloc[date_row, 1:]
    ncols = int(dates_raw.notna().sum())
    fy_end = pd.to_datetime(dates_raw.iloc[:ncols], errors="coerce")

    data: dict[str, np.ndarray] = {}

    for label in PNL_LABELS:
        row = _find_label_row(col0, label, pnl_start, pnl_end)
        data[label] = _row_as_numeric_series(raw, row, ncols) if row is not None else np.full(ncols, np.nan)

    bs_start, bs_end = bounds["BALANCE SHEET"]
    for label in BS_LABELS:
        row = _find_label_row(col0, label, bs_start, bs_end)
        data[label] = _row_as_numeric_series(raw, row, ncols) if row is not None else np.full(ncols, np.nan)

    total_rows = _find_all_label_rows(col0, "Total", bs_start, bs_end)
    if len(total_rows) >= 2:
        data["Total Liabilities & Equity"] = _row_as_numeric_series(raw, total_rows[0], ncols)
        data["Total Assets"] = _row_as_numeric_series(raw, total_rows[1], ncols)
    elif len(total_rows) == 1:
        logger.warning("%s: only one 'Total' row found in BALANCE SHEET block, using it for both sides", path.name)
        vals = _row_as_numeric_series(raw, total_rows[0], ncols)
        data["Total Liabilities & Equity"] = vals
        data["Total Assets"] = vals
    else:
        logger.warning("%s: no 'Total' row found in BALANCE SHEET block", path.name)
        data["Total Liabilities & Equity"] = np.full(ncols, np.nan)
        data["Total Assets"] = np.full(ncols, np.nan)

    if "CASH FLOW" in bounds:
        cf_start, cf_end = bounds["CASH FLOW"]
        for label in CASHFLOW_LABELS:
            row = _find_label_row(col0, label, cf_start, cf_end)
            data[label] = _row_as_numeric_series(raw, row, ncols) if row is not None else np.full(ncols, np.nan)
    else:
        for label in CASHFLOW_LABELS:
            data[label] = np.full(ncols, np.nan)

    annual = pd.DataFrame(data, index=fy_end.iloc[:ncols])
    annual.index.name = "FiscalYearEnd"
    annual = annual[~annual.index.isna()].sort_index()

    total_check = (annual["Total Assets"] - annual["Total Liabilities & Equity"]).abs()
    tolerance = 0.005 * annual["Total Assets"].abs().clip(lower=1)
    mismatched = total_check > tolerance
    if mismatched.any():
        logger.warning(
            "%s: balance sheet Total Assets != Total Liabilities+Equity for FY(s) %s",
            path.name,
            list(annual.index[mismatched].strftime("%Y-%m-%d")),
        )

    return ParsedFundamentals(company_name=company_name, annual=annual)


def parse_all_fundamentals(fundamentals_dir: Path | None = None) -> dict[str, ParsedFundamentals]:
    """Parse every xlsx in fundamentals/, keyed by filename stem. Skips files that fail to parse."""
    fundamentals_dir = fundamentals_dir or config.FUNDAMENTALS_DIR
    results: dict[str, ParsedFundamentals] = {}
    for path in sorted(fundamentals_dir.glob("*.xlsx")):
        try:
            results[path.stem] = parse_fundamentals_file(path)
        except Exception:
            logger.exception("Failed to parse %s", path.name)
    return results
