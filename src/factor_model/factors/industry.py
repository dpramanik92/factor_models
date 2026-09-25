"""Industry exposure: the 10-bucket weighted matrix from nse_200_industry_exposure.xlsx.

Weights are static/time-invariant across the whole sample (a single snapshot,
no historical industry-weight series exists in the source data) - this is a
documented assumption, not an oversight. Weights sum to ~1 per stock and are
used in place of a regression intercept (Barra convention) - see
.claude/skills/statistics/SKILL.md.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from factor_model import config

logger = logging.getLogger(__name__)


def load_industry_exposure(universe: list[str]) -> pd.DataFrame:
    """Returns (Symbol index, industry columns) exposure weights for the given universe."""
    df = pd.read_excel(config.INDUSTRY_EXPOSURE_XLSX, sheet_name="Industry Exposure Matrix")
    df = df.set_index("Symbol")[config.INDUSTRY_FACTORS]
    missing = sorted(set(universe) - set(df.index))
    if missing:
        logger.warning("Industry exposure missing for %d universe symbols: %s", len(missing), missing)
    df = df.reindex(universe)

    row_sums = df.sum(axis=1)
    bad = (row_sums - 1.0).abs() > 1e-3
    if bad.any():
        logger.warning("Industry weights do not sum to 1 for: %s", list(df.index[bad]))

    return df


def expand_industry_exposure_daily(
    industry_by_symbol: pd.DataFrame, business_day_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Broadcast the static per-symbol industry weights across every date in the panel,
    returning a long (Date, Symbol, <industry columns>) frame.
    """
    n_dates = len(business_day_index)
    n_symbols = len(industry_by_symbol)
    values = np.tile(industry_by_symbol.to_numpy(), (n_dates, 1))
    long = pd.DataFrame(values, columns=industry_by_symbol.columns)
    long.insert(0, "Symbol", np.tile(industry_by_symbol.index.to_numpy(), n_dates))
    long.insert(0, "Date", np.repeat(business_day_index.to_numpy(), n_symbols))
    return long
