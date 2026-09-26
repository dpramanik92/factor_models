"""Loaders for the raw wide-format time-series parquet files in data/.

Every loader returns a (Date index, Symbol columns) DataFrame, optionally
subset to a given universe of symbols. data/ is read-only - see
.claude/rules/data-boundaries.md.
"""

from __future__ import annotations

import logging

import pandas as pd

from factor_model import config

logger = logging.getLogger(__name__)


def _load_wide_parquet(filename: str, universe: list[str] | None = None) -> pd.DataFrame:
    path = config.DATA_DIR / filename
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    if universe is not None:
        missing = sorted(set(universe) - set(df.columns))
        if missing:
            logger.warning("%s missing %d of %d universe symbols: %s", filename, len(missing), len(universe), missing)
        cols = [c for c in universe if c in df.columns]
        df = df[cols]
    return df


def load_prices(universe: list[str] | None = None) -> pd.DataFrame:
    """Daily close prices. Includes a NIFTY50 benchmark column when no universe filter is given."""
    return _load_wide_parquet("nse200_nifty50_daily_prices.parquet", universe)


def load_benchmark_prices() -> pd.Series:
    """NIFTY50 benchmark daily close price series."""
    df = _load_wide_parquet("nse200_nifty50_daily_prices.parquet")
    return df["NIFTY50"]


def load_returns(universe: list[str] | None = None) -> pd.DataFrame:
    """Daily simple returns computed from close prices."""
    prices = load_prices(universe)
    if "NIFTY50" in prices.columns and (universe is None or "NIFTY50" not in universe):
        prices = prices.drop(columns=["NIFTY50"])
    return prices.pct_change()


def load_benchmark_returns() -> pd.Series:
    return load_benchmark_prices().pct_change()


def load_volume(universe: list[str] | None = None) -> pd.DataFrame:
    """Daily traded share volume - used by factors/liquidity.py's turnover calculation."""
    return _load_wide_parquet("nse198_daily_volume.parquet", universe)
