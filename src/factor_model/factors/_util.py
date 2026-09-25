"""Shared helpers for factor construction modules."""

from __future__ import annotations

import pandas as pd


def wide_to_long(wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """(Date index, Symbol columns) -> long (Date, Symbol, value_name) frame.

    Forces the index name to "Date" before reset_index() so this is correct
    regardless of whether the caller's DataFrame index happened to be named
    - reset_index() on an unnamed index produces a column literally called
    "index", not "Date".
    """
    wide = wide.copy()
    wide.index.name = "Date"
    return wide.reset_index().melt(id_vars="Date", var_name="Symbol", value_name=value_name)
