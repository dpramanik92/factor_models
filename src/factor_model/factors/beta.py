"""Beta factor: rolling 1-year beta vs NIFTY50, reused from data/nse200_rolling_1y_beta."""

from __future__ import annotations

import pandas as pd

from factor_model.factors._util import wide_to_long
from factor_model.io.loaders import load_beta


def compute_beta(universe: list[str]) -> pd.DataFrame:
    """Returns long (Date, Symbol, beta) for the given universe."""
    wide = load_beta(universe)
    return wide_to_long(wide, "beta")
