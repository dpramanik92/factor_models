"""Idiosyncratic volatility factor: reused from data/nse198_idio_vol_full."""

from __future__ import annotations

import pandas as pd

from factor_model.factors._util import wide_to_long
from factor_model.io.loaders import load_idio_vol


def compute_idio_vol(universe: list[str]) -> pd.DataFrame:
    """Returns long (Date, Symbol, idio_vol) for the given universe."""
    wide = load_idio_vol(universe)
    return wide_to_long(wide, "idio_vol")
