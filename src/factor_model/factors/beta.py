"""Beta factor: rolling market-model beta vs NIFTY50, computed in-house from price data
(factors/market_model.py) so extending price history extends this factor's usable history too.
"""

from __future__ import annotations

import pandas as pd

from factor_model.factors._util import wide_to_long
from factor_model.factors.market_model import compute_rolling_beta_and_idio_vol


def compute_beta(universe: list[str]) -> pd.DataFrame:
    """Returns long (Date, Symbol, beta) for the given universe."""
    beta, _ = compute_rolling_beta_and_idio_vol(universe)
    return wide_to_long(beta, "beta")
