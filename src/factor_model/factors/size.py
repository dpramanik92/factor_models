"""Size factor: log(market cap), z-scored, plus a diagnostic-only quintile bucket.

log(market_cap) z-scored is the actual regression exposure (Barra
convention). A separate size_quintile (Q1=smallest .. Q5=largest) column is
also produced but kept out of the regression design matrix - it exists only
for diagnostics/reporting (e.g. model_reviewer's Q1-vs-Q5 spread test). See
.claude/skills/statistics/SKILL.md for why size is handled this way instead
of as a plain z-scored continuous factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_model.factors._util import wide_to_long
from factor_model.io.loaders import load_market_cap


def compute_size(universe: list[str]) -> pd.DataFrame:
    """Returns long (Date, Symbol, size, size_quintile) for the given universe.

    'size' is raw log(market_cap) (not yet z-scored - that happens uniformly
    in factors/panel.py). 'size_quintile' is an integer 1-5 label computed
    per date, kept for diagnostics only.
    """
    market_cap = load_market_cap(universe)
    log_cap = np.log(market_cap.where(market_cap > 0))
    long = wide_to_long(log_cap, "size")

    def _quintile(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if len(valid) < 5:
            return pd.Series(np.nan, index=s.index)
        return pd.qcut(s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

    quintiles = log_cap.apply(_quintile, axis=1)
    long_q = wide_to_long(quintiles, "size_quintile")
    long = long.merge(long_q, on=["Date", "Symbol"], how="left")
    return long
