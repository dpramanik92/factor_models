"""Size factor: log(market cap), z-scored, plus a decile bucket (D1=smallest .. D10=largest).

log(market_cap) z-scored is the continuous regression exposure (Barra convention) - captures a
smooth, linear-in-log-cap size effect. size_decile itself is also fed into the regression, but
not from here directly: factors/panel.py dummy-encodes it into 9 indicator columns
(size_decile_2..size_decile_10, D1 dropped as the reference category - config.INDICATOR_FACTORS)
so the model can capture size-effect *nonlinearity* across the cap spectrum on top of the smooth
continuous term, without the crude single top100/rest split or the interaction-term/segmentation
approaches tried and set aside (see CLAUDE.md). size_decile is also still used standalone for the
D1-vs-D10 diagnostic spread test. See .claude/skills/statistics/SKILL.md for the standardization
convention. Deciles (not quintiles) give finer-grained resolution now that the universe has grown
to ~209 names (~20 stocks per decile vs. ~40 per quintile).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_model.factors._util import wide_to_long

N_SIZE_BUCKETS = 10


def compute_size(market_cap: pd.DataFrame) -> pd.DataFrame:
    """Returns long (Date, Symbol, size, size_decile) from a (Date, Symbol) market cap panel
    (see factors/market_cap.py - callers share one market-cap computation since it's also used
    as the WLS regression weight in factors/panel.py).

    'size' is raw log(market_cap) (not yet z-scored - that happens uniformly
    in factors/panel.py). 'size_decile' is an integer 1-10 label computed
    per date, kept for diagnostics only.
    """
    log_cap = np.log(market_cap.where(market_cap > 0))
    long = wide_to_long(log_cap, "size")

    def _decile(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if len(valid) < N_SIZE_BUCKETS:
            return pd.Series(np.nan, index=s.index)
        return pd.qcut(s.rank(method="first"), N_SIZE_BUCKETS, labels=list(range(1, N_SIZE_BUCKETS + 1)))

    deciles = log_cap.apply(_decile, axis=1)
    long_d = wide_to_long(deciles, "size_decile")
    long = long.merge(long_d, on=["Date", "Symbol"], how="left")
    return long


def add_size_decile_dummies(panel: pd.DataFrame, decile_col: str = "size_decile") -> pd.DataFrame:
    """Dummy-encodes `decile_col` into size_decile_2..size_decile_10 columns (D1/smallest
    dropped as the reference category - required so this doesn't perfectly collide with the
    industry weights, which already sum to 1 per row and serve as the regression's intercept
    replacement). NaN-safe: a NaN decile (insufficient cross-sectional coverage that date) stays
    NaN in every dummy, not 0 - that row is correctly excluded from the regression rather than
    silently treated as "not in this decile", which would be a false claim.
    """
    panel = panel.copy()
    is_nan = panel[decile_col].isna()
    for i in range(2, N_SIZE_BUCKETS + 1):
        panel[f"size_decile_{i}"] = (panel[decile_col] == i).astype(float)
        panel.loc[is_nan, f"size_decile_{i}"] = float("nan")
    return panel
