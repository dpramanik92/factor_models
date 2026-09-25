"""Cross-sectional winsorization and z-scoring, applied per date.

Convention (see .claude/skills/statistics/SKILL.md): always winsorize before
z-scoring, NaN-safe throughout. Size (quintile diagnostic aside) and industry
weights are the two exceptions that don't go through zscore() as plain
continuous factors - see factors/panel.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_model import config


def winsorize(
    s: pd.Series,
    method: str = "percentile",
    bounds: tuple[float, float] = (config.WINSOR_LOWER_PCT, config.WINSOR_UPPER_PCT),
) -> pd.Series:
    """Clip a cross-sectional series to the given bounds. NaN-safe (NaNs pass through)."""
    valid = s.dropna()
    if valid.empty:
        return s
    if method == "percentile":
        lo, hi = valid.quantile(bounds[0]), valid.quantile(bounds[1])
    elif method == "std":
        mean, std = valid.mean(), valid.std()
        if std == 0 or pd.isna(std):
            return s
        lo, hi = mean + bounds[0] * std, mean + bounds[1] * std
    else:
        raise ValueError(f"unknown winsorize method: {method}")
    return s.clip(lower=lo, upper=hi)


def zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score. NaN-safe (NaNs stay NaN, never imputed)."""
    valid = s.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=s.index)
    mean, std = valid.mean(), valid.std()
    if std == 0 or pd.isna(std):
        return pd.Series(np.nan, index=s.index)
    return (s - mean) / std


def winsorize_zscore(
    s: pd.Series,
    method: str = "percentile",
    bounds: tuple[float, float] = (config.WINSOR_LOWER_PCT, config.WINSOR_UPPER_PCT),
) -> pd.Series:
    return zscore(winsorize(s, method=method, bounds=bounds))


def cross_sectional_winsorize_zscore(
    panel: pd.DataFrame,
    date_col: str,
    value_col: str,
    method: str = "percentile",
    bounds: tuple[float, float] = (config.WINSOR_LOWER_PCT, config.WINSOR_UPPER_PCT),
) -> pd.Series:
    """Apply winsorize_zscore per date group to a long-format (Date, Symbol, value) panel.

    Returns a Series aligned to panel's index.
    """
    return panel.groupby(date_col)[value_col].transform(lambda s: winsorize_zscore(s, method=method, bounds=bounds))
