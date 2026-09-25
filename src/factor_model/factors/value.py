"""Value factor: composite of earnings yield, book yield, and dividend yield.

E/P = 1/PE and B/P = 1/PB are used instead of raw PE/PB to avoid the sign
flips and blow-ups that raw P/E or P/B ratios produce near zero or negative
earnings/book value. Each subcomponent is winsorized and z-scored per date,
then averaged (requiring at least 2 of 3 non-missing) and re-z-scored into a
single Value factor - see .claude/skills/statistics/SKILL.md.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from factor_model.factors._util import wide_to_long
from factor_model.io.loaders import load_dividend_yield, load_pb, load_pe
from factor_model.standardize import winsorize_zscore


def compute_value(universe: list[str]) -> pd.DataFrame:
    """Returns long (Date, Symbol, value) for the given universe: the raw (pre-final-zscore)
    composite of the three standardized subcomponents (E/P, B/P, dividend yield). Subcomponent
    standardization happens here (needed to combine incommensurable ratios); the final
    re-standardization of the composite happens uniformly in factors/panel.py.
    """
    pe = load_pe(universe)
    pb = load_pb(universe)
    div_yield = load_dividend_yield(universe)

    ep = 1.0 / pe.where(pe > 0)
    bp = 1.0 / pb.where(pb > 0)
    dy = div_yield.where(div_yield >= 0)

    # Align all three to the same (Date, Symbol) grid before combining.
    common_index = ep.index.union(bp.index).union(dy.index)
    common_cols = sorted(set(ep.columns) | set(bp.columns) | set(dy.columns))
    ep = ep.reindex(index=common_index, columns=common_cols)
    bp = bp.reindex(index=common_index, columns=common_cols)
    dy = dy.reindex(index=common_index, columns=common_cols)

    ep_z = ep.apply(lambda row: winsorize_zscore(row), axis=1)
    bp_z = bp.apply(lambda row: winsorize_zscore(row), axis=1)
    dy_z = dy.apply(lambda row: winsorize_zscore(row), axis=1)

    stacked = np.stack([ep_z.to_numpy(), bp_z.to_numpy(), dy_z.to_numpy()])
    n_valid = np.sum(~np.isnan(stacked), axis=0)
    with warnings.catch_warnings():
        # All-NaN slices are expected wherever a (date, symbol) has no valid subcomponents;
        # they're masked to NaN by the n_valid check below regardless of nanmean's own value.
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        composite = np.nanmean(stacked, axis=0)
    composite = np.where(n_valid >= 2, composite, np.nan)

    # Composite is an average of already-standardized subcomponents; the final
    # re-standardization happens uniformly in factors/panel.py alongside every
    # other style factor, so we return the raw composite here.
    composite_df = pd.DataFrame(composite, index=common_index, columns=common_cols)
    composite_df.index.name = "Date"

    return wide_to_long(composite_df, "value")
