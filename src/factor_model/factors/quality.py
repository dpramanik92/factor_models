"""Quality factor: composite of ROA and ROE ("ROI"), both computed from fundamentals for
point-in-time consistency (see CLAUDE.md decision on ROI=ROE, recomputed rather than reused
from data/nse198_roe_full_pointintime.parquet). Each subcomponent is standardized per date
before averaging; the raw composite is returned for panel.py's uniform final z-score pass,
matching the convention in factors/value.py.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from factor_model.factors._util import wide_to_long
from factor_model.factors.pit import expand_frame_to_daily
from factor_model.standardize import winsorize_zscore


def compute_quality(annual_ratios: pd.DataFrame, business_day_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Returns long (Date, Symbol, quality) for every symbol present in annual_ratios."""
    roa_by_symbol = {
        symbol: group.set_index("FiscalYearEnd")["ROA"] for symbol, group in annual_ratios.groupby("Symbol")
    }
    roe_by_symbol = {
        symbol: group.set_index("FiscalYearEnd")["ROE"] for symbol, group in annual_ratios.groupby("Symbol")
    }
    roa_daily = expand_frame_to_daily(roa_by_symbol, business_day_index)
    roe_daily = expand_frame_to_daily(roe_by_symbol, business_day_index)

    roa_z = roa_daily.apply(lambda row: winsorize_zscore(row), axis=1)
    roe_z = roe_daily.apply(lambda row: winsorize_zscore(row), axis=1)

    stacked = np.stack([roa_z.to_numpy(), roe_z.to_numpy()])
    n_valid = np.sum(~np.isnan(stacked), axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        composite = np.nanmean(stacked, axis=0)
    composite = np.where(n_valid >= 1, composite, np.nan)

    composite_df = pd.DataFrame(composite, index=roa_daily.index, columns=roa_daily.columns)
    composite_df.index.name = "Date"
    return wide_to_long(composite_df, "quality")
