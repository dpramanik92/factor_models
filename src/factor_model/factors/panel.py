"""Assemble the full factor exposure panel: every style/industry factor joined
onto a complete (Date x Symbol) grid, with the uniform winsorize+zscore pass
applied to the style factors. Writes model_data/factor_exposure_panel.parquet
and model_data/returns_panel.parquet - the stable artifacts every downstream
step (regression, risk, validation) reads from.
"""

from __future__ import annotations

import logging

import pandas as pd

from factor_model import config, standardize
from factor_model.factors.beta import compute_beta
from factor_model.factors.capex import compute_capex
from factor_model.factors.growth import compute_growth
from factor_model.factors.idio_vol import compute_idio_vol
from factor_model.factors.industry import expand_industry_exposure_daily, load_industry_exposure
from factor_model.factors.leverage import compute_leverage
from factor_model.factors.momentum import compute_momentum
from factor_model.factors.profitability import compute_profitability
from factor_model.factors.quality import compute_quality
from factor_model.factors.size import compute_size
from factor_model.factors.value import compute_value
from factor_model.io.loaders import load_market_cap, load_returns

logger = logging.getLogger(__name__)


def _full_grid(universe: list[str], business_day_index: pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product([business_day_index, universe], names=["Date", "Symbol"])
    return idx.to_frame(index=False)


def build_returns_panel(universe: list[str]) -> pd.DataFrame:
    """Daily simple returns for the universe, saved to model_data/returns_panel.parquet."""
    returns = load_returns(universe)
    config.MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(config.RETURNS_PANEL_FILE)
    logger.info("Wrote returns panel: %s (%s)", config.RETURNS_PANEL_FILE, returns.shape)
    return returns


def build_exposure_panel(
    universe: list[str],
    annual_ratios: pd.DataFrame,
    business_day_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build and save the full (Date, Symbol, <10 industry weights>, <10 style factors>,
    size_quintile, market_cap) exposure panel.
    """
    panel = _full_grid(universe, business_day_index)

    raw_frames = [
        compute_beta(universe),
        compute_idio_vol(universe),
        compute_momentum(universe),
        compute_size(universe),  # contributes 'size' and 'size_quintile'
        compute_value(universe),
        compute_leverage(annual_ratios, business_day_index),
        compute_quality(annual_ratios, business_day_index),
        compute_capex(annual_ratios, business_day_index),
        compute_growth(annual_ratios, business_day_index),
        compute_profitability(annual_ratios, business_day_index),
    ]
    for frame in raw_frames:
        panel = panel.merge(frame, on=["Date", "Symbol"], how="left")

    industry_by_symbol = load_industry_exposure(universe)
    industry_long = expand_industry_exposure_daily(industry_by_symbol, business_day_index)
    panel = panel.merge(industry_long, on=["Date", "Symbol"], how="left")

    market_cap_wide = load_market_cap(universe)
    market_cap_wide.index.name = "Date"
    market_cap_long = market_cap_wide.reset_index().melt(
        id_vars="Date", var_name="Symbol", value_name="market_cap"
    )
    panel = panel.merge(market_cap_long, on=["Date", "Symbol"], how="left")

    for col in config.STYLE_FACTORS:
        panel[col] = standardize.cross_sectional_winsorize_zscore(panel, "Date", col)

    config.MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(config.FACTOR_EXPOSURE_PANEL_FILE, index=False)
    logger.info("Wrote factor exposure panel: %s (%s)", config.FACTOR_EXPOSURE_PANEL_FILE, panel.shape)
    return panel
