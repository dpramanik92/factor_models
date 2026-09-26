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
from factor_model.factors.liquidity import compute_liquidity
from factor_model.factors.market_cap import compute_market_cap
from factor_model.factors.momentum import compute_momentum
from factor_model.factors.profitability import compute_profitability
from factor_model.factors.psu import compute_psu_flag
from factor_model.factors.quality import compute_quality
from factor_model.factors.size import compute_size
from factor_model.factors.top100 import compute_top100_flag
from factor_model.factors.value import compute_value
from factor_model.io.loaders import load_returns

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
    """Build and save the full (Date, Symbol, <10 industry weights>, <11 style factors>,
    <2 indicator factors: top100_flag, psu_flag>, size_decile, market_cap) exposure panel.
    """
    panel = _full_grid(universe, business_day_index)

    # Computed once and shared: the size factor's raw input and the WLS regression weight
    # (sqrt(market_cap)) both need the same market cap panel - see factors/market_cap.py.
    market_cap_wide = compute_market_cap(annual_ratios, business_day_index)

    raw_frames = [
        compute_beta(universe),
        compute_idio_vol(universe),
        compute_momentum(universe),
        compute_size(market_cap_wide),  # contributes 'size' and 'size_decile'
        compute_value(annual_ratios, business_day_index),
        compute_leverage(annual_ratios, business_day_index),
        compute_quality(annual_ratios, business_day_index),
        compute_capex(annual_ratios, business_day_index),
        compute_growth(annual_ratios, business_day_index),
        compute_profitability(annual_ratios, business_day_index),
        compute_liquidity(universe, annual_ratios, business_day_index),
        compute_top100_flag(market_cap_wide),
        compute_psu_flag(universe, business_day_index),
    ]
    for frame in raw_frames:
        panel = panel.merge(frame, on=["Date", "Symbol"], how="left")

    industry_by_symbol = load_industry_exposure(universe)
    industry_long = expand_industry_exposure_daily(industry_by_symbol, business_day_index)
    panel = panel.merge(industry_long, on=["Date", "Symbol"], how="left")

    market_cap_wide.index.name = "Date"
    market_cap_long = market_cap_wide.reset_index().melt(
        id_vars="Date", var_name="Symbol", value_name="market_cap"
    )
    panel = panel.merge(market_cap_long, on=["Date", "Symbol"], how="left")

    for col in config.STYLE_FACTORS:
        panel[col] = standardize.cross_sectional_winsorize_zscore(panel, "Date", col)

    # add_size_decile_dummies (factors/size.py) was tried here and reverted - see
    # config.INDICATOR_FACTORS for why. Not called; kept as tested, working code in case it's
    # useful in a different combination later.

    # A single size x top100_flag interaction column was also tried here and reverted (see
    # CLAUDE.md's "Known limitations"): pooled residual alpha improved (0.42%->0.07% annualized)
    # but the segment-level bias this was meant to fix got *worse* (NW t 17.3/-7.8 -> 20.2/-17.6),
    # the same pooled-metric-masks-a-worse-segment-split pattern as the full interaction/decile-
    # dummy attempts. Not wired back in.

    config.MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(config.FACTOR_EXPOSURE_PANEL_FILE, index=False)
    logger.info("Wrote factor exposure panel: %s (%s)", config.FACTOR_EXPOSURE_PANEL_FILE, panel.shape)
    return panel
