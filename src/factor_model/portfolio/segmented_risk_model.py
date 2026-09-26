"""Combines the two independently-fit segment models (output/segments/top100,
output/segments/rest - see pipeline.run_segment / `run-segments`) into a single portfolio-ready
risk model spanning the full universe, rather than optimizing over one segment's fit and ignoring
the other's stocks.

Each segment fits its own coefficients daily (config.SEGMENT_DESIGN_COLS: industry + style +
psu_flag, no top100_flag), so its own daily factor-return series is a genuinely different time
series from the other segment's, even though the column *names* (e.g. "beta", "momentum") are
shared. To build one N-stock covariance matrix without conflating the two: every design column is
suffixed by segment (__top100 / __rest) into a joint 2K-column factor space, a stock's exposure
row is zero everywhere except the block for its own current-date segment (from `top100_flag` on
the latest exposure-panel date - a stock cannot load on the other segment's factors), and the
joint 2K x 2K factor covariance is estimated by ordinary EWMA covariance
(risk.ewma_covariance.compute_ewma_factor_covariance) over the two segments' factor-return series
concatenated column-wise and aligned by Date. This recovers the empirical cross-segment
covariance (how the large-cap fit's "momentum" moves with the small-cap fit's "momentum" on the
same calendar day) rather than assuming the two segments' risks are independent, while still
giving each segment's within-block covariance its own dedicated fit. See CLAUDE.md's "Known
limitations" for why the segmented model exists at all, and portfolio/pipeline.py's
run_segmented_portfolio_optimization for how this module is used.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from factor_model import config
from factor_model.risk.ewma_covariance import compute_ewma_factor_covariance

logger = logging.getLogger(__name__)


def joint_design_cols(design_cols: list[str] = config.SEGMENT_DESIGN_COLS) -> list[str]:
    """The 2K suffixed column names (e.g. 'beta__top100', 'beta__rest', ...) spanning both
    segments' factor spaces.
    """
    return [f"{c}__{label}" for label in config.SEGMENT_LABELS for c in design_cols]


def get_latest_segment_exposures(
    exposure_panel: pd.DataFrame, label: str, design_cols: list[str] = config.SEGMENT_DESIGN_COLS
) -> pd.DataFrame:
    """Latest-date, complete-case exposures (raw, unsuffixed column names) for symbols currently
    in segment `label` (top100_flag == config.SEGMENT_TOP100_FLAG_VALUE[label]) - mirrors
    portfolio.risk_model.get_latest_exposures, scoped to one segment and its design columns.
    `exposure_panel` may already be date-truncated by the caller (e.g. a walk-forward backtest
    passing an as-of slice) - the "latest date" is always just this frame's own max Date.
    """
    latest_date = exposure_panel["Date"].max()
    flag_value = config.SEGMENT_TOP100_FLAG_VALUE[label]
    latest = exposure_panel.loc[
        (exposure_panel["Date"] == latest_date) & (exposure_panel["top100_flag"] == flag_value)
    ].set_index("Symbol")[design_cols]

    complete = latest.dropna()
    excluded = sorted(set(latest.index) - set(complete.index))
    if excluded:
        logger.warning(
            "Excluding %d symbols from the %s segment's optimizable universe (incomplete factor "
            "exposure on the latest date %s): %s",
            len(excluded), label, latest_date.date(), excluded,
        )
    return complete


def build_joint_exposures(
    exposures_by_segment: dict[str, pd.DataFrame], design_cols: list[str] = config.SEGMENT_DESIGN_COLS
) -> pd.DataFrame:
    """Block-diagonal N x 2K exposure matrix: each stock's row is nonzero only in its own
    segment's block, zero in the other - a stock cannot load on a factor it wasn't fit against.
    Raises if a symbol appears in both segments (shouldn't happen - top100_flag partitions the
    universe on any given date).
    """
    cols = joint_design_cols(design_cols)
    frames = []
    for label, exposures in exposures_by_segment.items():
        block = pd.DataFrame(0.0, index=exposures.index, columns=cols)
        block[[f"{c}__{label}" for c in design_cols]] = exposures[design_cols].to_numpy()
        frames.append(block)
    joint = pd.concat(frames, axis=0)
    dup = joint.index[joint.index.duplicated()]
    if len(dup):
        raise ValueError(f"Symbol(s) present in more than one segment's current exposures: {list(dup)}")
    return joint


def build_joint_factor_covariance(
    factor_returns_by_segment: dict[str, pd.DataFrame],
    design_cols: list[str] = config.SEGMENT_DESIGN_COLS,
    halflife: int = config.EWMA_HALFLIFE_FACTOR,
) -> pd.DataFrame:
    """The 2K x 2K joint EWMA factor covariance: each segment's own K x K block plus the
    empirical cross-segment block, from a single EWMA covariance pass over both segments' daily
    factor-return series stacked column-wise (aligned by Date, inner join - a date present in
    only one segment's series can't contribute a cross-segment observation that day).
    """
    renamed = []
    for label, factor_returns in factor_returns_by_segment.items():
        r = factor_returns[design_cols].copy()
        r.columns = [f"{c}__{label}" for c in design_cols]
        renamed.append(r)
    joint_returns = pd.concat(renamed, axis=1, join="inner")
    return compute_ewma_factor_covariance(joint_returns, halflife=halflife)


def build_joint_specific_risk(
    specific_risk_by_segment: dict[str, pd.DataFrame], symbols_by_segment: dict[str, list[str]]
) -> pd.DataFrame:
    """Concatenates each segment's specific-risk table, restricted to the symbols actually
    assigned to that segment as of today - a stock's specific-risk row from a segment it has
    since left (e.g. it crossed the top-100 threshold since) is never reused.
    """
    frames = [
        specific_risk[specific_risk["Symbol"].isin(symbols_by_segment[label])]
        for label, specific_risk in specific_risk_by_segment.items()
    ]
    return pd.concat(frames, axis=0, ignore_index=True)


def build_max_stock_weight(
    symbols_by_segment: dict[str, list[str]],
    max_stock_weight_by_segment: dict[str, float] | None = None,
) -> float | pd.Series:
    """Per-symbol single-stock soft-cap Series, letting the "rest" (non-top100) segment carry a
    looser cap than the top100 segment - e.g. `{"top100": 0.10, "rest": 0.20}`, tried after the
    uniform 10% cap (config.MAX_STOCK_WEIGHT) was suspected of over-diversifying the rest
    segment's smaller, less numerous universe relative to what its own risk/return model would
    otherwise pick. Returns the plain scalar config.MAX_STOCK_WEIGHT unchanged (not a Series) when
    `max_stock_weight_by_segment` is None, so the default segmented run's behavior/output is
    byte-identical to before this was added - see
    portfolio/optimize.py's module docstring for how a Series cap plugs into the optimizer.
    """
    if max_stock_weight_by_segment is None:
        return config.MAX_STOCK_WEIGHT
    caps = {}
    for label, symbols in symbols_by_segment.items():
        for symbol in symbols:
            caps[symbol] = max_stock_weight_by_segment[label]
    return pd.Series(caps)


def build_group_exposure(symbols_by_segment: dict[str, list[str]], label: str) -> pd.Series:
    """A 0/1 Symbol-indexed membership vector for segment `label` (1.0 for its own symbols, 0.0
    for every other segment's) - the group_exposure input to
    portfolio/optimize.py's group-level soft cap (e.g. capping total "rest"-segment weight).
    """
    all_symbols = [s for symbols in symbols_by_segment.values() for s in symbols]
    membership = {s: (1.0 if s in symbols_by_segment[label] else 0.0) for s in all_symbols}
    return pd.Series(membership)


def build_segmented_stock_covariance(
    exposures_joint: pd.DataFrame, factor_covariance_joint: pd.DataFrame, specific_risk_joint: pd.DataFrame
) -> pd.DataFrame:
    """Sigma = X F X' + D, same construction as portfolio.risk_model.build_stock_covariance, but
    against the joint 2K-column design instead of the pooled model's single-K-column one. Stocks
    with no specific-risk estimate are dropped, not imputed - same convention as the pooled model.
    """
    spec = specific_risk_joint.set_index("Symbol")["specific_risk_annualized"]
    symbols = [s for s in exposures_joint.index if s in spec.index and pd.notna(spec.loc[s])]
    missing_spec = sorted(set(exposures_joint.index) - set(symbols))
    if missing_spec:
        logger.warning("Excluding %d symbols with no specific-risk estimate: %s", len(missing_spec), missing_spec)

    cols = list(exposures_joint.columns)
    X = exposures_joint.loc[symbols, cols].to_numpy()
    F = factor_covariance_joint.loc[cols, cols].to_numpy()
    factor_risk = X @ F @ X.T

    daily_specific_var = (spec.loc[symbols].to_numpy() ** 2) / config.TRADING_DAYS_PER_YEAR
    Sigma = factor_risk + np.diag(daily_specific_var)
    Sigma = (Sigma + Sigma.T) / 2
    return pd.DataFrame(Sigma, index=symbols, columns=symbols)
