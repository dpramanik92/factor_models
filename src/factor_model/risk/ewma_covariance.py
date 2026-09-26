"""EWMA factor covariance matrix from the daily factor-return time series.

.ewm(halflife=...).cov() (pandas' native pairwise EWMA covariance), default
half-life 90 trading days (config.EWMA_HALFLIFE_FACTOR) - see
.claude/skills/statistics/SKILL.md. Only the latest as-of-today K x K matrix
is persisted by default; a full daily time series of matrices is not (would
be unbounded in size).
"""

from __future__ import annotations

import logging

import pandas as pd

from factor_model import config
from factor_model.io.writers import save_csv

logger = logging.getLogger(__name__)


def compute_ewma_factor_covariance(
    factor_returns: pd.DataFrame, halflife: int = config.EWMA_HALFLIFE_FACTOR
) -> pd.DataFrame:
    """Latest K x K EWMA covariance matrix across all factors (industry + style)."""
    min_periods = min(2 * halflife, max(len(factor_returns) - 1, 1))
    ewma_cov = factor_returns.ewm(halflife=halflife, min_periods=min_periods).cov()
    latest_date = factor_returns.index.max()
    latest_cov = ewma_cov.loc[latest_date]
    latest_cov.index.name = "Factor"
    latest_cov.columns.name = "Factor"
    return latest_cov


def save_factor_covariance(cov: pd.DataFrame, paths: config.OutputPaths = config.DEFAULT_OUTPUT_PATHS) -> None:
    save_csv(cov, paths.factor_covariance_file, index=True)
    logger.info("Saved factor covariance matrix (%s) to %s", cov.shape, paths.factor_covariance_file)
