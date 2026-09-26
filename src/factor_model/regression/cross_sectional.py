"""Daily cross-sectional WLS regression: the core of the Barra-style model.

Spec (see .claude/skills/statistics/SKILL.md): for each date t,
  y = excess return over (t-1, t]
  X = [10 industry exposure weights, no separate intercept] + [10 z-scored style factors]
  weight = sqrt(market_cap)
Exposures and weights are dated t-1, predicting the return realized over
(t-1, t] - never same-day-as-return. Dates with fewer than MIN_N_PER_DAY
complete-case observations are skipped and logged. Observations with a
single-day |return| above CORPORATE_ACTION_RETURN_THRESHOLD are excluded
from that day's fit and logged separately (see CLAUDE.md "Known
limitations" - we don't reconstruct corporate-action-adjusted returns,
we just keep one bad observation from contaminating a whole day's factor
returns and that stock's idiosyncratic-risk estimate).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from factor_model import config

logger = logging.getLogger(__name__)

DESIGN_COLS = config.INDUSTRY_FACTORS + config.STYLE_FACTORS + config.INDICATOR_FACTORS


@dataclass
class CrossSectionalResults:
    factor_returns: pd.DataFrame  # Date index, design_cols columns (DESIGN_COLS by default)
    r2_daily: pd.DataFrame  # Date, N, R2, Adj_R2
    residuals_long: pd.DataFrame  # Date, Symbol, ActualReturn, Fitted, Residual
    flagged_corporate_actions: pd.DataFrame = field(default_factory=pd.DataFrame)
    skipped_low_n_dates: list = field(default_factory=list)
    skipped_degenerate_dates: list = field(default_factory=list)
    coverage_start_date: pd.Timestamp | None = None


def _build_regression_input(exposure_panel: pd.DataFrame, returns_wide: pd.DataFrame) -> pd.DataFrame:
    """Join t-1 exposures to the (t-1, t] return, per the timing convention above."""
    dates = sorted(exposure_panel["Date"].unique())
    date_to_target = dict(zip(dates[:-1], dates[1:]))

    exposures = exposure_panel.copy()
    exposures["target_date"] = exposures["Date"].map(date_to_target)
    exposures = exposures.dropna(subset=["target_date"])

    returns_wide = returns_wide.copy()
    returns_wide.index.name = "Date"
    returns_long = returns_wide.reset_index().melt(id_vars="Date", var_name="Symbol", value_name="return")
    returns_long = returns_long.rename(columns={"Date": "target_date"})

    merged = exposures.merge(returns_long, on=["target_date", "Symbol"], how="inner")
    return merged


def run_cross_sectional_regression(
    exposure_panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    min_n: int = config.MIN_N_PER_DAY,
    rf_annual: float = config.RF_ANNUAL,
    corporate_action_threshold: float = config.CORPORATE_ACTION_RETURN_THRESHOLD,
    design_cols: list[str] = DESIGN_COLS,
) -> CrossSectionalResults:
    """`design_cols` defaults to the module-level DESIGN_COLS (the full pooled-model spec) but
    a segment-specific regression (pipeline.run_segment) passes config.SEGMENT_DESIGN_COLS
    instead, since top100_flag (and any interaction with it) is constant/absent within a single
    segment and would make the design matrix singular.
    """
    merged = _build_regression_input(exposure_panel, returns_wide)

    rf_daily = rf_annual / config.TRADING_DAYS_PER_YEAR
    merged["excess_return"] = merged["return"] - rf_daily

    flagged_mask = merged["return"].abs() > corporate_action_threshold
    flagged = merged.loc[flagged_mask, ["target_date", "Symbol", "return"]].rename(
        columns={"target_date": "Date", "return": "raw_return"}
    )
    if flagged_mask.any():
        logger.warning("Flagged %d likely-corporate-action observations (|return| > %.0f%%)", flagged_mask.sum(), corporate_action_threshold * 100)
    clean = merged.loc[~flagged_mask].copy()

    required_cols = design_cols + ["excess_return", "market_cap"]
    clean = clean.dropna(subset=required_cols)
    clean = clean[clean["market_cap"] > 0]

    coef_rows: list[dict] = []
    r2_rows: list[dict] = []
    residual_frames: list[pd.DataFrame] = []
    skipped: list[pd.Timestamp] = []
    degenerate_dates: list[pd.Timestamp] = []

    for target_date, group in clean.groupby("target_date"):
        n = len(group)
        if n < min_n:
            skipped.append(target_date)
            continue

        if group["excess_return"].std() < config.DEGENERATE_RETURN_STD_THRESHOLD:
            # Every stock shows an (near-)identical return - almost always an exchange
            # holiday the business-day calendar doesn't know about, where the data source
            # just repeated the prior close for every ticker. Fitting this would produce an
            # undefined/exploding R2 and a degenerate coefficient solution; skip it instead.
            degenerate_dates.append(target_date)
            logger.warning("Skipping %s: zero cross-sectional return variance (likely an unflagged holiday)", target_date)
            continue

        X = group[design_cols].to_numpy(dtype=float)
        y = group["excess_return"].to_numpy(dtype=float)
        w = np.sqrt(group["market_cap"].to_numpy(dtype=float))

        model = sm.WLS(y, X, weights=w).fit()

        # Deliberately excludes "N" - factor_returns must contain only the DESIGN_COLS
        # factor coefficients, since it's consumed directly by summarize_factor_returns,
        # the EWMA covariance step, and the stationarity validation test, all of which would
        # otherwise treat an "N" column as if it were a 21st factor. N lives in r2_daily and
        # is re-joined only when writing the CSV output (regression/results.py).
        coef_rows.append({"Date": target_date, **dict(zip(design_cols, model.params))})
        r2_rows.append(
            {"Date": target_date, "N": n, "R2": model.rsquared, "Adj_R2": model.rsquared_adj}
        )
        residual_frames.append(
            pd.DataFrame(
                {
                    "Date": target_date,
                    "Symbol": group["Symbol"].to_numpy(),
                    "ActualReturn": y,
                    "Fitted": model.fittedvalues,
                    "Residual": model.resid,
                }
            )
        )

    if not coef_rows:
        raise RuntimeError(
            "No dates had enough complete-case observations to run the regression - "
            "check factor coverage / min_n threshold."
        )

    factor_returns = pd.DataFrame(coef_rows).set_index("Date").sort_index()
    r2_daily = pd.DataFrame(r2_rows).sort_values("Date").reset_index(drop=True)
    residuals_long = pd.concat(residual_frames, ignore_index=True).sort_values(["Date", "Symbol"]).reset_index(drop=True)

    coverage_start = factor_returns.index.min()
    if skipped:
        logger.info("Skipped %d dates with fewer than %d complete-case observations", len(skipped), min_n)
    if degenerate_dates:
        logger.info("Skipped %d degenerate (zero-variance) dates: %s", len(degenerate_dates), degenerate_dates)
    logger.info("Regression coverage starts %s, runs through %s (%d dates)", coverage_start, factor_returns.index.max(), len(factor_returns))

    return CrossSectionalResults(
        factor_returns=factor_returns,
        r2_daily=r2_daily,
        residuals_long=residuals_long,
        flagged_corporate_actions=flagged.reset_index(drop=True),
        skipped_low_n_dates=skipped,
        skipped_degenerate_dates=degenerate_dates,
        coverage_start_date=coverage_start,
    )
