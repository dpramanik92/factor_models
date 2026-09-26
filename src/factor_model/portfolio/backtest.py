"""Walk-forward, monthly-rebalanced backtest of the long-only MVO strategy - GMV, Max-Sharpe
(soft-capped - see optimize.py's stock_weight_penalty), and an Equal-Weight benchmark, all
re-optimized at each month-end using only information available as of that date (an expanding
window), then held fixed and marked to market over the following month. This is a genuine
point-in-time backtest, distinct from the single-snapshot static-weight bias check in
portfolio/validation.py (which applies *today's* weights to past returns, purely to sanity-check
the risk model's volatility prediction magnitude).

Kept deliberately to just these three (plus NIFTY50 for context, added in the pipeline layer):
Beta-Neutral and the long-short GMV/Max-Sharpe variants were tried (see optimize.py's
neutralize_exposures / minimize_variance_long_short, both still available) and backtested, but
neither beat simple long-only Max-Sharpe on a risk-adjusted basis - Beta-Neutral closet-indexed
the universe, and long-short underperformed its long-only counterpart net of extra turnover and
leverage. This backtest is the "clean" comparison after that finding.

Point-in-time note: the per-day factor return series (output/factor_returns_daily.csv) and the
daily factor exposure panel (model_data/factor_exposure_panel.parquet) are already point-in-time
by construction - each day's cross-sectional regression coefficient only uses that day's own
(already t-1 lagged) exposures and that day's own return, and fundamentals are forward-filled
only after their reporting lag (factors/pit.py). That means truncating these two already-built
artifacts at each rebalance date is sufficient to get point-in-time-correct inputs as of that
date - there is no need to re-run the cross-sectional regression itself at every rebalance date,
only the EWMA covariance/specific-risk/expected-return/optimization steps that sit on top of it.
"""

from __future__ import annotations

import logging

import pandas as pd

from factor_model import config
from factor_model.portfolio.expected_returns import compute_significance_shrunk_expected_returns, project_to_stock_returns
from factor_model.portfolio.optimize import pick_max_sharpe, trace_efficient_frontier
from factor_model.portfolio.risk_model import build_stock_covariance, get_latest_exposures
from factor_model.portfolio.segmented_risk_model import (
    build_group_exposure,
    build_joint_exposures,
    build_joint_factor_covariance,
    build_joint_specific_risk,
    build_max_stock_weight,
    build_segmented_stock_covariance,
    get_latest_segment_exposures,
    joint_design_cols,
)
from factor_model.risk.ewma_covariance import compute_ewma_factor_covariance
from factor_model.risk.specific_risk import compute_ewma_specific_risk

logger = logging.getLogger(__name__)


def _month_end_rebalance_dates(index: pd.DatetimeIndex, start: str) -> list[pd.Timestamp]:
    """The last available trading date in each calendar month, from the month containing `start`
    onward through the end of `index`.
    """
    idx = index[index >= pd.Timestamp(start).to_period("M").to_timestamp()]
    if idx.empty:
        return []
    periods = idx.to_period("M")
    return sorted(idx[periods == p].max() for p in periods.unique())


def run_walk_forward_backtest(
    start: str = "2025-01-01", frontier_n_points: int = config.BACKTEST_FRONTIER_N_POINTS
) -> dict:
    """Runs the walk-forward backtest from the last trading day at-or-before the month prior to
    `start` (used to form the first holding period's weights, so the holding periods themselves
    start exactly at `start`) through the latest available data. Requires `run-all` to have been
    run first (reads its output/model_data artifacts; does not rebuild them).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    required = [config.FACTOR_EXPOSURE_PANEL_FILE, config.FACTOR_RETURNS_FILE, config.RESIDUALS_LONG_FILE, config.RETURNS_PANEL_FILE]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing factor-model outputs, run `run-all` first: {missing}")

    exposure_panel = pd.read_parquet(config.FACTOR_EXPOSURE_PANEL_FILE)
    factor_returns_daily = pd.read_csv(config.FACTOR_RETURNS_FILE, parse_dates=["Date"]).set_index("Date")
    residuals_long = pd.read_parquet(config.RESIDUALS_LONG_FILE)
    returns_wide = pd.read_parquet(config.RETURNS_PANEL_FILE)

    lookback_start = (pd.Timestamp(start) - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    rebalance_dates = _month_end_rebalance_dates(returns_wide.index, lookback_start)
    if len(rebalance_dates) < 2:
        raise RuntimeError(f"Not enough monthly rebalance dates found from {lookback_start} onward")

    portfolio_names = ["GMV", "Max-Sharpe", "Equal-Weight"]
    daily_returns: dict[str, list[pd.Series]] = {name: [] for name in portfolio_names}
    weights_history: dict[str, dict[pd.Timestamp, pd.Series]] = {name: {} for name in portfolio_names}
    prev_weights: dict[str, pd.Series | None] = {name: None for name in portfolio_names}
    turnover_records = []

    n_periods = len(rebalance_dates) - 1
    for i in range(n_periods):
        as_of, period_end = rebalance_dates[i], rebalance_dates[i + 1]

        factor_returns_asof = factor_returns_daily.loc[:as_of]
        residuals_asof = residuals_long[residuals_long["Date"] <= as_of]
        exposure_asof = exposure_panel[exposure_panel["Date"] <= as_of]

        factor_covariance_asof = compute_ewma_factor_covariance(factor_returns_asof)
        specific_risk_asof = compute_ewma_specific_risk(residuals_asof)
        mu_factor_asof, _ = compute_significance_shrunk_expected_returns(factor_returns_asof)
        exposures_asof = get_latest_exposures(exposure_asof)
        mu_stock_asof = project_to_stock_returns(mu_factor_asof, exposures_asof)
        Sigma_asof = build_stock_covariance(exposures_asof, factor_covariance_asof, specific_risk_asof)

        symbols = list(Sigma_asof.index)
        mu_stock_asof = mu_stock_asof.reindex(symbols)
        exposures_asof = exposures_asof.reindex(symbols)

        frontier_df, gmv = trace_efficient_frontier(
            mu_stock_asof, Sigma_asof, n_points=frontier_n_points,
            industry_exposures=exposures_asof[config.INDUSTRY_FACTORS],
            max_stock_weight=config.MAX_STOCK_WEIGHT, max_industry_weight=config.MAX_INDUSTRY_WEIGHT,
            stock_weight_penalty=config.STOCK_WEIGHT_SOFT_PENALTY_COEF,
        )
        max_sharpe = pick_max_sharpe(frontier_df)
        equal_weight = pd.Series(1.0 / len(symbols), index=symbols)

        period_portfolios = {
            "GMV": gmv.weights,
            "Max-Sharpe": max_sharpe.weights,
            "Equal-Weight": equal_weight,
        }
        holding = returns_wide.loc[(returns_wide.index > as_of) & (returns_wide.index <= period_end)]

        for name, w in period_portfolios.items():
            weights_history[name][as_of] = w
            held_symbols = [s for s in w.index if s in holding.columns]
            w_norm = w.loc[held_symbols] / w.loc[held_symbols].sum()
            daily_returns[name].append((holding[held_symbols].fillna(0.0) @ w_norm.to_numpy()))

            if prev_weights[name] is not None:
                all_symbols = prev_weights[name].index.union(w.index)
                prev_full = prev_weights[name].reindex(all_symbols, fill_value=0.0)
                new_full = w.reindex(all_symbols, fill_value=0.0)
                turnover_records.append(
                    {"Date": as_of, "Portfolio": name, "Turnover": float((new_full - prev_full).abs().sum() / 2.0)}
                )
            prev_weights[name] = w

        logger.info(
            "Backtest rebalance %d/%d: %s -> %s (%d symbols)",
            i + 1, n_periods, as_of.date(), period_end.date(), len(symbols),
        )

    return_series = {name: pd.concat(series).sort_index() for name, series in daily_returns.items()}
    turnover_df = pd.DataFrame(turnover_records)

    return {
        "return_series": return_series,
        "weights_history": weights_history,
        "turnover": turnover_df,
        "rebalance_dates": rebalance_dates,
    }


def run_segmented_walk_forward_backtest(
    start: str = "2025-01-01",
    frontier_n_points: int = config.BACKTEST_FRONTIER_N_POINTS,
    max_stock_weight_by_segment: dict[str, float] | None = None,
    max_industry_weight: float | None = None,
    rest_segment_max_weight: float | None = None,
    rest_segment_weight_penalty: float | None = None,
) -> dict:
    """Same walk-forward, monthly-rebalanced methodology as run_walk_forward_backtest, but against
    the segmented (top100/rest) risk model (portfolio/segmented_risk_model.py) instead of the
    pooled one - see config.SEGMENT_DESIGN_COLS / CLAUDE.md's "Known limitations". Requires
    `run-segments` to have been run at least once (reads output/segments/<label>/'s
    factor_returns_daily.csv and residuals_long.parquet; does not rebuild them).

    `max_stock_weight_by_segment` (e.g. {"top100": 0.10, "rest": 0.20}), `max_industry_weight`
    (override/remove the hard per-industry cap), and `rest_segment_max_weight`/
    `rest_segment_weight_penalty` (a soft cap on total rest-segment weight) are all re-derived
    fresh at every rebalance date from that date's own symbols_by_segment (a stock's segment
    membership can drift over time) - see segmented_risk_model.build_max_stock_weight/
    build_group_exposure and portfolio/pipeline.py's run_segmented_portfolio_optimization for the
    full parameter semantics. All None (the defaults) reproduces the original uniform-10%-cap,
    hard-20%-industry-cap, no-group-cap behavior unchanged.

    Same point-in-time argument as the pooled backtest applies per segment: each segment's own
    daily factor-return/residual series is already point-in-time by construction (a given day's
    segment-scoped regression only used that day's own, already-lagged exposures and that day's
    own return), so truncating both segments' already-built artifacts at each rebalance date is
    sufficient - no need to re-run either segment's cross-sectional regression here.
    """
    effective_max_industry_weight = max_industry_weight if max_industry_weight is not None else config.MAX_INDUSTRY_WEIGHT
    effective_rest_segment_penalty = (
        rest_segment_weight_penalty if rest_segment_weight_penalty is not None else config.REST_SEGMENT_WEIGHT_SOFT_PENALTY_COEF
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    segment_paths = {label: config.OutputPaths(config.OUTPUT_DIR / "segments" / label) for label in config.SEGMENT_LABELS}
    required = [config.FACTOR_EXPOSURE_PANEL_FILE, config.RETURNS_PANEL_FILE]
    for paths in segment_paths.values():
        required += [paths.factor_returns_file, paths.residuals_long_file]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing segment-model outputs, run `run-segments` first: {missing}")

    exposure_panel = pd.read_parquet(config.FACTOR_EXPOSURE_PANEL_FILE)
    returns_wide = pd.read_parquet(config.RETURNS_PANEL_FILE)
    factor_returns_by_segment = {
        label: pd.read_csv(paths.factor_returns_file, parse_dates=["Date"]).set_index("Date")
        for label, paths in segment_paths.items()
    }
    residuals_by_segment = {label: pd.read_parquet(paths.residuals_long_file) for label, paths in segment_paths.items()}

    lookback_start = (pd.Timestamp(start) - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    rebalance_dates = _month_end_rebalance_dates(returns_wide.index, lookback_start)
    if len(rebalance_dates) < 2:
        raise RuntimeError(f"Not enough monthly rebalance dates found from {lookback_start} onward")

    jcols = joint_design_cols()
    portfolio_names = ["GMV", "Max-Sharpe", "Equal-Weight"]
    daily_returns: dict[str, list[pd.Series]] = {name: [] for name in portfolio_names}
    weights_history: dict[str, dict[pd.Timestamp, pd.Series]] = {name: {} for name in portfolio_names}
    prev_weights: dict[str, pd.Series | None] = {name: None for name in portfolio_names}
    turnover_records = []

    n_periods = len(rebalance_dates) - 1
    for i in range(n_periods):
        as_of, period_end = rebalance_dates[i], rebalance_dates[i + 1]
        exposure_asof = exposure_panel[exposure_panel["Date"] <= as_of]

        exposures_by_segment = {
            label: get_latest_segment_exposures(exposure_asof, label) for label in config.SEGMENT_LABELS
        }
        factor_covariance_asof = build_joint_factor_covariance(
            {label: factor_returns_by_segment[label].loc[:as_of] for label in config.SEGMENT_LABELS}
        )
        specific_risk_asof = {
            label: compute_ewma_specific_risk(residuals_by_segment[label][residuals_by_segment[label]["Date"] <= as_of])
            for label in config.SEGMENT_LABELS
        }
        symbols_by_segment = {label: list(exposures_by_segment[label].index) for label in config.SEGMENT_LABELS}
        specific_risk_joint_asof = build_joint_specific_risk(specific_risk_asof, symbols_by_segment)
        exposures_joint_asof = build_joint_exposures(exposures_by_segment)
        Sigma_asof = build_segmented_stock_covariance(exposures_joint_asof, factor_covariance_asof, specific_risk_joint_asof)
        max_stock_weight_asof = build_max_stock_weight(symbols_by_segment, max_stock_weight_by_segment)
        rest_group_exposure_asof = (
            build_group_exposure(symbols_by_segment, "rest") if rest_segment_max_weight is not None else None
        )

        mu_factor_parts = []
        for label in config.SEGMENT_LABELS:
            mu_factor_label, _ = compute_significance_shrunk_expected_returns(
                factor_returns_by_segment[label].loc[:as_of], design_cols=config.SEGMENT_DESIGN_COLS
            )
            mu_factor_label.index = [f"{c}__{label}" for c in mu_factor_label.index]
            mu_factor_parts.append(mu_factor_label)
        mu_factor_asof = pd.concat(mu_factor_parts)
        mu_stock_asof = project_to_stock_returns(mu_factor_asof, exposures_joint_asof, design_cols=jcols)

        symbols = list(Sigma_asof.index)
        mu_stock_asof = mu_stock_asof.reindex(symbols)
        industry_exposures_asof = pd.concat(
            [exposures_by_segment[label][config.INDUSTRY_FACTORS] for label in config.SEGMENT_LABELS], axis=0
        ).reindex(symbols)

        frontier_df, gmv = trace_efficient_frontier(
            mu_stock_asof, Sigma_asof, n_points=frontier_n_points,
            industry_exposures=industry_exposures_asof,
            max_stock_weight=max_stock_weight_asof, max_industry_weight=effective_max_industry_weight,
            stock_weight_penalty=config.STOCK_WEIGHT_SOFT_PENALTY_COEF,
            group_exposure=rest_group_exposure_asof,
            group_max_weight=rest_segment_max_weight if rest_segment_max_weight is not None else 1.0,
            group_weight_penalty=effective_rest_segment_penalty if rest_segment_max_weight is not None else None,
        )
        max_sharpe = pick_max_sharpe(frontier_df)
        equal_weight = pd.Series(1.0 / len(symbols), index=symbols)

        period_portfolios = {
            "GMV": gmv.weights,
            "Max-Sharpe": max_sharpe.weights,
            "Equal-Weight": equal_weight,
        }
        holding = returns_wide.loc[(returns_wide.index > as_of) & (returns_wide.index <= period_end)]

        for name, w in period_portfolios.items():
            weights_history[name][as_of] = w
            held_symbols = [s for s in w.index if s in holding.columns]
            w_norm = w.loc[held_symbols] / w.loc[held_symbols].sum()
            daily_returns[name].append((holding[held_symbols].fillna(0.0) @ w_norm.to_numpy()))

            if prev_weights[name] is not None:
                all_symbols = prev_weights[name].index.union(w.index)
                prev_full = prev_weights[name].reindex(all_symbols, fill_value=0.0)
                new_full = w.reindex(all_symbols, fill_value=0.0)
                turnover_records.append(
                    {"Date": as_of, "Portfolio": name, "Turnover": float((new_full - prev_full).abs().sum() / 2.0)}
                )
            prev_weights[name] = w

        logger.info(
            "Segmented backtest rebalance %d/%d: %s -> %s (%d symbols)",
            i + 1, n_periods, as_of.date(), period_end.date(), len(symbols),
        )

    return_series = {name: pd.concat(series).sort_index() for name, series in daily_returns.items()}
    turnover_df = pd.DataFrame(turnover_records)

    return {
        "return_series": return_series,
        "weights_history": weights_history,
        "turnover": turnover_df,
        "rebalance_dates": rebalance_dates,
    }
