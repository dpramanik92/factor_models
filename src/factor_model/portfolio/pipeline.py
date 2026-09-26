"""Orchestrates the portfolio optimization exercise: load the full-window factor model's
outputs -> expected returns + stock covariance -> GMV / efficient frontier / max-Sharpe -> save
everything under output/portfolio/ (mirrored to data/portfolio/ - io/writers.py) - never
touches the factor model's own output/ files. See cli.py's `optimize-portfolio` command.
"""

from __future__ import annotations

import logging

import pandas as pd

from factor_model import config
from factor_model.io.loaders import load_benchmark_returns
from factor_model.io.writers import save_csv
from factor_model.portfolio import analytics, plots as pplots
from factor_model.portfolio import validation as pvalidation
from factor_model.portfolio.backtest import run_segmented_walk_forward_backtest, run_walk_forward_backtest
from factor_model.portfolio.backtest_report import save_backtest_report
from factor_model.portfolio.expected_returns import compute_significance_shrunk_expected_returns, project_to_stock_returns
from factor_model.portfolio.optimize import pick_max_sharpe, trace_efficient_frontier
from factor_model.portfolio.report import save_report as save_portfolio_report
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

logger = logging.getLogger(__name__)


def _segmented_output_paths(
    rest_max_stock_weight: float | None = None,
    max_industry_weight: float | None = None,
    rest_segment_max_weight: float | None = None,
) -> dict:
    """Output/plots/report paths for a segmented portfolio run. The all-defaults case (uniform
    config.MAX_STOCK_WEIGHT single-stock cap, config.MAX_INDUSTRY_WEIGHT industry cap, no group-
    level rest-segment-total cap) writes to output/portfolio_segments/, unchanged from before any
    of these knobs existed. Any non-default combination writes to a distinct, suffixed sibling
    directory instead (e.g. output/portfolio_segments_rest20_noindcap_restcap20/), so trying a
    different set of caps never overwrites another combination's result - every experiment stays
    available side by side.
    """
    parts = []
    if rest_max_stock_weight is not None:
        parts.append(f"rest{int(round(rest_max_stock_weight * 100))}")
    if max_industry_weight is not None and max_industry_weight >= 1.0:
        parts.append("noindcap")
    if rest_segment_max_weight is not None:
        parts.append(f"restcap{int(round(rest_segment_max_weight * 100))}")

    output_dir = config.PORTFOLIO_SEGMENTS_OUTPUT_DIR if not parts else config.OUTPUT_DIR / ("portfolio_segments_" + "_".join(parts))
    return {
        "output_dir": output_dir,
        "plots_dir": output_dir / "plots",
        "validation_report_file": output_dir / "portfolio_validation_report.md",
        "backtest_output_dir": output_dir / "backtest",
        "backtest_report_file": output_dir / "backtest" / "backtest_report.md",
    }


def run_portfolio_optimization() -> dict:
    """Runs the full long-only mean-variance optimization exercise against the full-window
    factor model's already-built outputs. Raises if those don't exist yet - run `run-all` first.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    required = [config.FACTOR_EXPOSURE_PANEL_FILE, config.FACTOR_RETURNS_FILE, config.FACTOR_COVARIANCE_FILE, config.SPECIFIC_RISK_FILE, config.RETURNS_PANEL_FILE]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing factor-model outputs, run `run-all` first: {missing}")

    exposure_panel = pd.read_parquet(config.FACTOR_EXPOSURE_PANEL_FILE)
    factor_returns_daily = pd.read_csv(config.FACTOR_RETURNS_FILE, parse_dates=["Date"]).set_index("Date")
    factor_covariance = pd.read_csv(config.FACTOR_COVARIANCE_FILE, index_col=0)
    specific_risk = pd.read_csv(config.SPECIFIC_RISK_FILE)
    returns_wide = pd.read_parquet(config.RETURNS_PANEL_FILE)

    config.PORTFOLIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.PORTFOLIO_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Expected returns and risk model -----------------------------------
    mu_factor, shrinkage_detail = compute_significance_shrunk_expected_returns(factor_returns_daily)
    exposures = get_latest_exposures(exposure_panel)
    mu_stock = project_to_stock_returns(mu_factor, exposures)
    Sigma = build_stock_covariance(exposures, factor_covariance, specific_risk)

    # mu/exposures may include symbols build_stock_covariance dropped (missing specific risk).
    symbols = list(Sigma.index)
    mu_stock = mu_stock.reindex(symbols)
    exposures = exposures.reindex(symbols)
    logger.info("Optimizable universe: %d symbols", len(symbols))

    save_csv(shrinkage_detail, config.PORTFOLIO_OUTPUT_DIR / "expected_factor_returns.csv", index=True)
    save_csv(mu_stock.to_frame(), config.PORTFOLIO_OUTPUT_DIR / "expected_stock_returns.csv", index=True)
    save_csv(Sigma, config.PORTFOLIO_OUTPUT_DIR / "stock_covariance.csv", index=True)

    # --- Optimization --------------------------------------------------------
    frontier_df, gmv = trace_efficient_frontier(
        mu_stock, Sigma,
        industry_exposures=exposures[config.INDUSTRY_FACTORS],
        max_stock_weight=config.MAX_STOCK_WEIGHT,
        max_industry_weight=config.MAX_INDUSTRY_WEIGHT,
        stock_weight_penalty=config.STOCK_WEIGHT_SOFT_PENALTY_COEF,
    )
    max_sharpe = pick_max_sharpe(frontier_df)
    equal_weight_w = pd.Series(1.0 / len(symbols), index=symbols)

    portfolios = {
        "GMV": gmv.weights,
        "Max-Sharpe": max_sharpe.weights,
        "Equal-Weight": equal_weight_w,
    }

    save_csv(frontier_df.drop(columns=[]), config.PORTFOLIO_OUTPUT_DIR / "efficient_frontier.csv", index=False)
    for name, weights in portfolios.items():
        fname = name.lower().replace("-", "_").replace(" ", "_")
        save_csv(weights.rename("weight").sort_values(ascending=False).to_frame(), config.PORTFOLIO_OUTPUT_DIR / f"weights_{fname}.csv", index=True)

    # --- Analytics -------------------------------------------------------------
    decomp_by_portfolio = {
        name: analytics.risk_decomposition(w, exposures, factor_covariance, specific_risk) for name, w in portfolios.items()
    }
    exposure_by_portfolio = {name: analytics.portfolio_factor_exposure(w, exposures) for name, w in portfolios.items()}
    sector_by_portfolio = {name: analytics.sector_weights(w, exposures) for name, w in portfolios.items()}
    comparison_table = analytics.build_comparison_table(portfolios, exposures, Sigma, mu_stock, factor_covariance, specific_risk)
    save_csv(comparison_table, config.PORTFOLIO_OUTPUT_DIR / "portfolio_comparison.csv", index=False)

    # --- Plots -------------------------------------------------------------
    plot_paths = []
    plot_paths.append(
        pplots.plot_efficient_frontier(
            frontier_df,
            {name: (analytics.risk_decomposition(w, exposures, factor_covariance, specific_risk)["total_variance_daily"] ** 0.5, float(w @ mu_stock.reindex(w.index))) for name, w in portfolios.items()},
            config.PORTFOLIO_PLOTS_DIR,
        )
    )
    for name, weights in portfolios.items():
        if name != "Equal-Weight":
            fname = f"weights_{name.lower().replace('-', '_').replace(' ', '_')}.png"
            plot_paths.append(pplots.plot_portfolio_weights(weights, name, fname, config.PORTFOLIO_PLOTS_DIR))
    plot_paths.append(pplots.plot_factor_tilts(exposure_by_portfolio, config.PORTFOLIO_PLOTS_DIR))
    plot_paths.append(pplots.plot_sector_weights(sector_by_portfolio, config.PORTFOLIO_PLOTS_DIR))
    plot_paths.append(pplots.plot_risk_decomposition(decomp_by_portfolio, config.PORTFOLIO_PLOTS_DIR))
    plot_paths.append(pplots.plot_performance_metrics(comparison_table, config.PORTFOLIO_PLOTS_DIR))
    backtest_returns_by_portfolio = {
        name: analytics.compute_backtest_return_series(w, returns_wide, config.PORTFOLIO_BACKTEST_LOOKBACK_DAYS)
        for name, w in portfolios.items()
    }
    plot_paths.append(
        pplots.plot_cumulative_backtest(backtest_returns_by_portfolio, config.PORTFOLIO_BACKTEST_LOOKBACK_DAYS, config.PORTFOLIO_PLOTS_DIR)
    )
    realized_full = analytics.realized_mean_daily_return(returns_wide[symbols])
    realized_recent = analytics.realized_mean_daily_return(returns_wide[symbols], config.PORTFOLIO_BACKTEST_LOOKBACK_DAYS)
    plot_paths.append(
        pplots.plot_expected_vs_realized_returns(mu_stock, realized_full, realized_recent, gmv.weights, max_sharpe.weights, config.PORTFOLIO_PLOTS_DIR)
    )

    logger.info("Portfolio optimization complete: %d plots written to %s", len(plot_paths), config.PORTFOLIO_PLOTS_DIR)

    return {
        "mu_factor": mu_factor,
        "shrinkage_detail": shrinkage_detail,
        "mu_stock": mu_stock,
        "exposures": exposures,
        "Sigma": Sigma,
        "frontier_df": frontier_df,
        "gmv": gmv,
        "max_sharpe": max_sharpe,
        "portfolios": portfolios,
        "decomp_by_portfolio": decomp_by_portfolio,
        "exposure_by_portfolio": exposure_by_portfolio,
        "sector_by_portfolio": sector_by_portfolio,
        "comparison_table": comparison_table,
        "plot_paths": plot_paths,
        "returns_wide": returns_wide,
        "factor_covariance": factor_covariance,
        "specific_risk": specific_risk,
    }


def run_portfolio_validation(result: dict) -> str:
    """Independent validation of the optimization in `result` (the dict run_portfolio_optimization
    returns) - see portfolio/validation.py for what each check does. Writes
    output/portfolio/portfolio_validation_report.md. This is the model_reviewer's job
    (.claude/agents/model_reviewer.md): it only reads the optimization's output, it never
    adjusts the optimizer to make a check pass.
    """
    portfolios = result["portfolios"]
    Sigma = result["Sigma"]
    factor_covariance = result["factor_covariance"]
    specific_risk = result["specific_risk"]
    frontier_df = result["frontier_df"]
    decomp_by_portfolio = result["decomp_by_portfolio"]
    mu_stock = result["mu_stock"]
    returns_wide = result["returns_wide"]

    tests = [
        pvalidation.test_constraint_satisfaction(portfolios),
        pvalidation.test_covariance_psd(Sigma),
        pvalidation.test_optimizer_convergence(frontier_df, result["gmv"].converged),
        pvalidation.test_variance_decomposition_consistency(portfolios, Sigma, decomp_by_portfolio),
        pvalidation.test_concentration(portfolios),
        pvalidation.test_frontier_monotonicity(frontier_df),
        pvalidation.test_realized_vs_predicted_risk(portfolios, Sigma, result["returns_wide"]),
        pvalidation.test_expected_return_vs_realized_stock_returns(mu_stock, returns_wide),
    ]

    comparison_table_md = _to_markdown_table(result["comparison_table"].round(4))
    report_text = save_portfolio_report(tests, result["plot_paths"], comparison_table_md)
    return report_text


def _to_markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def run_portfolio_backtest(start: str = "2025-01-01") -> dict:
    """Runs the walk-forward, monthly-rebalanced backtest (portfolio/backtest.py) from `start`
    through the latest available data, computes performance metrics, saves everything under
    output/portfolio/backtest/ (mirrored to data/portfolio/backtest/), and generates the
    cumulative-return/drawdown/turnover plots. See cli.py's `backtest-portfolio` command.
    """
    result = run_walk_forward_backtest(start=start)
    return_series = result["return_series"]
    turnover_df = result["turnover"]

    config.PORTFOLIO_BACKTEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.PORTFOLIO_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        benchmark_returns = load_benchmark_returns()
    except (FileNotFoundError, KeyError) as exc:
        logger.warning("Could not load NIFTY50 benchmark returns for the backtest comparison: %s", exc)
        benchmark_returns = None

    any_index = next(iter(return_series.values())).index
    metrics_input = dict(return_series)
    if benchmark_returns is not None:
        metrics_input["NIFTY50"] = benchmark_returns.reindex(any_index).fillna(0.0)

    metrics = analytics.backtest_performance_metrics(metrics_input, turnover_df)
    save_csv(metrics, config.PORTFOLIO_BACKTEST_OUTPUT_DIR / "performance_metrics.csv", index=False)

    returns_df = pd.DataFrame(metrics_input)
    save_csv(returns_df, config.PORTFOLIO_BACKTEST_OUTPUT_DIR / "daily_returns.csv", index=True)
    save_csv(turnover_df, config.PORTFOLIO_BACKTEST_OUTPUT_DIR / "turnover.csv", index=False)

    plot_paths = [
        pplots.plot_backtest_cumulative_returns(return_series, benchmark_returns, config.PORTFOLIO_PLOTS_DIR),
        pplots.plot_backtest_drawdown(return_series, config.PORTFOLIO_PLOTS_DIR),
        pplots.plot_backtest_turnover(turnover_df, config.PORTFOLIO_PLOTS_DIR),
    ]
    if benchmark_returns is not None:
        plot_paths.append(pplots.plot_return_correlation_scatter(return_series, benchmark_returns, config.PORTFOLIO_PLOTS_DIR))

    report_text = None
    benchmark_relative = None
    if benchmark_returns is not None:
        benchmark_relative = analytics.benchmark_relative_metrics(return_series, benchmark_returns)
        save_csv(benchmark_relative, config.PORTFOLIO_BACKTEST_OUTPUT_DIR / "benchmark_relative_metrics.csv", index=False)

        performance_table_md = _to_markdown_table(metrics.round(4))
        benchmark_relative_table_md = _to_markdown_table(benchmark_relative.round(4))
        report_text = save_backtest_report(
            performance_table_md, benchmark_relative_table_md, plot_paths,
            start=start, end=str(any_index.max().date()), n_rebalances=len(result["rebalance_dates"]) - 1,
        )
    else:
        logger.warning("Skipping backtest_report.md - no benchmark data available for comparison")

    logger.info(
        "Backtest complete: %d rebalances from %s, %d plots written to %s",
        len(result["rebalance_dates"]) - 1, start, len(plot_paths), config.PORTFOLIO_PLOTS_DIR,
    )

    return {
        "return_series": return_series,
        "turnover": turnover_df,
        "metrics": metrics,
        "benchmark_relative": benchmark_relative,
        "rebalance_dates": result["rebalance_dates"],
        "weights_history": result["weights_history"],
        "plot_paths": plot_paths,
        "report_text": report_text,
    }


def run_segmented_portfolio_optimization(
    max_stock_weight_by_segment: dict[str, float] | None = None,
    max_industry_weight: float | None = None,
    rest_segment_max_weight: float | None = None,
    rest_segment_weight_penalty: float | None = None,
) -> dict:
    """Mirrors run_portfolio_optimization, but against the segmented (top100/rest) risk model
    (portfolio/segmented_risk_model.py) instead of the pooled full-window one - see
    config.SEGMENT_DESIGN_COLS and CLAUDE.md's "Known limitations". Saves under
    output/portfolio_segments/ (mirrored to data/portfolio_segments/) - never touches
    output/portfolio/, so the pooled-model portfolio result stays available for comparison.
    Requires `run-segments` to have been run first (reads output/segments/<label>/'s outputs).

    `max_stock_weight_by_segment` (e.g. {"top100": 0.10, "rest": 0.20}) tries a looser single-
    stock soft cap on one segment than the other - see segmented_risk_model.build_max_stock_weight.
    Left as None, the single-stock cap is the uniform config.MAX_STOCK_WEIGHT for every symbol
    regardless of segment (unchanged from before this parameter existed).

    `max_industry_weight` overrides config.MAX_INDUSTRY_WEIGHT's hard per-industry cap; pass 1.0
    to remove the industry cap entirely (tried alongside a group-level rest-segment cap instead -
    see below). Left as None, the default config.MAX_INDUSTRY_WEIGHT (a hard bound) still applies.

    `rest_segment_max_weight`/`rest_segment_weight_penalty` add a *soft* cap (discouraged, not
    forbidden - portfolio/optimize.py's group_exposure mechanism) on the total weight held across
    the whole "rest" segment, as a single group, independent of any individual stock's own cap -
    e.g. rest_segment_max_weight=0.20 with the default rest_segment_weight_penalty
    (config.REST_SEGMENT_WEIGHT_SOFT_PENALTY_COEF) discourages more than 20% of the portfolio
    sitting in non-top100 names in aggregate. Left as None (the default), no group-level cap is
    applied at all - unchanged from before this parameter existed.

    Any non-default combination of the three writes to a distinct, suffixed sibling directory
    instead of output/portfolio_segments/ (see _segmented_output_paths), so trying a different
    combination never overwrites another run's result.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    rest_max_stock_weight = max_stock_weight_by_segment["rest"] if max_stock_weight_by_segment else None
    paths = _segmented_output_paths(rest_max_stock_weight, max_industry_weight, rest_segment_max_weight)
    output_dir, plots_dir = paths["output_dir"], paths["plots_dir"]
    effective_max_industry_weight = max_industry_weight if max_industry_weight is not None else config.MAX_INDUSTRY_WEIGHT
    effective_rest_segment_penalty = (
        rest_segment_weight_penalty if rest_segment_weight_penalty is not None else config.REST_SEGMENT_WEIGHT_SOFT_PENALTY_COEF
    )

    segment_paths = {label: config.OutputPaths(config.OUTPUT_DIR / "segments" / label) for label in config.SEGMENT_LABELS}
    required = [config.FACTOR_EXPOSURE_PANEL_FILE, config.RETURNS_PANEL_FILE]
    for sp in segment_paths.values():
        required += [sp.factor_returns_file, sp.specific_risk_file]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing segment-model outputs, run `run-segments` first: {missing}")

    exposure_panel = pd.read_parquet(config.FACTOR_EXPOSURE_PANEL_FILE)
    returns_wide = pd.read_parquet(config.RETURNS_PANEL_FILE)
    factor_returns_by_segment = {
        label: pd.read_csv(sp.factor_returns_file, parse_dates=["Date"]).set_index("Date")
        for label, sp in segment_paths.items()
    }
    specific_risk_by_segment = {label: pd.read_csv(sp.specific_risk_file) for label, sp in segment_paths.items()}

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Expected returns and risk model -----------------------------------
    jcols = joint_design_cols()
    exposures_by_segment = {label: get_latest_segment_exposures(exposure_panel, label) for label in config.SEGMENT_LABELS}
    exposures_joint = build_joint_exposures(exposures_by_segment)
    factor_covariance_joint = build_joint_factor_covariance(factor_returns_by_segment)
    symbols_by_segment = {label: list(exposures_by_segment[label].index) for label in config.SEGMENT_LABELS}
    specific_risk_joint = build_joint_specific_risk(specific_risk_by_segment, symbols_by_segment)
    Sigma = build_segmented_stock_covariance(exposures_joint, factor_covariance_joint, specific_risk_joint)
    max_stock_weight = build_max_stock_weight(symbols_by_segment, max_stock_weight_by_segment)
    rest_group_exposure = build_group_exposure(symbols_by_segment, "rest") if rest_segment_max_weight is not None else None

    mu_factor_parts, shrinkage_parts = [], []
    for label in config.SEGMENT_LABELS:
        mu_factor_label, detail_label = compute_significance_shrunk_expected_returns(
            factor_returns_by_segment[label], design_cols=config.SEGMENT_DESIGN_COLS
        )
        mu_factor_label.index = [f"{c}__{label}" for c in mu_factor_label.index]
        detail_label.index = [f"{c}__{label}" for c in detail_label.index]
        mu_factor_parts.append(mu_factor_label)
        shrinkage_parts.append(detail_label)
    mu_factor = pd.concat(mu_factor_parts)
    shrinkage_detail = pd.concat(shrinkage_parts)
    mu_stock = project_to_stock_returns(mu_factor, exposures_joint, design_cols=jcols)

    # mu/exposures may include symbols build_segmented_stock_covariance dropped (missing specific risk).
    symbols = list(Sigma.index)
    mu_stock = mu_stock.reindex(symbols)
    exposures_joint = exposures_joint.reindex(symbols)
    # Raw (unsuffixed) exposures - needed for sector_weights and the optimizer's industry-cap
    # constraint, which both expect config.INDUSTRY_FACTORS' plain column names, not the
    # segment-suffixed joint design.
    exposures_raw = pd.concat([exposures_by_segment[label] for label in config.SEGMENT_LABELS], axis=0).reindex(symbols)
    logger.info("Optimizable universe (segmented model): %d symbols", len(symbols))

    save_csv(shrinkage_detail, output_dir / "expected_factor_returns.csv", index=True)
    save_csv(mu_stock.to_frame(), output_dir / "expected_stock_returns.csv", index=True)
    save_csv(Sigma, output_dir / "stock_covariance.csv", index=True)

    # --- Optimization --------------------------------------------------------
    frontier_df, gmv = trace_efficient_frontier(
        mu_stock, Sigma,
        industry_exposures=exposures_raw[config.INDUSTRY_FACTORS],
        max_stock_weight=max_stock_weight,
        max_industry_weight=effective_max_industry_weight,
        stock_weight_penalty=config.STOCK_WEIGHT_SOFT_PENALTY_COEF,
        group_exposure=rest_group_exposure,
        group_max_weight=rest_segment_max_weight if rest_segment_max_weight is not None else 1.0,
        group_weight_penalty=effective_rest_segment_penalty if rest_segment_max_weight is not None else None,
    )
    max_sharpe = pick_max_sharpe(frontier_df)
    equal_weight_w = pd.Series(1.0 / len(symbols), index=symbols)

    portfolios = {
        "GMV": gmv.weights,
        "Max-Sharpe": max_sharpe.weights,
        "Equal-Weight": equal_weight_w,
    }

    save_csv(frontier_df.drop(columns=[]), output_dir / "efficient_frontier.csv", index=False)
    for name, weights in portfolios.items():
        fname = name.lower().replace("-", "_").replace(" ", "_")
        save_csv(weights.rename("weight").sort_values(ascending=False).to_frame(), output_dir / f"weights_{fname}.csv", index=True)

    # --- Analytics -------------------------------------------------------------
    decomp_by_portfolio = {
        name: analytics.risk_decomposition(w, exposures_joint, factor_covariance_joint, specific_risk_joint, design_cols=jcols)
        for name, w in portfolios.items()
    }
    exposure_by_portfolio = {
        name: analytics.portfolio_factor_exposure(w, exposures_joint, design_cols=jcols) for name, w in portfolios.items()
    }
    sector_by_portfolio = {name: analytics.sector_weights(w, exposures_raw) for name, w in portfolios.items()}
    comparison_table = analytics.build_comparison_table(
        portfolios, exposures_joint, Sigma, mu_stock, factor_covariance_joint, specific_risk_joint, design_cols=jcols
    )
    save_csv(comparison_table, output_dir / "portfolio_comparison.csv", index=False)

    # --- Plots -------------------------------------------------------------
    plot_paths = []
    plot_paths.append(
        pplots.plot_efficient_frontier(
            frontier_df,
            {
                name: (
                    analytics.risk_decomposition(w, exposures_joint, factor_covariance_joint, specific_risk_joint, design_cols=jcols)["total_variance_daily"] ** 0.5,
                    float(w @ mu_stock.reindex(w.index)),
                )
                for name, w in portfolios.items()
            },
            plots_dir,
        )
    )
    for name, weights in portfolios.items():
        if name != "Equal-Weight":
            fname = f"weights_{name.lower().replace('-', '_').replace(' ', '_')}.png"
            plot_paths.append(pplots.plot_portfolio_weights(weights, name, fname, plots_dir))
    plot_paths.append(pplots.plot_factor_tilts(exposure_by_portfolio, plots_dir, design_cols=jcols))
    plot_paths.append(pplots.plot_sector_weights(sector_by_portfolio, plots_dir))
    plot_paths.append(pplots.plot_risk_decomposition(decomp_by_portfolio, plots_dir))
    plot_paths.append(pplots.plot_performance_metrics(comparison_table, plots_dir))
    backtest_returns_by_portfolio = {
        name: analytics.compute_backtest_return_series(w, returns_wide, config.PORTFOLIO_BACKTEST_LOOKBACK_DAYS)
        for name, w in portfolios.items()
    }
    plot_paths.append(
        pplots.plot_cumulative_backtest(backtest_returns_by_portfolio, config.PORTFOLIO_BACKTEST_LOOKBACK_DAYS, plots_dir)
    )
    realized_full = analytics.realized_mean_daily_return(returns_wide[symbols])
    realized_recent = analytics.realized_mean_daily_return(returns_wide[symbols], config.PORTFOLIO_BACKTEST_LOOKBACK_DAYS)
    plot_paths.append(
        pplots.plot_expected_vs_realized_returns(mu_stock, realized_full, realized_recent, gmv.weights, max_sharpe.weights, plots_dir)
    )

    logger.info("Segmented portfolio optimization complete: %d plots written to %s", len(plot_paths), plots_dir)

    return {
        "mu_factor": mu_factor,
        "shrinkage_detail": shrinkage_detail,
        "mu_stock": mu_stock,
        "exposures": exposures_joint,
        "design_cols": jcols,
        "Sigma": Sigma,
        "frontier_df": frontier_df,
        "gmv": gmv,
        "max_sharpe": max_sharpe,
        "portfolios": portfolios,
        "decomp_by_portfolio": decomp_by_portfolio,
        "exposure_by_portfolio": exposure_by_portfolio,
        "sector_by_portfolio": sector_by_portfolio,
        "comparison_table": comparison_table,
        "plot_paths": plot_paths,
        "returns_wide": returns_wide,
        "factor_covariance": factor_covariance_joint,
        "specific_risk": specific_risk_joint,
        "max_stock_weight_by_segment": max_stock_weight_by_segment,
        "max_industry_weight": effective_max_industry_weight,
        "rest_segment_max_weight": rest_segment_max_weight,
        "rest_segment_weight_penalty": effective_rest_segment_penalty if rest_segment_max_weight is not None else None,
        "output_paths": paths,
    }


def run_segmented_portfolio_validation(result: dict) -> str:
    """Same independent validation suite as run_portfolio_validation, run against the segmented
    model's result dict instead. Writes <output_dir>/portfolio_validation_report.md, where
    <output_dir> is whatever output/portfolio_segments[_rest<N>]/ directory `result` came from
    (result["output_paths"], set by run_segmented_portfolio_optimization) - never
    output/portfolio/portfolio_validation_report.md.
    """
    portfolios = result["portfolios"]
    Sigma = result["Sigma"]
    factor_covariance = result["factor_covariance"]
    specific_risk = result["specific_risk"]
    frontier_df = result["frontier_df"]
    decomp_by_portfolio = result["decomp_by_portfolio"]
    mu_stock = result["mu_stock"]
    returns_wide = result["returns_wide"]

    tests = [
        pvalidation.test_constraint_satisfaction(portfolios),
        pvalidation.test_covariance_psd(Sigma),
        pvalidation.test_optimizer_convergence(frontier_df, result["gmv"].converged),
        pvalidation.test_variance_decomposition_consistency(portfolios, Sigma, decomp_by_portfolio),
        pvalidation.test_concentration(portfolios),
        pvalidation.test_frontier_monotonicity(frontier_df),
        pvalidation.test_realized_vs_predicted_risk(portfolios, Sigma, result["returns_wide"]),
        pvalidation.test_expected_return_vs_realized_stock_returns(mu_stock, returns_wide),
    ]

    comparison_table_md = _to_markdown_table(result["comparison_table"].round(4))
    report_text = save_portfolio_report(
        tests, result["plot_paths"], comparison_table_md, report_file=result["output_paths"]["validation_report_file"]
    )
    return report_text


def run_segmented_portfolio_backtest(
    start: str = "2025-01-01",
    max_stock_weight_by_segment: dict[str, float] | None = None,
    max_industry_weight: float | None = None,
    rest_segment_max_weight: float | None = None,
    rest_segment_weight_penalty: float | None = None,
) -> dict:
    """Same walk-forward, monthly-rebalanced backtest methodology as run_portfolio_backtest, but
    against the segmented risk model (portfolio/backtest.py's run_segmented_walk_forward_backtest).
    Saves under output/portfolio_segments/backtest/ (mirrored to data/portfolio_segments/backtest/)
    by default, or a distinct sibling directory if any of the cap parameters are given (see
    run_segmented_portfolio_optimization / _segmented_output_paths) - never
    output/portfolio/backtest/, so the pooled-model backtest stays available alongside it.
    """
    rest_max_stock_weight = max_stock_weight_by_segment["rest"] if max_stock_weight_by_segment else None
    paths = _segmented_output_paths(rest_max_stock_weight, max_industry_weight, rest_segment_max_weight)
    plots_dir, backtest_output_dir = paths["plots_dir"], paths["backtest_output_dir"]

    result = run_segmented_walk_forward_backtest(
        start=start, max_stock_weight_by_segment=max_stock_weight_by_segment,
        max_industry_weight=max_industry_weight,
        rest_segment_max_weight=rest_segment_max_weight,
        rest_segment_weight_penalty=rest_segment_weight_penalty,
    )
    return_series = result["return_series"]
    turnover_df = result["turnover"]

    backtest_output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        benchmark_returns = load_benchmark_returns()
    except (FileNotFoundError, KeyError) as exc:
        logger.warning("Could not load NIFTY50 benchmark returns for the backtest comparison: %s", exc)
        benchmark_returns = None

    any_index = next(iter(return_series.values())).index
    metrics_input = dict(return_series)
    if benchmark_returns is not None:
        metrics_input["NIFTY50"] = benchmark_returns.reindex(any_index).fillna(0.0)

    metrics = analytics.backtest_performance_metrics(metrics_input, turnover_df)
    save_csv(metrics, backtest_output_dir / "performance_metrics.csv", index=False)

    returns_df = pd.DataFrame(metrics_input)
    save_csv(returns_df, backtest_output_dir / "daily_returns.csv", index=True)
    save_csv(turnover_df, backtest_output_dir / "turnover.csv", index=False)

    plot_paths = [
        pplots.plot_backtest_cumulative_returns(return_series, benchmark_returns, plots_dir),
        pplots.plot_backtest_drawdown(return_series, plots_dir),
        pplots.plot_backtest_turnover(turnover_df, plots_dir),
    ]
    if benchmark_returns is not None:
        plot_paths.append(pplots.plot_return_correlation_scatter(return_series, benchmark_returns, plots_dir))

    report_text = None
    benchmark_relative = None
    if benchmark_returns is not None:
        benchmark_relative = analytics.benchmark_relative_metrics(return_series, benchmark_returns)
        save_csv(benchmark_relative, backtest_output_dir / "benchmark_relative_metrics.csv", index=False)

        performance_table_md = _to_markdown_table(metrics.round(4))
        benchmark_relative_table_md = _to_markdown_table(benchmark_relative.round(4))
        report_text = save_backtest_report(
            performance_table_md, benchmark_relative_table_md, plot_paths,
            start=start, end=str(any_index.max().date()), n_rebalances=len(result["rebalance_dates"]) - 1,
            report_file=paths["backtest_report_file"],
        )
    else:
        logger.warning("Skipping backtest_report.md - no benchmark data available for comparison")

    logger.info(
        "Segmented backtest complete: %d rebalances from %s, %d plots written to %s",
        len(result["rebalance_dates"]) - 1, start, len(plot_paths), plots_dir,
    )

    return {
        "return_series": return_series,
        "turnover": turnover_df,
        "metrics": metrics,
        "benchmark_relative": benchmark_relative,
        "rebalance_dates": result["rebalance_dates"],
        "weights_history": result["weights_history"],
        "plot_paths": plot_paths,
        "report_text": report_text,
    }
