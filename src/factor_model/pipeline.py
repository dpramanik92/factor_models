"""Orchestrates the full build: universe -> fundamentals -> factor panel ->
regression -> risk -> validation. See cli.py for the command-line entrypoints
and CLAUDE.md for the pipeline overview.
"""

from __future__ import annotations

import logging

import pandas as pd

from factor_model import config
from factor_model.factors.fundamentals_ratios import build_fundamentals_annual_long, compute_annual_ratios
from factor_model.factors.panel import build_exposure_panel, build_returns_panel
from factor_model.io.fundamentals import parse_all_fundamentals
from factor_model.mapping.ticker_mapping import build_universe_mapping, get_locked_universe, save_universe_mapping
from factor_model.regression.cross_sectional import run_cross_sectional_regression
from factor_model.regression.results import save_regression_results
from factor_model.risk.ewma_covariance import compute_ewma_factor_covariance, save_factor_covariance
from factor_model.risk.specific_risk import compute_ewma_specific_risk, save_specific_risk
from factor_model.validation.report import generate_plots, run_all_tests, save_report

logger = logging.getLogger(__name__)


def step_build_universe() -> tuple[pd.DataFrame, list[str]]:
    mapping = build_universe_mapping()
    save_universe_mapping(mapping)
    universe = get_locked_universe(mapping)
    logger.info("Locked universe: %d symbols", len(universe))
    return mapping, universe


def step_build_fundamentals(mapping: pd.DataFrame) -> pd.DataFrame:
    parsed = parse_all_fundamentals()
    annual_long = build_fundamentals_annual_long(mapping, parsed)
    annual_ratios = compute_annual_ratios(annual_long)
    config.MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    annual_ratios.to_parquet(config.FUNDAMENTALS_ANNUAL_FILE, index=False)
    logger.info("Wrote fundamentals annual table: %s (%s)", config.FUNDAMENTALS_ANNUAL_FILE, annual_ratios.shape)
    return annual_ratios


def step_build_factors(universe: list[str], annual_ratios: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns_wide = build_returns_panel(universe)
    business_day_index = returns_wide.index
    exposure_panel = build_exposure_panel(universe, annual_ratios, business_day_index)
    return exposure_panel, returns_wide


def step_run_regression(
    exposure_panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    paths: config.OutputPaths = config.DEFAULT_OUTPUT_PATHS,
    design_cols: list[str] | None = None,
):
    kwargs = {"design_cols": design_cols} if design_cols is not None else {}
    results = run_cross_sectional_regression(exposure_panel, returns_wide, **kwargs)
    summary = save_regression_results(results, paths)
    return results, summary


def step_run_risk(results, paths: config.OutputPaths = config.DEFAULT_OUTPUT_PATHS) -> tuple[pd.DataFrame, pd.DataFrame]:
    cov = compute_ewma_factor_covariance(results.factor_returns)
    save_factor_covariance(cov, paths)
    specific_risk = compute_ewma_specific_risk(results.residuals_long)
    save_specific_risk(specific_risk, paths)
    return cov, specific_risk


def step_validate(
    universe,
    annual_ratios,
    exposure_panel,
    returns_wide,
    results,
    summary,
    cov,
    paths: config.OutputPaths = config.DEFAULT_OUTPUT_PATHS,
    is_segment: bool = False,
) -> tuple[str, list[str]]:
    tests = run_all_tests(universe, annual_ratios, exposure_panel, returns_wide, results, summary, is_segment=is_segment)
    plot_paths = generate_plots(exposure_panel, returns_wide, results, summary, cov, paths, is_segment=is_segment)
    report_text = save_report(tests, plot_paths, paths)
    return report_text, plot_paths


def run_all() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config.ensure_output_dirs()

    mapping, universe = step_build_universe()
    annual_ratios = step_build_fundamentals(mapping)
    exposure_panel, returns_wide = step_build_factors(universe, annual_ratios)
    results, summary = step_run_regression(exposure_panel, returns_wide)
    cov, specific_risk = step_run_risk(results)
    report_text, plot_paths = step_validate(universe, annual_ratios, exposure_panel, returns_wide, results, summary, cov)

    return {
        "universe": universe,
        "annual_ratios": annual_ratios,
        "exposure_panel": exposure_panel,
        "returns_wide": returns_wide,
        "results": results,
        "summary": summary,
        "cov": cov,
        "specific_risk": specific_risk,
        "report_text": report_text,
        "plot_paths": plot_paths,
    }


def run_period(
    universe: list[str],
    annual_ratios: pd.DataFrame,
    exposure_panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    label: str,
    start_date: str,
    end_date: str,
) -> dict:
    """Run the regression/risk/validation stages restricted to [start_date, end_date), saving
    under output/periods/<label>/ (mirrored to data/periods/<label>/) - never touches the
    full-window result at the output/ top level. See config.MODEL_PERIODS.
    """
    paths = config.OutputPaths(config.OUTPUT_DIR / "periods" / label)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    period_exposure = exposure_panel[
        (exposure_panel["Date"] >= start_date) & (exposure_panel["Date"] < end_date)
    ].copy()
    period_returns = returns_wide.loc[(returns_wide.index >= start_date) & (returns_wide.index < end_date)].copy()

    logger.info(
        "Running period %s: %s to %s (%d exposure dates)",
        label,
        start_date,
        end_date,
        period_exposure["Date"].nunique(),
    )

    results, summary = step_run_regression(period_exposure, period_returns, paths)
    cov, specific_risk = step_run_risk(results, paths)
    report_text, plot_paths = step_validate(
        universe, annual_ratios, period_exposure, period_returns, results, summary, cov, paths
    )

    return {
        "label": label,
        "results": results,
        "summary": summary,
        "cov": cov,
        "specific_risk": specific_risk,
        "report_text": report_text,
        "plot_paths": plot_paths,
        "output_dir": str(paths.output_dir),
    }


def run_all_periods(periods: dict[str, tuple[str, str]] | None = None) -> dict[str, dict]:
    """Runs each named period in config.MODEL_PERIODS (or a custom periods dict) as a separate
    scoped model, reusing the universe/fundamentals/factor-panel artifacts already built by the
    most recent run_all() (read directly from model_data/, not rebuilt) - see run_period. Raises
    if those artifacts don't exist yet; run `run-all` first.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config.ensure_output_dirs()
    periods = periods or config.MODEL_PERIODS

    if not config.FACTOR_EXPOSURE_PANEL_FILE.exists() or not config.UNIVERSE_FILE.exists():
        raise RuntimeError(
            "model_data/ artifacts not found - run `run-all` (or `build-factors`) at least once "
            "before running period-scoped models, so they share the same universe/exposure panel "
            "as the full-window result."
        )

    universe_mapping = pd.read_csv(config.UNIVERSE_FILE)
    universe = sorted(universe_mapping.loc[universe_mapping["included"], "matched_symbol"].unique().tolist())
    annual_ratios = pd.read_parquet(config.FUNDAMENTALS_ANNUAL_FILE)
    exposure_panel = pd.read_parquet(config.FACTOR_EXPOSURE_PANEL_FILE)
    returns_wide = pd.read_parquet(config.RETURNS_PANEL_FILE)

    period_results = {}
    for label, (start_date, end_date) in periods.items():
        period_results[label] = run_period(
            universe, annual_ratios, exposure_panel, returns_wide, label, start_date, end_date
        )
    return period_results


def run_segment(
    universe: list[str],
    annual_ratios: pd.DataFrame,
    exposure_panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    label: str,
    top100_value: float,
) -> dict:
    """Run the regression/risk/validation stages restricted to one cap segment
    (top100_flag == top100_value), using config.SEGMENT_DESIGN_COLS (drops top100_flag and any
    interaction with it, which would be constant/absent within a single segment and make the
    design matrix singular) - saving under output/segments/<label>/ (mirrored to
    data/segments/<label>/), never touching the pooled full-universe result at the output/ top
    level. Replaces an earlier style_factor x top100_flag interaction-term attempt at fixing the
    large-cap/small-cap structural break, which made the segment-level residual bias worse, not
    better - see CLAUDE.md.
    """
    paths = config.OutputPaths(config.OUTPUT_DIR / "segments" / label)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    segment_exposure = exposure_panel[exposure_panel["top100_flag"] == top100_value].copy()

    logger.info(
        "Running segment %s (top100_flag=%s): %d exposure rows across %d symbols",
        label, top100_value, len(segment_exposure), segment_exposure["Symbol"].nunique(),
    )

    results, summary = step_run_regression(segment_exposure, returns_wide, paths, design_cols=config.SEGMENT_DESIGN_COLS)
    cov, specific_risk = step_run_risk(results, paths)
    report_text, plot_paths = step_validate(
        universe, annual_ratios, segment_exposure, returns_wide, results, summary, cov, paths, is_segment=True
    )

    return {
        "label": label,
        "results": results,
        "summary": summary,
        "cov": cov,
        "specific_risk": specific_risk,
        "report_text": report_text,
        "plot_paths": plot_paths,
        "output_dir": str(paths.output_dir),
    }


def run_all_segments() -> dict[str, dict]:
    """Runs both cap segments (top100, rest) as separate scoped models, reusing the
    universe/fundamentals/factor-panel artifacts already built by the most recent run_all()
    (read directly from model_data/, not rebuilt) - see run_segment. Raises if those artifacts
    don't exist yet; run `run-all` first.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config.ensure_output_dirs()

    if not config.FACTOR_EXPOSURE_PANEL_FILE.exists() or not config.UNIVERSE_FILE.exists():
        raise RuntimeError(
            "model_data/ artifacts not found - run `run-all` (or `build-factors`) at least once "
            "before running segment-scoped models, so they share the same universe/exposure panel "
            "as the full-window result."
        )

    universe_mapping = pd.read_csv(config.UNIVERSE_FILE)
    universe = sorted(universe_mapping.loc[universe_mapping["included"], "matched_symbol"].unique().tolist())
    annual_ratios = pd.read_parquet(config.FUNDAMENTALS_ANNUAL_FILE)
    exposure_panel = pd.read_parquet(config.FACTOR_EXPOSURE_PANEL_FILE)
    returns_wide = pd.read_parquet(config.RETURNS_PANEL_FILE)

    segment_results = {}
    for label, top100_value in config.SEGMENT_TOP100_FLAG_VALUE.items():
        segment_results[label] = run_segment(
            universe, annual_ratios, exposure_panel, returns_wide, label, top100_value
        )
    return segment_results
