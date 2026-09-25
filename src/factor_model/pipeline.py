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


def step_run_regression(exposure_panel: pd.DataFrame, returns_wide: pd.DataFrame):
    results = run_cross_sectional_regression(exposure_panel, returns_wide)
    summary = save_regression_results(results)
    return results, summary


def step_run_risk(results) -> tuple[pd.DataFrame, pd.DataFrame]:
    cov = compute_ewma_factor_covariance(results.factor_returns)
    save_factor_covariance(cov)
    specific_risk = compute_ewma_specific_risk(results.residuals_long)
    save_specific_risk(specific_risk)
    return cov, specific_risk


def step_validate(universe, annual_ratios, exposure_panel, returns_wide, results, summary, cov) -> tuple[str, list[str]]:
    tests = run_all_tests(universe, annual_ratios, exposure_panel, returns_wide, results, summary)
    plot_paths = generate_plots(exposure_panel, returns_wide, results, summary, cov)
    report_text = save_report(tests, plot_paths)
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
