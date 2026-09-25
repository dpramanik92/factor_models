"""Assemble all statistical validation tests into output/validation_report.md.

Executive-summary pass/fail table first, then one section per test - see
.claude/skills/design/SKILL.md. Written by the model_reviewer agent
(.claude/agents/model_reviewer.md); this module only reads model/regression
output, it never modifies it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from factor_model import config
from factor_model.factors.industry import load_industry_exposure
from factor_model.regression.cross_sectional import CrossSectionalResults
from factor_model.validation import plots as vp
from factor_model.validation import statistical_tests as vt

logger = logging.getLogger(__name__)


def run_all_tests(
    universe: list[str],
    annual_ratios: pd.DataFrame,
    exposure_panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    regression_results: CrossSectionalResults,
    summary: pd.DataFrame,
) -> list[dict]:
    industry_by_symbol = load_industry_exposure(universe)

    tests = [
        vt.test_r2_distribution(regression_results.r2_daily),
        vt.test_newey_west_significance(summary),
        vt.test_vif(exposure_panel),
        vt.test_residual_normality(regression_results.residuals_long),
        vt.test_residual_alpha_significance(regression_results.residuals_long),
        vt.test_residual_heteroskedasticity(exposure_panel, regression_results.residuals_long),
        vt.test_residual_autocorrelation(regression_results.residuals_long),
        vt.test_industry_weights_sum_to_one(industry_by_symbol),
        vt.test_factor_stationarity(regression_results.factor_returns),
        vt.test_lookahead_bias(annual_ratios, exposure_panel),
        vt.test_size_quintile_spread(exposure_panel, returns_wide),
        vt.test_regression_coverage(
            regression_results.coverage_start_date,
            regression_results.skipped_low_n_dates,
            regression_results.skipped_degenerate_dates,
        ),
    ]
    return tests


def generate_plots(
    exposure_panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    regression_results: CrossSectionalResults,
    summary: pd.DataFrame,
    factor_covariance: pd.DataFrame,
) -> list[str]:
    vif_by_factor = vt.compute_mean_vif(exposure_panel)
    return vp.generate_all_plots(
        factor_returns=regression_results.factor_returns,
        summary=summary,
        r2_daily=regression_results.r2_daily,
        factor_covariance=factor_covariance,
        residuals_long=regression_results.residuals_long,
        exposure_panel=exposure_panel,
        returns_wide=returns_wide,
        vif_by_factor=vif_by_factor,
    )


def render_report(tests: list[dict], plot_paths: list[str] | None = None) -> str:
    lines = ["# Factor Model Validation Report", "", f"Generated: {datetime.now(timezone.utc).isoformat()}", ""]

    lines += ["## Executive summary", "", "| Test | Verdict | Statistic |", "|---|---|---|"]
    for t in tests:
        lines.append(f"| {t['name']} | **{t['verdict']}** | {t['statistic']} |")
    lines.append("")

    n_fail = sum(1 for t in tests if t["verdict"] == "FAIL")
    n_warn = sum(1 for t in tests if t["verdict"] == "WARN")
    lines.append(f"**{n_fail} FAIL, {n_warn} WARN** out of {len(tests)} checks.")
    lines.append("")

    lines.append("## Detail")
    lines.append("")
    for t in tests:
        lines.append(f"### {t['name']}")
        lines.append("")
        lines.append(f"- **Verdict**: {t['verdict']}")
        lines.append(f"- **Statistic**: {t['statistic']}")
        lines.append(f"- **Threshold**: {t['threshold']}")
        if t["detail"]:
            lines.append(f"- **Detail**: {t['detail']}")
        lines.append("")

    if plot_paths:
        lines.append("## Plots")
        lines.append("")
        for p in plot_paths:
            lines.append(f"- `plots/{Path(p).name}`")
        lines.append("")

    return "\n".join(lines)


def save_report(tests: list[dict], plot_paths: list[str] | None = None) -> str:
    report_text = render_report(tests, plot_paths)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.VALIDATION_REPORT_FILE.write_text(report_text)
    logger.info("Wrote validation report to %s", config.VALIDATION_REPORT_FILE)
    return report_text
