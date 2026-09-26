"""Assembles the walk-forward backtest's NIFTY50-benchmark comparison report - same
executive-summary-first structure as validation/report.py and portfolio/report.py.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from factor_model import config

logger = logging.getLogger(__name__)


def render_backtest_report(
    performance_table_md: str,
    benchmark_relative_table_md: str,
    plot_paths: list[str],
    start: str,
    end: str,
    n_rebalances: int,
) -> str:
    lines = [
        "# Portfolio Backtest Report: Comparison vs. NIFTY50",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Methodology",
        "",
        f"Walk-forward, monthly-rebalanced backtest from {start} to {end} ({n_rebalances} monthly "
        "holding periods). At each month-end, GMV and Max-Sharpe are re-optimized using only "
        "information available as of that date (an expanding window: point-in-time factor "
        "exposures, EWMA factor covariance/specific risk, and significance-shrunk, "
        "EWMA-recency-weighted expected factor returns - see portfolio/backtest.py and "
        "portfolio/expected_returns.py), subject to the same 10% single-stock / 20% single-"
        "industry position caps as the standalone optimization. Equal-Weight splits evenly "
        "across that month's optimizable universe. Weights are held fixed (not rebalanced) "
        "over the following month, then the cycle repeats. NIFTY50 is the raw buy-and-hold "
        "benchmark over the same trading days, for context - it is not itself rebalanced or "
        "risk-managed by this model.",
        "",
        "## Performance vs. NIFTY50",
        "",
        performance_table_md,
        "",
        "## Benchmark-relative statistics",
        "",
        benchmark_relative_table_md,
        "",
        "*Beta/alpha from a CAPM-style decomposition against NIFTY50 daily returns; alpha is "
        "annualized. Tracking error and information ratio are computed on the daily "
        "(portfolio - benchmark) excess-return series. Up/down capture: portfolio's average "
        "daily return on benchmark-up (or benchmark-down) days, divided by the benchmark's own "
        "average return on those same days.*",
        "",
    ]

    if plot_paths:
        lines.append("## Plots")
        lines.append("")
        for p in plot_paths:
            lines.append(f"- `plots/{Path(p).name}`")
        lines.append("")

    return "\n".join(lines)


def save_backtest_report(
    performance_table_md: str,
    benchmark_relative_table_md: str,
    plot_paths: list[str],
    start: str,
    end: str,
    n_rebalances: int,
    report_file: Path = config.PORTFOLIO_BACKTEST_REPORT_FILE,
) -> str:
    report_text = render_backtest_report(performance_table_md, benchmark_relative_table_md, plot_paths, start, end, n_rebalances)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(report_text)
    logger.info("Wrote backtest report to %s", report_file)
    return report_text
