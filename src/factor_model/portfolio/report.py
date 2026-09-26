"""Assembles the portfolio optimization validation report - same executive-summary-first
structure as validation/report.py. Written by the model_reviewer agent; only reads the
optimization's output, never adjusts it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from factor_model import config

logger = logging.getLogger(__name__)


def render_report(tests: list[dict], plot_paths: list[str] | None = None, comparison_table_md: str | None = None) -> str:
    lines = ["# Portfolio Optimization Validation Report", "", f"Generated: {datetime.now(timezone.utc).isoformat()}", ""]

    lines += ["## Executive summary", "", "| Test | Verdict | Statistic |", "|---|---|---|"]
    for t in tests:
        lines.append(f"| {t['name']} | **{t['verdict']}** | {t['statistic']} |")
    lines.append("")

    n_fail = sum(1 for t in tests if t["verdict"] == "FAIL")
    lines.append(f"**{n_fail} FAIL** out of {len(tests)} checks.")
    lines.append("")

    if comparison_table_md:
        lines.append("## Portfolio comparison")
        lines.append("")
        lines.append(comparison_table_md)
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


def save_report(
    tests: list[dict],
    plot_paths: list[str] | None = None,
    comparison_table_md: str | None = None,
    report_file: Path = config.PORTFOLIO_VALIDATION_REPORT_FILE,
) -> str:
    report_text = render_report(tests, plot_paths, comparison_table_md)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(report_text)
    logger.info("Wrote portfolio validation report to %s", report_file)
    return report_text
