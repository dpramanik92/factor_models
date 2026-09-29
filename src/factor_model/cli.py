"""CLI entrypoints. Run via `factor-model <command>` (after `pip install -e .`)
or `python -m factor_model.cli <command>`. See CLAUDE.md for usage.
"""

from __future__ import annotations

import logging

import click

from factor_model import config, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@click.group()
def cli() -> None:
    pass


@cli.command("build-universe")
def build_universe_cmd() -> None:
    config.ensure_output_dirs()
    pipeline.step_build_universe()


@cli.command("build-fundamentals")
def build_fundamentals_cmd() -> None:
    config.ensure_output_dirs()
    mapping, _ = pipeline.step_build_universe()
    pipeline.step_build_fundamentals(mapping)


@cli.command("build-factors")
def build_factors_cmd() -> None:
    config.ensure_output_dirs()
    mapping, universe = pipeline.step_build_universe()
    annual_ratios = pipeline.step_build_fundamentals(mapping)
    pipeline.step_build_factors(universe, annual_ratios)


@cli.command("run-regression")
def run_regression_cmd() -> None:
    config.ensure_output_dirs()
    mapping, universe = pipeline.step_build_universe()
    annual_ratios = pipeline.step_build_fundamentals(mapping)
    exposure_panel, returns_wide = pipeline.step_build_factors(universe, annual_ratios)
    _, summary = pipeline.step_run_regression(exposure_panel, returns_wide)
    click.echo(summary.to_string(index=False))


@cli.command("run-risk")
def run_risk_cmd() -> None:
    config.ensure_output_dirs()
    mapping, universe = pipeline.step_build_universe()
    annual_ratios = pipeline.step_build_fundamentals(mapping)
    exposure_panel, returns_wide = pipeline.step_build_factors(universe, annual_ratios)
    results, _ = pipeline.step_run_regression(exposure_panel, returns_wide)
    pipeline.step_run_risk(results)


@cli.command("validate")
def validate_cmd() -> None:
    config.ensure_output_dirs()
    result = pipeline.run_all()
    click.echo(result["report_text"])


@cli.command("refresh-prices")
@click.option("--force", is_flag=True, help="Refresh even if local data already runs through today.")
def refresh_prices_cmd(force: bool) -> None:
    from factor_model.io.yahoo_refresh import refresh_prices

    refresh_prices(force=force)


@cli.command("run-all")
def run_all_cmd() -> None:
    result = pipeline.run_all()
    click.echo(result["summary"].to_string(index=False))
    click.echo("")
    click.echo(result["report_text"])


@cli.command("run-periods")
def run_periods_cmd() -> None:
    """Runs each named period in config.MODEL_PERIODS as a separate scoped model, saved under
    output/periods/<label>/ - never overwrites the full-window output/ result. Requires
    `run-all` to have been run at least once already (reuses its model_data/ artifacts).
    """
    period_results = pipeline.run_all_periods()
    for label, result in period_results.items():
        click.echo(f"=== {label} ({result['output_dir']}) ===")
        click.echo(result["summary"].to_string(index=False))
        click.echo("")


@cli.command("run-segments")
def run_segments_cmd() -> None:
    """Runs both cap segments (top100-by-market-cap, rest) as separate scoped regressions, saved
    under output/segments/<label>/ - never overwrites the full-window output/ result. Requires
    `run-all` to have been run at least once already (reuses its model_data/ artifacts). See
    config.SEGMENT_DESIGN_COLS / pipeline.run_segment / CLAUDE.md for why this replaced an
    earlier interaction-term attempt at the same problem.
    """
    segment_results = pipeline.run_all_segments()
    for label, result in segment_results.items():
        click.echo(f"=== {label} ({result['output_dir']}) ===")
        click.echo(result["summary"].to_string(index=False))
        click.echo("")


_macro_timing_option = click.option(
    "--macro-timing", is_flag=True, default=False,
    help="Add the two macro-momentum expected-return tilts from portfolio/macro_timing.py "
    "(Oil-sector vs. oil-price momentum, IT-exporter vs. USD/INR momentum - see CLAUDE.md's "
    "'Alpha research' section). Off by default; fetches Brent crude and USD/INR via yfinance.",
)


@cli.command("optimize-portfolio")
@_macro_timing_option
def optimize_portfolio_cmd(macro_timing: bool) -> None:
    """Long-only mean-variance optimization against the full-window factor model - see
    portfolio/pipeline.py. Requires `run-all` to have been run first.
    """
    from factor_model.portfolio.pipeline import run_portfolio_optimization

    result = run_portfolio_optimization(apply_macro_timing=macro_timing)
    click.echo("Expected factor returns (Newey-West significance-shrunk):")
    click.echo(result["shrinkage_detail"].round(6).to_string())
    click.echo("")
    click.echo(result["comparison_table"].to_string(index=False))


@cli.command("validate-portfolio")
@_macro_timing_option
def validate_portfolio_cmd(macro_timing: bool) -> None:
    """Runs the portfolio optimization and its independent validation suite (model_reviewer's
    checks - see portfolio/validation.py), writing output/portfolio/portfolio_validation_report.md.
    """
    from factor_model.portfolio.pipeline import run_portfolio_optimization, run_portfolio_validation

    result = run_portfolio_optimization(apply_macro_timing=macro_timing)
    click.echo("Expected factor returns (Newey-West significance-shrunk):")
    click.echo(result["shrinkage_detail"].round(6).to_string())
    click.echo("")
    report_text = run_portfolio_validation(result)
    click.echo(report_text)


@cli.command("backtest-portfolio")
@click.option("--start", default="2025-01-01", help="Backtest start date (holding periods begin here).")
@_macro_timing_option
def backtest_portfolio_cmd(start: str, macro_timing: bool) -> None:
    """Walk-forward, monthly-rebalanced backtest of GMV/Max-Sharpe/Equal-Weight from `start`
    through the latest available data - see portfolio/backtest.py. Requires `run-all` to have
    been run first.
    """
    from factor_model.portfolio.pipeline import run_portfolio_backtest

    result = run_portfolio_backtest(start=start, apply_macro_timing=macro_timing)
    if result["report_text"]:
        click.echo(result["report_text"])
    else:
        click.echo(result["metrics"].round(4).to_string(index=False))


_long_short_option = click.option(
    "--long-short", is_flag=True, default=False,
    help="Allow short positions for GMV/Max-Sharpe (portfolio/optimize.py's "
    "minimize_variance_long_short / trace_efficient_frontier_long_short) instead of long-only, "
    "subject to a gross-leverage cap (--max-leverage). Equal-Weight stays long-only. Ignores "
    "--rest-max-stock-weight/--rest-segment-max-weight (the long-short optimizer only supports a "
    "uniform scalar stock cap and the hard industry cap).",
)
_max_leverage_option = click.option(
    "--max-leverage", type=float, default=None,
    help="Gross exposure cap sum(|w|) for --long-short, e.g. 1.4 allows up to 20% of capital "
    "short (funding 20% extra long, a '120/40'-style book). Default: config.MAX_LEVERAGE (1.2, "
    "i.e. up to 10% short). No effect without --long-short.",
)

_rest_max_stock_weight_option = click.option(
    "--rest-max-stock-weight", type=float, default=None,
    help="Looser single-stock soft cap for the 'rest' (non-top100) segment only, e.g. 0.20 for 20% "
    "(top100 keeps config.MAX_STOCK_WEIGHT/10%). Default: uniform 10% cap for every symbol.",
)
_no_industry_cap_option = click.option(
    "--no-industry-cap", is_flag=True, default=False,
    help="Remove the hard per-industry cap (config.MAX_INDUSTRY_WEIGHT, normally 20%) entirely.",
)
_rest_segment_max_weight_option = click.option(
    "--rest-segment-max-weight", type=float, default=None,
    help="Soft cap (discouraged, not forbidden) on the TOTAL weight held across the whole 'rest' "
    "segment as a group, e.g. 0.20 for 20% - independent of any individual stock's own cap. "
    "Default: no group-level cap.",
)
_rest_segment_weight_penalty_option = click.option(
    "--rest-segment-weight-penalty", type=float, default=None,
    help="Penalty coefficient for --rest-segment-max-weight (default: "
    "config.REST_SEGMENT_WEIGHT_SOFT_PENALTY_COEF). Ignored unless --rest-segment-max-weight is given.",
)


def _segment_caps_from_options(rest_max_stock_weight, no_industry_cap, rest_segment_max_weight):
    from factor_model import config

    caps = {"top100": config.MAX_STOCK_WEIGHT, "rest": rest_max_stock_weight} if rest_max_stock_weight is not None else None
    max_industry_weight = 1.0 if no_industry_cap else None
    return caps, max_industry_weight


@cli.command("optimize-portfolio-segments")
@_rest_max_stock_weight_option
@_no_industry_cap_option
@_rest_segment_max_weight_option
@_rest_segment_weight_penalty_option
@_macro_timing_option
@_long_short_option
@_max_leverage_option
def optimize_portfolio_segments_cmd(
    rest_max_stock_weight: float | None, no_industry_cap: bool, rest_segment_max_weight: float | None, rest_segment_weight_penalty: float | None,
    macro_timing: bool, long_short: bool, max_leverage: float | None,
) -> None:
    """Long-only mean-variance optimization against the segmented (top100/rest) risk model
    instead of the pooled full-window one - see portfolio/segmented_risk_model.py and CLAUDE.md's
    "Known limitations". Saves under output/portfolio_segments/ (or a distinct suffixed sibling
    directory if any cap option is given), alongside (never overwriting) the pooled-model result
    in output/portfolio/. Requires `run-segments` to have been run first.
    """
    from factor_model.portfolio.pipeline import run_segmented_portfolio_optimization

    caps, max_industry_weight = _segment_caps_from_options(rest_max_stock_weight, no_industry_cap, rest_segment_max_weight)
    result = run_segmented_portfolio_optimization(
        max_stock_weight_by_segment=caps, max_industry_weight=max_industry_weight,
        rest_segment_max_weight=rest_segment_max_weight, rest_segment_weight_penalty=rest_segment_weight_penalty,
        apply_macro_timing=macro_timing, long_short=long_short, max_leverage=max_leverage,
    )
    click.echo("Expected factor returns (Newey-West significance-shrunk, per segment):")
    click.echo(result["shrinkage_detail"].round(6).to_string())
    click.echo("")
    click.echo(result["comparison_table"].to_string(index=False))


@cli.command("validate-portfolio-segments")
@_rest_max_stock_weight_option
@_no_industry_cap_option
@_rest_segment_max_weight_option
@_rest_segment_weight_penalty_option
@_macro_timing_option
@_long_short_option
@_max_leverage_option
def validate_portfolio_segments_cmd(
    rest_max_stock_weight: float | None, no_industry_cap: bool, rest_segment_max_weight: float | None, rest_segment_weight_penalty: float | None,
    macro_timing: bool, long_short: bool, max_leverage: float | None,
) -> None:
    """Runs the segmented portfolio optimization and its independent validation suite, writing
    <output_dir>/portfolio_validation_report.md (output/portfolio_segments/ by default, or a
    distinct suffixed sibling directory if any cap option is given).
    """
    from factor_model.portfolio.pipeline import run_segmented_portfolio_optimization, run_segmented_portfolio_validation

    caps, max_industry_weight = _segment_caps_from_options(rest_max_stock_weight, no_industry_cap, rest_segment_max_weight)
    result = run_segmented_portfolio_optimization(
        max_stock_weight_by_segment=caps, max_industry_weight=max_industry_weight,
        rest_segment_max_weight=rest_segment_max_weight, rest_segment_weight_penalty=rest_segment_weight_penalty,
        apply_macro_timing=macro_timing, long_short=long_short, max_leverage=max_leverage,
    )
    click.echo("Expected factor returns (Newey-West significance-shrunk, per segment):")
    click.echo(result["shrinkage_detail"].round(6).to_string())
    click.echo("")
    report_text = run_segmented_portfolio_validation(result)
    click.echo(report_text)


@cli.command("backtest-portfolio-segments")
@click.option("--start", default="2025-01-01", help="Backtest start date (holding periods begin here).")
@_rest_max_stock_weight_option
@_no_industry_cap_option
@_rest_segment_max_weight_option
@_rest_segment_weight_penalty_option
@_macro_timing_option
@_long_short_option
@_max_leverage_option
def backtest_portfolio_segments_cmd(
    start: str, rest_max_stock_weight: float | None, no_industry_cap: bool, rest_segment_max_weight: float | None, rest_segment_weight_penalty: float | None,
    macro_timing: bool, long_short: bool, max_leverage: float | None,
) -> None:
    """Walk-forward, monthly-rebalanced backtest of GMV/Max-Sharpe/Equal-Weight against the
    segmented risk model - see portfolio/backtest.py's run_segmented_walk_forward_backtest.
    Saves under output/portfolio_segments/backtest/ by default (or a distinct suffixed sibling
    directory if any cap option is given), alongside (never overwriting) the pooled-model
    backtest in output/portfolio/backtest/. Requires `run-segments` to have been run first.
    """
    from factor_model.portfolio.pipeline import run_segmented_portfolio_backtest

    caps, max_industry_weight = _segment_caps_from_options(rest_max_stock_weight, no_industry_cap, rest_segment_max_weight)
    result = run_segmented_portfolio_backtest(
        start=start, max_stock_weight_by_segment=caps, max_industry_weight=max_industry_weight,
        rest_segment_max_weight=rest_segment_max_weight, rest_segment_weight_penalty=rest_segment_weight_penalty,
        apply_macro_timing=macro_timing, long_short=long_short, max_leverage=max_leverage,
    )
    if result["report_text"]:
        click.echo(result["report_text"])
    else:
        click.echo(result["metrics"].round(4).to_string(index=False))


if __name__ == "__main__":
    cli()
