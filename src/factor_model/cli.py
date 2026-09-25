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


if __name__ == "__main__":
    cli()
