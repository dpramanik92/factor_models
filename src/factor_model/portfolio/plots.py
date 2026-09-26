"""Portfolio optimization diagnostic plots, saved to output/portfolio/plots/. Same conventions
as validation/plots.py - plain matplotlib, internal engineering diagnostics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factor_model import config

logger = logging.getLogger(__name__)

_PORTFOLIO_COLORS = {
    "GMV": "#1f77b4",
    "Max-Sharpe": "#d62728",
    "Beta-Neutral": "#2ca02c",
    "Long-Short GMV": "#9467bd",
    "Long-Short Max-Sharpe": "#8c564b",
    "Equal-Weight": "#7f7f7f",
}


def plot_efficient_frontier(
    frontier_df: pd.DataFrame, portfolios: dict[str, tuple[float, float]], plots_dir: Path
) -> str:
    """portfolios: name -> (volatility, return), both daily, annualized for the plot axes."""
    ann = config.TRADING_DAYS_PER_YEAR ** 0.5
    ann_ret = config.TRADING_DAYS_PER_YEAR

    fig, ax = plt.subplots(figsize=(9, 7))
    converged = frontier_df[frontier_df["converged"]]
    ax.plot(converged["volatility"] * ann * 100, converged["realized_return"] * ann_ret * 100, color="#1f77b4", linewidth=1.8, label="Efficient frontier")

    for name, (vol, ret) in portfolios.items():
        ax.scatter([vol * ann * 100], [ret * ann_ret * 100], s=90, color=_PORTFOLIO_COLORS.get(name, "black"), label=name, zorder=5, edgecolors="black")

    ax.set_xlabel("Annualized volatility (%)")
    ax.set_ylabel("Annualized expected excess return (%)")
    ax.set_title("Long-only efficient frontier")
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "efficient_frontier.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_portfolio_weights(weights: pd.Series, title: str, filename: str, plots_dir: Path, top_n: int = 20, by_abs: bool = False) -> str:
    """by_abs=True ranks by |weight| and plots the signed value - needed for a long-short
    portfolio, where sorting by plain descending value would hide large short positions off the
    bottom of a truncated top_n instead of showing them alongside the largest longs.
    """
    if by_abs:
        top = weights.reindex(weights.abs().sort_values(ascending=False).head(top_n).index)
        colors = ["#d62728" if v < 0 else "#1f77b4" for v in top.to_numpy()[::-1]]
    else:
        top = weights.sort_values(ascending=False).head(top_n)
        colors = "#1f77b4"
    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(top))))
    ax.barh(top.index[::-1], top.to_numpy()[::-1] * 100, color=colors)
    if by_abs:
        ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Weight (%)")
    ax.set_title(f"{title} - top {len(top)} holdings" + (" by |weight| (short in red)" if by_abs else ""))
    fig.tight_layout()
    path = plots_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_factor_tilts(
    exposures_by_portfolio: dict[str, pd.Series], plots_dir: Path, design_cols: list[str] | None = None, filename: str = "factor_tilts.png"
) -> str:
    if design_cols is None:
        from factor_model.regression.cross_sectional import DESIGN_COLS

        design_cols = DESIGN_COLS

    fig, ax = plt.subplots(figsize=(10, max(8, len(design_cols) * 0.22)))
    n_portfolios = len(exposures_by_portfolio)
    bar_width = 0.8 / n_portfolios
    y = np.arange(len(design_cols))

    for i, (name, exposure) in enumerate(exposures_by_portfolio.items()):
        vals = exposure.reindex(design_cols).to_numpy()
        ax.barh(y + i * bar_width, vals, height=bar_width, label=name, color=_PORTFOLIO_COLORS.get(name))

    ax.set_yticks(y + bar_width * (n_portfolios - 1) / 2)
    ax.set_yticklabels(design_cols)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Net portfolio exposure")
    ax.set_title("Portfolio factor tilts vs. equal-weight benchmark")
    ax.legend()
    fig.tight_layout()
    path = plots_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_sector_weights(sector_weights_by_portfolio: dict[str, pd.Series], plots_dir: Path) -> str:
    df = pd.DataFrame(sector_weights_by_portfolio).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 7))
    df.plot(kind="barh", ax=ax, color=[_PORTFOLIO_COLORS.get(c, None) for c in df.columns])
    ax.set_xlabel("Portfolio weight")
    ax.set_title("Sector allocation by portfolio")
    fig.tight_layout()
    path = plots_dir / "sector_weights.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_performance_metrics(comparison_table: pd.DataFrame, plots_dir: Path) -> str:
    """Grouped bar chart of annualized return, volatility, and Sharpe per portfolio, read
    straight from the comparison table printed to chat/CSV - no recomputation.
    """
    names = comparison_table["Portfolio"].tolist()
    metrics = [("Return_annualized", "Return (annualized, %)", 100.0), ("Vol_annualized", "Volatility (annualized, %)", 100.0), ("Sharpe", "Sharpe ratio", 1.0)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (col, label, scale) in zip(axes, metrics):
        vals = comparison_table[col].to_numpy() * scale
        colors = [_PORTFOLIO_COLORS.get(n, "#1f77b4") for n in names]
        ax.bar(names, vals, color=colors)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=20)
        ax.axhline(0, color="black", linewidth=0.6)
    fig.suptitle("Portfolio performance metrics")
    fig.tight_layout()
    path = plots_dir / "performance_metrics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_cumulative_backtest(return_series_by_portfolio: dict[str, pd.Series], lookback_days: int, plots_dir: Path) -> str:
    """Cumulative growth of 1 unit over the trailing `lookback_days`, static weights (not
    rebalanced) - the same series the realized-vs-predicted-risk validation check uses
    (portfolio/analytics.py's compute_backtest_return_series), just cumulated for display.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, returns in return_series_by_portfolio.items():
        cumulative = (1.0 + returns).cumprod()
        ax.plot(cumulative.index, cumulative.to_numpy(), label=name, color=_PORTFOLIO_COLORS.get(name, "#1f77b4"), linewidth=1.6)
    ax.set_ylabel("Growth of 1 unit")
    ax.set_title(f"Static-weight backtest, trailing {lookback_days} trading days (not rebalanced)")
    ax.legend()
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--")
    fig.tight_layout()
    path = plots_dir / "cumulative_backtest.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_expected_vs_realized_returns(
    mu_stock: pd.Series, realized_full: pd.Series, realized_recent: pd.Series, gmv_weights: pd.Series, max_sharpe_weights: pd.Series, plots_dir: Path
) -> str:
    """Scatter of each stock's projected annualized expected return (x) against its realized
    annualized return (y) - full-sample (left) and trailing-window (right) - with GMV/Max-Sharpe
    holdings picked out, so a systematic disconnect (or a portfolio's holdings clustering away
    from the y=x line) is visible at a glance rather than buried in a correlation number.
    """
    ann = config.TRADING_DAYS_PER_YEAR
    symbols = mu_stock.index
    x = mu_stock.to_numpy() * ann * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=False)
    for ax, realized, label in [(axes[0], realized_full, "Full-sample"), (axes[1], realized_recent, f"Trailing {config.PORTFOLIO_BACKTEST_LOOKBACK_DAYS}d")]:
        y = realized.reindex(symbols).to_numpy() * ann * 100
        ax.scatter(x, y, s=25, color="#c7c7c7", zorder=2, label="All optimizable stocks")
        in_gmv = [s in gmv_weights[gmv_weights > 1e-6].index for s in symbols]
        in_ms = [s in max_sharpe_weights[max_sharpe_weights > 1e-6].index for s in symbols]
        ax.scatter(x[in_gmv], y[in_gmv], s=45, color="#1f77b4", zorder=3, label="GMV holdings")
        ax.scatter(x[in_ms], y[in_ms], s=45, color="#d62728", zorder=4, marker="D", label="Max-Sharpe holdings")
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, linestyle="--", label="y = x")
        ax.axhline(0, color="black", linewidth=0.4)
        ax.axvline(0, color="black", linewidth=0.4)
        ax.set_xlabel("Projected expected return (annualized, %)")
        ax.set_ylabel(f"{label} realized return (annualized, %)")
        ax.set_title(label)
        ax.legend(fontsize=8)
    fig.suptitle("Projected expected return vs. realized stock returns")
    fig.tight_layout()
    path = plots_dir / "expected_vs_realized_returns.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_backtest_cumulative_returns(
    return_series_by_portfolio: dict[str, pd.Series], benchmark_returns: pd.Series | None, plots_dir: Path
) -> str:
    """Growth of 1 unit over the walk-forward backtest, one line per portfolio, plus the NIFTY50
    benchmark for context if given.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, r in return_series_by_portfolio.items():
        cumulative = (1.0 + r).cumprod()
        ax.plot(cumulative.index, cumulative.to_numpy(), label=name, color=_PORTFOLIO_COLORS.get(name, "#1f77b4"), linewidth=1.6)
    if benchmark_returns is not None:
        any_index = next(iter(return_series_by_portfolio.values())).index
        aligned = benchmark_returns.reindex(any_index).fillna(0.0)
        cumulative = (1.0 + aligned).cumprod()
        ax.plot(cumulative.index, cumulative.to_numpy(), label="NIFTY50", color="black", linewidth=1.2, linestyle="--")
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    ax.set_ylabel("Growth of 1 unit")
    ax.set_title("Walk-forward backtest: monthly-rebalanced cumulative return")
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "backtest_cumulative_returns.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_backtest_drawdown(return_series_by_portfolio: dict[str, pd.Series], plots_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, r in return_series_by_portfolio.items():
        cumulative = (1.0 + r).cumprod()
        drawdown = cumulative / cumulative.cummax() - 1.0
        ax.plot(drawdown.index, drawdown.to_numpy() * 100, label=name, color=_PORTFOLIO_COLORS.get(name, "#1f77b4"), linewidth=1.4)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Walk-forward backtest: drawdown")
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "backtest_drawdown.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_return_correlation_scatter(
    return_series_by_portfolio: dict[str, pd.Series], benchmark_returns: pd.Series, plots_dir: Path
) -> str:
    """One scatter subplot per portfolio: daily NIFTY50 return (x) vs. that portfolio's daily
    backtest return (y), with a best-fit line and the Pearson correlation annotated - the same
    correlation reported in benchmark_relative_metrics.csv, just shown visually.
    """
    names = list(return_series_by_portfolio.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5.5 * len(names), 5), sharex=True, sharey=True)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        r = return_series_by_portfolio[name]
        bench = benchmark_returns.reindex(r.index).fillna(0.0)
        corr = float(r.corr(bench))

        ax.scatter(bench * 100, r * 100, s=14, alpha=0.5, color=_PORTFOLIO_COLORS.get(name, "#1f77b4"))
        slope, intercept = np.polyfit(bench, r, 1)
        x_line = np.array([bench.min(), bench.max()])
        ax.plot(x_line * 100, (slope * x_line + intercept) * 100, color="black", linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.4)
        ax.axvline(0, color="black", linewidth=0.4)
        ax.set_xlabel("NIFTY50 daily return (%)")
        ax.set_title(f"{name} (corr={corr:.3f})")

    axes[0].set_ylabel("Portfolio daily return (%)")
    fig.suptitle("Portfolio vs. NIFTY50 daily return correlation")
    fig.tight_layout()
    path = plots_dir / "return_correlation_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_backtest_turnover(turnover_df: pd.DataFrame, plots_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, group in turnover_df.groupby("Portfolio"):
        g = group.sort_values("Date")
        ax.plot(g["Date"], g["Turnover"] * 100, marker="o", markersize=4, label=name, color=_PORTFOLIO_COLORS.get(name, "#1f77b4"))
    ax.set_ylabel("One-way turnover (%)")
    ax.set_title("Monthly rebalance turnover")
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "backtest_turnover.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_risk_decomposition(decomp_by_portfolio: dict[str, dict], plots_dir: Path) -> str:
    names = list(decomp_by_portfolio.keys())
    factor_pct = [decomp_by_portfolio[n]["factor_pct"] * 100 for n in names]
    specific_pct = [decomp_by_portfolio[n]["specific_pct"] * 100 for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, factor_pct, label="Factor risk", color="#1f77b4")
    ax.bar(names, specific_pct, bottom=factor_pct, label="Specific risk", color="#d3d3d3")
    ax.set_ylabel("% of total portfolio variance")
    ax.set_title("Risk decomposition: factor vs. specific")
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "risk_decomposition.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
