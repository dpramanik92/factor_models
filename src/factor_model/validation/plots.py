"""Diagnostic plots saved to output/plots/. Plain matplotlib - these are internal engineering
diagnostics, not chat-facing visualizations (see .claude/skills/design/SKILL.md). Same color per
factor across every plot; industry vs style visually distinguished by linestyle.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factor_model import config

logger = logging.getLogger(__name__)

_INDUSTRY_STYLE = {"linestyle": "-", "alpha": 0.85}
_STYLE_STYLE = {"linestyle": "-", "alpha": 0.95, "linewidth": 1.8}


def _factor_colors() -> dict[str, str]:
    cmap = plt.get_cmap("tab20")
    return {f: cmap(i % 20) for i, f in enumerate(config.ALL_FACTORS)}


def plot_cumulative_factor_returns(factor_returns: pd.DataFrame) -> list[str]:
    """Two panels: cumulative industry factor returns, cumulative style factor returns."""
    colors = _factor_colors()
    cum = (1.0 + factor_returns).cumprod() - 1.0
    paths = []

    for group_name, cols, kw in [
        ("industry", config.INDUSTRY_FACTORS, _INDUSTRY_STYLE),
        ("style", config.STYLE_FACTORS, _STYLE_STYLE),
    ]:
        fig, ax = plt.subplots(figsize=(11, 6))
        for col in cols:
            if col in cum.columns:
                ax.plot(cum.index, cum[col] * 100, label=col, color=colors[col], **kw)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(f"Cumulative {group_name} factor returns")
        ax.set_ylabel("Cumulative return (%)")
        ax.set_xlabel("Date")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        fig.autofmt_xdate()
        path = config.PLOTS_DIR / f"cumulative_{group_name}_factor_returns.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_factor_significance(summary: pd.DataFrame) -> str:
    df = summary.sort_values("NW_t")
    colors = ["#d62728" if abs(t) > 2 else "#7f7f7f" for t in df["NW_t"]]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(df["Factor"], df["NW_t"], color=colors)
    ax.axvline(2, color="black", linestyle="--", linewidth=0.7)
    ax.axvline(-2, color="black", linestyle="--", linewidth=0.7)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Newey-West t-stat")
    ax.set_title("Factor significance (red = |NW t| > 2)")
    fig.tight_layout()
    path = config.PLOTS_DIR / "factor_significance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_r2_timeseries(r2_daily: pd.DataFrame) -> str:
    df = r2_daily.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    rolling = df.set_index("Date")["R2"].rolling(21, min_periods=5).mean()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["Date"], df["R2"], color="#1f77b4", alpha=0.25, linewidth=0.8, label="Daily R2")
    ax.plot(rolling.index, rolling.values, color="#1f77b4", linewidth=1.8, label="21-day rolling mean")
    ax.set_title("Cross-sectional regression R2 over time")
    ax.set_ylabel("R2")
    ax.set_xlabel("Date")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    path = config.PLOTS_DIR / "r2_daily.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_factor_correlation_heatmap(factor_covariance: pd.DataFrame) -> str:
    cov = factor_covariance.to_numpy()
    std = np.sqrt(np.diag(cov))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(std, std)
    corr = np.nan_to_num(corr)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(factor_covariance.columns)))
    ax.set_xticklabels(factor_covariance.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(factor_covariance.index)))
    ax.set_yticklabels(factor_covariance.index, fontsize=7)
    ax.set_title("EWMA factor correlation matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = config.PLOTS_DIR / "factor_correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_vif(summary_or_vif: dict[str, float]) -> str:
    factors = list(summary_or_vif.keys())
    values = list(summary_or_vif.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728" if v > 10 else "#2ca02c" for v in values]
    ax.barh(factors, values, color=colors)
    ax.axvline(10, color="black", linestyle="--", linewidth=0.7, label="VIF = 10 threshold")
    ax.set_xlabel("VIF")
    ax.set_title("Style factor multicollinearity (VIF)")
    ax.legend()
    fig.tight_layout()
    path = config.PLOTS_DIR / "vif.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_residual_histogram(residuals_long: pd.DataFrame) -> str:
    resid = residuals_long["Residual"].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(resid, bins=200, density=True, color="#1f77b4", alpha=0.7, label="Residuals")
    x = np.linspace(resid.min(), resid.max(), 200)
    normal_pdf = (1 / (resid.std() * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - resid.mean()) / resid.std()) ** 2)
    ax.plot(x, normal_pdf, color="#d62728", linewidth=1.5, label="Normal fit (same mean/std)")
    ax.set_title("Pooled residual distribution vs. normal")
    ax.set_xlabel("Residual (excess return)")
    ax.legend()
    fig.tight_layout()
    path = config.PLOTS_DIR / "residual_histogram.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_daily_alpha(residuals_long: pd.DataFrame) -> str:
    daily_alpha = residuals_long.groupby("Date")["Residual"].mean().sort_index()
    cum_alpha = (1.0 + daily_alpha).cumprod() - 1.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(daily_alpha.index, daily_alpha.values * 100, color="#7f7f7f", linewidth=0.7)
    ax1.axhline(0, color="black", linewidth=0.6)
    ax1.set_title("Daily equal-weighted residual ('alpha') - unexplained by the 20 factors")
    ax1.set_ylabel("Daily alpha (%)")

    ax2.plot(cum_alpha.index, cum_alpha.values * 100, color="#1f77b4", linewidth=1.5)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_ylabel("Cumulative alpha (%)")
    ax2.set_xlabel("Date")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = config.PLOTS_DIR / "residual_alpha.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_size_quintile_spread(exposure_panel: pd.DataFrame, returns_wide: pd.DataFrame) -> str:
    returns_wide = returns_wide.copy()
    returns_wide.index.name = "Date"
    returns_long = returns_wide.reset_index().melt(id_vars="Date", var_name="Symbol", value_name="return")

    merged = exposure_panel[["Date", "Symbol", "size_quintile"]].merge(
        returns_long, on=["Date", "Symbol"], how="inner"
    ).dropna(subset=["size_quintile", "return"])

    means = merged.groupby("size_quintile")["return"].mean() * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([f"Q{int(q)}" for q in means.index], means.values, color="#1f77b4")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("Average daily return by size quintile (Q1=smallest, Q5=largest)")
    ax.set_ylabel("Mean daily return (%)")
    fig.tight_layout()
    path = config.PLOTS_DIR / "size_quintile_spread.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def generate_all_plots(
    factor_returns: pd.DataFrame,
    summary: pd.DataFrame,
    r2_daily: pd.DataFrame,
    factor_covariance: pd.DataFrame,
    residuals_long: pd.DataFrame,
    exposure_panel: pd.DataFrame,
    returns_wide: pd.DataFrame,
    vif_by_factor: dict[str, float] | None = None,
) -> list[str]:
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    paths += plot_cumulative_factor_returns(factor_returns)
    paths.append(plot_factor_significance(summary))
    paths.append(plot_r2_timeseries(r2_daily))
    paths.append(plot_factor_correlation_heatmap(factor_covariance))
    if vif_by_factor:
        paths.append(plot_vif(vif_by_factor))
    paths.append(plot_residual_histogram(residuals_long))
    paths.append(plot_daily_alpha(residuals_long))
    paths.append(plot_size_quintile_spread(exposure_panel, returns_wide))
    logger.info("Wrote %d plots to %s", len(paths), config.PLOTS_DIR)
    return paths
