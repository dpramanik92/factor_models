"""Portfolio analytics: risk decomposition, factor tilts, concentration, and cross-portfolio
comparison tables. Everything here reads an OptimizationResult / weights Series and the same
exposures/Sigma/mu used to build it - it doesn't re-optimize anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_model import config
from factor_model.regression.cross_sectional import DESIGN_COLS


def portfolio_factor_exposure(weights: pd.Series, exposures: pd.DataFrame, design_cols: list[str] = DESIGN_COLS) -> pd.Series:
    """The optimized portfolio's net exposure to each design factor (w'X). `design_cols` defaults
    to the pooled model's DESIGN_COLS; the segmented risk model passes its joint (segment-
    suffixed) column list instead - see portfolio/segmented_risk_model.py.
    """
    aligned = exposures.reindex(weights.index)
    return (aligned[design_cols].T @ weights).reindex(design_cols)


def risk_decomposition(
    weights: pd.Series,
    exposures: pd.DataFrame,
    factor_covariance: pd.DataFrame,
    specific_risk: pd.DataFrame,
    design_cols: list[str] = DESIGN_COLS,
) -> dict:
    """Splits total portfolio variance into factor-risk and specific-risk contributions.
    `design_cols` defaults to DESIGN_COLS; the segmented risk model passes its joint column list.
    """
    symbols = list(weights.index)
    w = weights.to_numpy()
    X = exposures.loc[symbols, design_cols].to_numpy()
    F = factor_covariance.loc[design_cols, design_cols].to_numpy()

    port_factor_exposure = w @ X
    factor_variance = float(port_factor_exposure @ F @ port_factor_exposure)

    spec = specific_risk.set_index("Symbol")["specific_risk_annualized"].reindex(symbols)
    daily_specific_var = (spec.to_numpy() ** 2) / config.TRADING_DAYS_PER_YEAR
    specific_variance = float(np.sum((w ** 2) * daily_specific_var))

    total_variance = factor_variance + specific_variance
    return {
        "total_variance_daily": total_variance,
        "factor_variance_daily": factor_variance,
        "specific_variance_daily": specific_variance,
        "factor_pct": factor_variance / total_variance if total_variance > 0 else np.nan,
        "specific_pct": specific_variance / total_variance if total_variance > 0 else np.nan,
        "total_vol_annualized": (total_variance * config.TRADING_DAYS_PER_YEAR) ** 0.5,
    }


def concentration_stats(weights: pd.Series) -> dict:
    """Herfindahl-Hirschman index and the equivalent 'effective number of holdings' (1/HHI).
    Sign-agnostic throughout (uses |weight|), so a long-short portfolio's short positions count
    toward concentration exactly like longs - for a long-only portfolio (all weights >= 0) this
    is identical to the plain-weight version.
    """
    abs_w = weights.abs()
    nonzero = abs_w[abs_w > 1e-9]
    hhi = float((nonzero ** 2).sum())
    top_sorted = abs_w.sort_values(ascending=False)
    return {
        "n_holdings": int((abs_w > 1e-6).sum()),
        "hhi": hhi,
        "effective_n_holdings": 1.0 / hhi if hhi > 0 else np.nan,
        "top1_weight": float(top_sorted.iloc[0]) if len(top_sorted) else 0.0,
        "top5_weight": float(top_sorted.head(5).sum()),
        "top10_weight": float(top_sorted.head(10).sum()),
    }


def sector_weights(weights: pd.Series, exposures: pd.DataFrame) -> pd.Series:
    """Portfolio weight aggregated by industry (using each stock's industry weights, which may
    be fractional across up to 5 industries - see factors/industry.py).
    """
    aligned = exposures.reindex(weights.index)[config.INDUSTRY_FACTORS]
    return (aligned.T @ weights).sort_values(ascending=False)


def compute_backtest_return_series(
    weights: pd.Series, returns_wide: pd.DataFrame, lookback_days: int = config.PORTFOLIO_BACKTEST_LOOKBACK_DAYS
) -> pd.Series:
    """Static-weight (not rebalanced) daily portfolio return series over the trailing
    `lookback_days` of actual historical stock returns - the same methodology used by both the
    realized-vs-predicted-risk validation check and the cumulative-performance plot, kept in one
    place so the two can't silently diverge.
    """
    recent = returns_wide.tail(lookback_days)
    symbols = [s for s in weights.index if s in recent.columns]
    w = weights.loc[symbols]
    w = w / w.sum()
    return (recent[symbols].fillna(0.0) @ w.to_numpy()).rename("return")


def realized_mean_daily_return(returns_wide: pd.DataFrame, lookback_days: int | None = None) -> pd.Series:
    """Per-stock realized mean daily return - full-sample if lookback_days is None, else the
    trailing lookback_days window. Used to sanity-check the optimizer's projected mu_stock
    against what each stock actually returned, independent of the factor machinery that
    produced mu_stock.
    """
    window = returns_wide if lookback_days is None else returns_wide.tail(lookback_days)
    return window.mean()


def backtest_performance_metrics(return_series: dict[str, pd.Series], turnover: pd.DataFrame) -> pd.DataFrame:
    """One row per portfolio in a walk-forward backtest's daily return series
    (portfolio/backtest.py): geometric annualized return/vol/Sharpe, max drawdown, Calmar ratio,
    and average one-way monthly turnover.
    """
    rows = []
    for name, r in return_series.items():
        cumulative = (1.0 + r).cumprod()
        n_days = len(r)
        years = n_days / config.TRADING_DAYS_PER_YEAR
        total_return = float(cumulative.iloc[-1] - 1.0)
        ann_return = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else np.nan
        ann_vol = float(r.std(ddof=1) * (config.TRADING_DAYS_PER_YEAR ** 0.5))
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

        drawdown = cumulative / cumulative.cummax() - 1.0
        max_dd = float(drawdown.min())
        calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan

        avg_turnover = (
            float(turnover.loc[turnover["Portfolio"] == name, "Turnover"].mean()) if not turnover.empty else np.nan
        )

        rows.append(
            {
                "Portfolio": name,
                "Total_return": total_return,
                "Annualized_return": ann_return,
                "Annualized_vol": ann_vol,
                "Sharpe": sharpe,
                "Max_drawdown": max_dd,
                "Calmar": calmar,
                "Avg_monthly_turnover": avg_turnover,
                "N_days": n_days,
            }
        )
    return pd.DataFrame(rows)


def benchmark_relative_metrics(return_series: dict[str, pd.Series], benchmark_returns: pd.Series) -> pd.DataFrame:
    """One row per portfolio in a walk-forward backtest's daily return series, benchmarked
    against NIFTY50 over the exact same trading days: beta (cov/var against the benchmark),
    Jensen's alpha (annualized, CAPM-style: portfolio_ann_return - beta * benchmark_ann_return),
    annualized tracking error and information ratio on the daily excess-return series, return
    correlation, and up/down capture (portfolio's average daily return on benchmark-up/down days
    divided by the benchmark's own average return on those same days - a simple, standard daily
    approximation, not a compounded-period capture ratio).
    """
    rows = []
    for name, r in return_series.items():
        bench = benchmark_returns.reindex(r.index).fillna(0.0)
        n = len(r)

        var_bench = float(bench.var(ddof=1))
        beta = float(np.cov(r, bench, ddof=1)[0, 1] / var_bench) if var_bench > 0 else np.nan

        ann_port_return = float((1.0 + r).prod() ** (config.TRADING_DAYS_PER_YEAR / n) - 1.0)
        ann_bench_return = float((1.0 + bench).prod() ** (config.TRADING_DAYS_PER_YEAR / n) - 1.0)
        alpha_annualized = ann_port_return - beta * ann_bench_return if not np.isnan(beta) else np.nan

        excess = r - bench
        tracking_error = float(excess.std(ddof=1) * (config.TRADING_DAYS_PER_YEAR ** 0.5))
        information_ratio = (
            float(excess.mean() * config.TRADING_DAYS_PER_YEAR / tracking_error) if tracking_error > 0 else np.nan
        )

        up = bench > 0
        down = bench < 0
        up_capture = float(r[up].mean() / bench[up].mean()) if up.any() and bench[up].mean() != 0 else np.nan
        down_capture = float(r[down].mean() / bench[down].mean()) if down.any() and bench[down].mean() != 0 else np.nan

        rows.append(
            {
                "Portfolio": name,
                "Beta": beta,
                "Alpha_annualized": alpha_annualized,
                "Tracking_error_annualized": tracking_error,
                "Information_ratio": information_ratio,
                "Correlation": float(r.corr(bench)),
                "Up_capture": up_capture,
                "Down_capture": down_capture,
            }
        )
    return pd.DataFrame(rows)


def build_comparison_table(
    portfolios: dict[str, pd.Series],
    exposures: pd.DataFrame,
    Sigma: pd.DataFrame,
    mu: pd.Series,
    factor_covariance: pd.DataFrame,
    specific_risk: pd.DataFrame,
    design_cols: list[str] = DESIGN_COLS,
) -> pd.DataFrame:
    """One row per named portfolio (weights Series): annualized return/vol/Sharpe, concentration,
    factor vs specific risk split. `design_cols` defaults to DESIGN_COLS; the segmented risk model
    passes its joint column list.
    """
    rows = []
    for name, weights in portfolios.items():
        symbols = list(weights.index)
        w = weights.to_numpy()
        ret_daily = float(w @ mu.reindex(symbols).to_numpy())
        vol_daily = float(np.sqrt(w @ Sigma.loc[symbols, symbols].to_numpy() @ w))
        decomp = risk_decomposition(weights, exposures, factor_covariance, specific_risk, design_cols=design_cols)
        conc = concentration_stats(weights)
        rows.append(
            {
                "Portfolio": name,
                "Return_annualized": ret_daily * config.TRADING_DAYS_PER_YEAR,
                "Vol_annualized": vol_daily * (config.TRADING_DAYS_PER_YEAR ** 0.5),
                "Sharpe": (ret_daily / vol_daily) * (config.TRADING_DAYS_PER_YEAR ** 0.5) if vol_daily > 0 else np.nan,
                "Factor_risk_pct": decomp["factor_pct"],
                "Specific_risk_pct": decomp["specific_pct"],
                "N_holdings": conc["n_holdings"],
                "Effective_N_holdings": conc["effective_n_holdings"],
                "Top5_weight": conc["top5_weight"],
            }
        )
    return pd.DataFrame(rows)
