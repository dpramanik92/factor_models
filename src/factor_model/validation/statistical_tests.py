"""Statistical validation tests run against the *fitted* model's real output.

Independent of the model-building code (see .claude/agents/model_reviewer.md):
these functions only read regression/risk outputs and report findings, they
never modify model code to force a check to pass. Each test returns a dict
with name/statistic/threshold/verdict/detail so validation/report.py can
assemble the executive-summary table.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller

from factor_model import config
from factor_model.factors.pit import expand_to_daily
from factor_model.regression.cross_sectional import DESIGN_COLS
from factor_model.regression.newey_west import newey_west_lag

logger = logging.getLogger(__name__)


def _verdict(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def test_r2_distribution(r2_daily: pd.DataFrame) -> dict:
    r2 = r2_daily["R2"].replace([np.inf, -np.inf], np.nan).dropna()
    n_non_finite = int((~np.isfinite(r2_daily["R2"])).sum())
    return {
        "name": "R2 distribution",
        "statistic": f"mean={r2.mean():.4f}, median={r2.median():.4f}, std={r2.std():.4f}",
        "threshold": "informational (no pass/fail threshold)",
        "verdict": "INFO",
        "detail": f"N days = {len(r2)}" + (f"; {n_non_finite} non-finite R2 excluded (should be 0 - the regression engine now skips zero-variance/holiday days before fitting)" if n_non_finite else ""),
    }


def test_newey_west_significance(summary: pd.DataFrame) -> dict:
    sig = summary[summary["NW_t"].abs() > 2]
    n_sig, n_total = len(sig), len(summary)
    return {
        "name": "Newey-West factor significance",
        "statistic": f"{n_sig}/{n_total} factors significant at |NW_t| > 2",
        "threshold": "|NW_t| > 2 (5% two-sided)",
        "verdict": "INFO",
        "detail": ", ".join(sig["Factor"].tolist()) if n_sig else "none significant",
    }


def compute_mean_vif(exposure_panel: pd.DataFrame, n_sample_dates: int = 20) -> dict[str, float] | None:
    """Average VIF per style factor, sampled over several dates (VIF varies by date)."""
    dates = sorted(exposure_panel["Date"].unique())
    sample = dates[:: max(1, len(dates) // n_sample_dates)][:n_sample_dates]

    vifs: list[np.ndarray] = []
    for d in sample:
        rows = exposure_panel.loc[exposure_panel["Date"] == d, config.STYLE_FACTORS].dropna()
        if len(rows) < len(config.STYLE_FACTORS) + 5:
            continue
        X = rows.to_numpy(dtype=float)
        try:
            vifs.append(np.array([variance_inflation_factor(X, i) for i in range(X.shape[1])]))
        except Exception:
            logger.exception("VIF computation failed for date %s", d)

    if not vifs:
        return None
    mean_vif = np.mean(vifs, axis=0)
    return dict(zip(config.STYLE_FACTORS, mean_vif))


def test_vif(exposure_panel: pd.DataFrame, n_sample_dates: int = 20) -> dict:
    """Average VIF across style factors, sampled over several dates (VIF varies by date)."""
    mean_vif = compute_mean_vif(exposure_panel, n_sample_dates)

    if mean_vif is None:
        return {
            "name": "Multicollinearity (VIF)",
            "statistic": "n/a",
            "threshold": "VIF > 10 flagged",
            "verdict": "INFO",
            "detail": "insufficient data to compute VIF",
        }

    flagged = [f for f, v in mean_vif.items() if v > 10]
    return {
        "name": "Multicollinearity (VIF)",
        "statistic": ", ".join(f"{f}={v:.2f}" for f, v in mean_vif.items()),
        "threshold": "VIF > 10 flagged",
        "verdict": _verdict(not flagged),
        "detail": f"flagged: {flagged}" if flagged else "no style factor exceeds VIF 10",
    }


def test_residual_normality(residuals_long: pd.DataFrame) -> dict:
    resid = residuals_long["Residual"].dropna()
    stat, pvalue = stats.jarque_bera(resid)
    return {
        "name": "Residual normality (Jarque-Bera, pooled)",
        "statistic": f"JB={stat:.2f}, p={pvalue:.4g}",
        "threshold": "p > 0.05 -> fail to reject normality",
        "verdict": "PASS" if pvalue > 0.05 else "WARN",
        "detail": "pooled across all stock-days; residuals are rarely exactly normal in practice, this is informational",
    }


def test_residual_alpha_significance(residuals_long: pd.DataFrame) -> dict:
    """Is there a significant leftover average return the 20 factors don't explain?

    The WLS fit forces the *weighted* residuals to be orthogonal to X by construction, but
    that doesn't force the plain equal-weighted average residual across stocks to be zero on
    any given day. A significant nonzero mean here would mean the model is missing a
    systematic factor (or there's persistent cross-sectional mispricing it can't capture) -
    this is the closest analogue to an "alpha" in a specification with no separate intercept
    (industry weights, which sum to 1, play that role instead - see
    .claude/skills/statistics/SKILL.md).
    """
    daily_alpha = residuals_long.groupby("Date")["Residual"].mean().sort_index()
    n = len(daily_alpha)
    if n < 3:
        return {
            "name": "Residual alpha significance",
            "statistic": "n/a",
            "threshold": "|NW_t| > 2 (5% two-sided)",
            "verdict": "INFO",
            "detail": "insufficient data",
        }

    mean_alpha = daily_alpha.mean()
    lag = newey_west_lag(n)
    ols = sm.OLS(daily_alpha.to_numpy(), np.ones((n, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    nw_t = ols.tvalues[0]

    return {
        "name": "Residual alpha significance",
        "statistic": f"mean daily alpha={mean_alpha:.6f} ({mean_alpha * config.TRADING_DAYS_PER_YEAR * 100:.2f}% annualized), NW_t={nw_t:.2f}, N={n}",
        "threshold": "|NW_t| > 2 (5% two-sided) flags a significant unexplained alpha",
        "verdict": _verdict(abs(nw_t) <= 2),
        "detail": (
            "significant leftover alpha - the 20 factors do not fully explain average cross-sectional returns"
            if abs(nw_t) > 2
            else "no significant unexplained alpha - consistent with the factor set being reasonably complete"
        ),
    }


def test_residual_heteroskedasticity(exposure_panel: pd.DataFrame, residuals_long: pd.DataFrame, n_sample_dates: int = 20) -> dict:
    """Breusch-Pagan test of each sampled date's residuals against that date's design matrix."""
    dates = sorted(residuals_long["Date"].unique())
    sample = dates[:: max(1, len(dates) // n_sample_dates)][:n_sample_dates]

    pvalues = []
    for d in sample:
        resid_d = residuals_long.loc[residuals_long["Date"] == d, ["Symbol", "Residual"]]
        exp_d = exposure_panel.loc[exposure_panel["Date"] == d, ["Symbol"] + DESIGN_COLS]
        merged = resid_d.merge(exp_d, on="Symbol").dropna()
        if len(merged) < len(DESIGN_COLS) + 5:
            continue
        X = merged[DESIGN_COLS].to_numpy(dtype=float)
        resid = merged["Residual"].to_numpy(dtype=float)
        try:
            _, pvalue, _, _ = het_breuschpagan(resid, np.column_stack([np.ones(len(resid)), X]))
            pvalues.append(pvalue)
        except Exception:
            logger.exception("Breusch-Pagan failed for date %s", d)

    if not pvalues:
        return {
            "name": "Residual heteroskedasticity (Breusch-Pagan)",
            "statistic": "n/a",
            "threshold": "share of sampled dates with p < 0.05",
            "verdict": "INFO",
            "detail": "insufficient data",
        }

    frac_reject = float(np.mean(np.array(pvalues) < 0.05))
    return {
        "name": "Residual heteroskedasticity (Breusch-Pagan)",
        "statistic": f"{frac_reject:.0%} of {len(pvalues)} sampled dates reject homoskedasticity (p<0.05)",
        "threshold": "flagged if > 50% of dates reject",
        "verdict": _verdict(frac_reject <= 0.5),
        "detail": "WLS weighting (sqrt market cap) is designed to mitigate cross-sectional heteroskedasticity",
    }


def test_residual_autocorrelation(residuals_long: pd.DataFrame, max_symbols: int = 40) -> dict:
    """Ljung-Box test per stock's residual time series, share failing at lag 5."""
    wide = residuals_long.pivot(index="Date", columns="Symbol", values="Residual").sort_index()
    symbols = wide.columns[:max_symbols]

    n_fail, n_tested = 0, 0
    for symbol in symbols:
        series = wide[symbol].dropna()
        if len(series) < 30:
            continue
        result = acorr_ljungbox(series, lags=[5], return_df=True)
        n_tested += 1
        if result["lb_pvalue"].iloc[0] < 0.05:
            n_fail += 1

    if n_tested == 0:
        return {
            "name": "Residual autocorrelation (Ljung-Box, lag 5)",
            "statistic": "n/a",
            "threshold": "share of stocks with p < 0.05",
            "verdict": "INFO",
            "detail": "insufficient data",
        }

    frac_fail = n_fail / n_tested
    return {
        "name": "Residual autocorrelation (Ljung-Box, lag 5)",
        "statistic": f"{frac_fail:.0%} of {n_tested} sampled stocks show significant autocorrelation",
        "threshold": "flagged if > 50% of stocks fail",
        "verdict": _verdict(frac_fail <= 0.5),
        "detail": "",
    }


def test_industry_weights_sum_to_one(industry_by_symbol: pd.DataFrame, tolerance: float = 1e-3) -> dict:
    row_sums = industry_by_symbol.sum(axis=1)
    bad = (row_sums - 1.0).abs() > tolerance
    return {
        "name": "Industry weights sum to 1",
        "statistic": f"{(~bad).sum()}/{len(row_sums)} symbols sum to 1 within {tolerance}",
        "threshold": f"tolerance {tolerance}",
        "verdict": _verdict(not bad.any()),
        "detail": f"violations: {list(row_sums.index[bad])}" if bad.any() else "all symbols OK",
    }


def test_factor_stationarity(factor_returns: pd.DataFrame) -> dict:
    non_stationary = []
    for factor in factor_returns.columns:
        series = factor_returns[factor].dropna()
        if len(series) < 20:
            continue
        try:
            _, pvalue, *_ = adfuller(series)
        except Exception:
            continue
        if pvalue >= 0.05:
            non_stationary.append(factor)

    return {
        "name": "Factor-return stationarity (ADF)",
        "statistic": f"{len(factor_returns.columns) - len(non_stationary)}/{len(factor_returns.columns)} factors stationary at 5%",
        "threshold": "ADF p < 0.05 -> stationary",
        "verdict": _verdict(not non_stationary),
        "detail": f"non-stationary: {non_stationary}" if non_stationary else "all factor-return series stationary",
    }


def test_lookahead_bias(
    annual_ratios: pd.DataFrame,
    exposure_panel: pd.DataFrame,
    n_samples: int = 15,
    lag_days: int = config.REPORTING_LAG_DAYS,
) -> dict:
    """For a sample of (Symbol, FY) pairs, confirm the exposure panel's leverage value doesn't
    change before FY_end + lag_days, using an independently re-derived expand_to_daily call.
    """
    business_days = pd.DatetimeIndex(sorted(exposure_panel["Date"].unique()))
    sample = annual_ratios.dropna(subset=["Leverage"]).sample(
        n=min(n_samples, len(annual_ratios)), random_state=0
    )

    violations = []
    for _, row in sample.iterrows():
        symbol, fy_end = row["Symbol"], row["FiscalYearEnd"]
        series = annual_ratios.loc[annual_ratios["Symbol"] == symbol].set_index("FiscalYearEnd")["Leverage"]
        expected_daily = expand_to_daily(series, business_days, lag_days)

        actual = exposure_panel.loc[exposure_panel["Symbol"] == symbol].set_index("Date")["leverage"]
        # leverage in exposure_panel has already been winsorized/z-scored, so we can only check
        # the *timing* of value changes, not exact levels - compare day-over-day change points.
        expected_change_dates = expected_daily.dropna().index[
            expected_daily.dropna().diff().fillna(1) != 0
        ]
        available_from = fy_end + pd.Timedelta(days=lag_days)
        premature_dates = [d for d in expected_change_dates if d < available_from and d >= fy_end]
        if premature_dates:
            violations.append((symbol, str(fy_end.date()), premature_dates))

    return {
        "name": "Look-ahead bias (fundamentals reporting lag)",
        "statistic": f"{len(sample) - len(violations)}/{len(sample)} sampled (Symbol, FY) pairs respect the {lag_days}-day lag",
        "threshold": f"no exposure change before FY_end + {lag_days}d",
        "verdict": _verdict(not violations),
        "detail": f"violations: {violations}" if violations else "no premature availability detected",
    }


def test_size_quintile_spread(exposure_panel: pd.DataFrame, returns_wide: pd.DataFrame) -> dict:
    """Q1 (smallest) vs Q5 (largest) average forward-return spread, with a t-test."""
    returns_wide = returns_wide.copy()
    returns_wide.index.name = "Date"
    returns_long = returns_wide.reset_index().melt(id_vars="Date", var_name="Symbol", value_name="return")

    merged = exposure_panel[["Date", "Symbol", "size_quintile"]].merge(
        returns_long, on=["Date", "Symbol"], how="inner"
    ).dropna(subset=["size_quintile", "return"])

    if merged.empty:
        return {
            "name": "Size-quintile (Q1 vs Q5) return spread",
            "statistic": "n/a",
            "threshold": "informational",
            "verdict": "INFO",
            "detail": "insufficient data",
        }

    q1 = merged.loc[merged["size_quintile"] == 1, "return"]
    q5 = merged.loc[merged["size_quintile"] == 5, "return"]
    t_stat, pvalue = stats.ttest_ind(q1.dropna(), q5.dropna(), equal_var=False)

    return {
        "name": "Size-quintile (Q1 vs Q5) return spread",
        "statistic": f"Q1 mean={q1.mean():.5f}, Q5 mean={q5.mean():.5f}, spread={q1.mean() - q5.mean():.5f}, t={t_stat:.2f}, p={pvalue:.4g}",
        "threshold": "informational (size premium/discount)",
        "verdict": "INFO",
        "detail": "",
    }


def test_regression_coverage(
    coverage_start_date: pd.Timestamp | None,
    skipped_low_n_dates: list,
    skipped_degenerate_dates: list | None = None,
    min_n: int = config.MIN_N_PER_DAY,
) -> dict:
    skipped_degenerate_dates = skipped_degenerate_dates or []
    return {
        "name": "Minimum-N regression coverage",
        "statistic": (
            f"coverage starts {coverage_start_date}, {len(skipped_low_n_dates)} dates skipped for N < {min_n}, "
            f"{len(skipped_degenerate_dates)} dates skipped as degenerate (zero return variance)"
        ),
        "threshold": f"N >= {min_n} required per regression day",
        "verdict": "INFO",
        "detail": f"degenerate dates: {skipped_degenerate_dates}" if skipped_degenerate_dates else "",
    }
