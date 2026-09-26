"""Independent validation of the portfolio optimization - the model_reviewer's checks on the
optimizer's output, not on the underlying factor model (that's validation/statistical_tests.py).
Same independence principle: these functions only read the optimization's output and report
findings, they never adjust the optimizer to make a check pass. Each test returns a dict with
name/statistic/threshold/verdict/detail, same shape as validation/statistical_tests.py so both
render through the same report format.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from factor_model import config
from factor_model.portfolio.analytics import compute_backtest_return_series, realized_mean_daily_return
from factor_model.regression.cross_sectional import DESIGN_COLS

logger = logging.getLogger(__name__)


def _verdict(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def test_constraint_satisfaction(
    portfolios: dict[str, pd.Series],
    tolerance: float = 1e-6,
    long_short_names: dict[str, float] | None = None,
) -> dict:
    """`long_short_names` maps a portfolio name allowed to hold negative (short) weights to its
    max-leverage cap (e.g. {"Long-Short GMV": 1.2}) - those are checked for sum=1 and
    sum(|w|)<=max_leverage instead of the long-only sum=1-and-w>=0 check.
    """
    long_short_names = long_short_names or {}
    violations = []
    for name, weights in portfolios.items():
        weight_sum = weights.sum()
        if abs(weight_sum - 1.0) > tolerance:
            violations.append(f"{name}: weights sum to {weight_sum:.6f}, not 1.0")
        if name in long_short_names:
            gross = weights.abs().sum()
            max_leverage = long_short_names[name]
            if gross > max_leverage + tolerance:
                violations.append(f"{name}: gross exposure {gross:.4f} exceeds max leverage {max_leverage}")
        else:
            min_weight = weights.min()
            if min_weight < -tolerance:
                violations.append(f"{name}: min weight {min_weight:.6f} < 0 (long-only violated)")

    name_label = "Long-only constraint satisfaction" if not long_short_names else "Constraint satisfaction (long-only sum=1/w>=0, long-short sum=1/leverage cap)"
    return {
        "name": name_label,
        "statistic": f"{len(portfolios) - len({v.split(':')[0] for v in violations})}/{len(portfolios)} portfolios satisfy their constraints",
        "threshold": f"tolerance {tolerance}",
        "verdict": _verdict(not violations),
        "detail": "; ".join(violations) if violations else "all portfolios satisfy their respective constraints",
    }


def test_covariance_psd(Sigma: pd.DataFrame, tolerance: float = -1e-8) -> dict:
    eigenvalues = np.linalg.eigvalsh(Sigma.to_numpy())
    min_eig = float(eigenvalues.min())
    return {
        "name": "Stock covariance matrix is positive semi-definite",
        "statistic": f"min eigenvalue = {min_eig:.2e}",
        "threshold": f"min eigenvalue >= {tolerance:.0e}",
        "verdict": _verdict(min_eig >= tolerance),
        "detail": "a non-PSD Sigma would mean the optimizer is minimizing a variance that isn't a real variance" if min_eig < tolerance else "",
    }


def test_optimizer_convergence(
    frontier_df: pd.DataFrame,
    gmv_converged: bool,
    additional_results: dict[str, bool] | None = None,
    name: str = "Optimizer convergence",
) -> dict:
    """`additional_results` covers standalone single-point optimizations beyond the GMV/frontier
    trace (e.g. {"Beta-Neutral": beta_neutral.converged}). `name` distinguishes this check's row
    when called more than once (e.g. once for the long-only frontier, once for the long-short
    one).
    """
    additional_results = additional_results or {}
    n_converged = int(frontier_df["converged"].sum()) + int(gmv_converged) + sum(int(v) for v in additional_results.values())
    n_total = len(frontier_df) + 1 + len(additional_results)
    extra_label = "".join(f" + {name}" for name in additional_results)
    return {
        "name": name,
        "statistic": f"{n_converged}/{n_total} optimization runs converged (GMV + {len(frontier_df)} frontier points{extra_label})",
        "threshold": "all runs should report scipy convergence success",
        "verdict": _verdict(n_converged == n_total),
        "detail": "" if n_converged == n_total else "non-converged points are excluded from the traced frontier and max-Sharpe selection",
    }


def test_variance_decomposition_consistency(
    portfolios: dict[str, pd.Series], Sigma: pd.DataFrame, decomp_by_portfolio: dict[str, dict], tolerance: float = 1e-10
) -> dict:
    """Confirms w'Sigma w (direct) equals the factor+specific decomposition sum - a correctness
    check on the decomposition arithmetic itself, not a modeling judgment call.
    """
    mismatches = []
    for name, weights in portfolios.items():
        w = weights.to_numpy()
        direct = float(w @ Sigma.loc[weights.index, weights.index].to_numpy() @ w)
        decomposed = decomp_by_portfolio[name]["total_variance_daily"]
        if abs(direct - decomposed) > tolerance * max(abs(direct), 1e-12):
            mismatches.append(f"{name}: direct={direct:.3e} vs decomposed={decomposed:.3e}")

    return {
        "name": "Variance decomposition consistency",
        "statistic": f"{len(portfolios) - len(mismatches)}/{len(portfolios)} portfolios: direct w'Sigma*w matches factor+specific decomposition",
        "threshold": f"relative tolerance {tolerance}",
        "verdict": _verdict(not mismatches),
        "detail": "; ".join(mismatches) if mismatches else "",
    }


def test_concentration(
    portfolios: dict[str, pd.Series],
    soft_target: float = config.PORTFOLIO_CONCENTRATION_FLAG_WEIGHT,
    hard_fail_weight: float = config.PORTFOLIO_CONCENTRATION_HARD_FAIL_WEIGHT,
) -> dict:
    """MAX_STOCK_WEIGHT is a soft penalty target in the optimizer (portfolio/optimize.py), not a
    hard bound - so exceeding `soft_target` a little is an accepted, intended outcome. WARN
    between soft_target and hard_fail_weight, FAIL only above hard_fail_weight.
    """
    warned, failed = [], []
    for name, weights in portfolios.items():
        top_symbol = weights.abs().idxmax()
        top_weight = float(weights.abs().loc[top_symbol])
        signed = weights.loc[top_symbol]
        label = f"{name}: {top_symbol} = {signed:.1%}"
        if top_weight > hard_fail_weight:
            failed.append(label)
        elif top_weight > soft_target:
            warned.append(label)

    verdict = "FAIL" if failed else ("WARN" if warned else "PASS")
    return {
        "name": "Single-name concentration",
        "statistic": "; ".join(
            f"{name}: max|w|={weights.abs().max():.1%} ({weights.abs().idxmax()}, signed={weights.loc[weights.abs().idxmax()]:.1%})"
            for name, weights in portfolios.items()
        ),
        "threshold": f"soft target {soft_target:.0%} (WARN above), hard fail above {hard_fail_weight:.0%}",
        "verdict": verdict,
        "detail": (
            f"FAIL - exceeds the hard-fail threshold: {'; '.join(failed)}"
            if failed
            else (
                f"WARN - over the soft {soft_target:.0%} target but within the {hard_fail_weight:.0%} hard-fail "
                f"threshold (expected under the soft-cap penalty): {'; '.join(warned)}"
                if warned
                else "no portfolio exceeds the soft concentration target"
            )
        ),
    }


def test_frontier_monotonicity(frontier_df: pd.DataFrame, name: str = "Efficient frontier monotonicity") -> dict:
    """Along a correctly-computed efficient frontier, volatility should be non-decreasing as
    the target return increases (each point is the *minimum* variance achieving that return).
    """
    converged = frontier_df[frontier_df["converged"]].sort_values("target_return")
    vol_diffs = converged["volatility"].diff().dropna()
    violations = int((vol_diffs < -1e-8).sum())

    return {
        "name": name,
        "statistic": f"{len(vol_diffs) - violations}/{len(vol_diffs)} consecutive frontier points have non-decreasing volatility",
        "threshold": "volatility must not decrease as target return increases",
        "verdict": _verdict(violations == 0),
        "detail": f"{violations} violations - suggests some frontier points didn't reach the global optimum" if violations else "",
    }


def test_realized_vs_predicted_risk(
    portfolios: dict[str, pd.Series],
    Sigma: pd.DataFrame,
    returns_wide: pd.DataFrame,
    lookback_days: int = 252,
) -> dict:
    """Bias-statistic-style check: apply each portfolio's (current, fixed) weights to the
    trailing `lookback_days` of actual historical stock returns and compare the realized
    volatility to the risk model's ex-ante prediction. This is a static backtest (weights held
    fixed, not rebalanced) - a genuine rolling backtest is future work, but this already tests
    whether the covariance matrix's risk *magnitude* is in the right ballpark.
    """
    rows = []
    for name, weights in portfolios.items():
        port_returns = compute_backtest_return_series(weights, returns_wide, lookback_days)
        symbols = [s for s in weights.index if s in returns_wide.columns]
        w = weights.loc[symbols]
        w = w / w.sum()
        realized_vol_daily = port_returns.std(ddof=1)
        predicted_vol_daily = float(np.sqrt(w.to_numpy() @ Sigma.loc[symbols, symbols].to_numpy() @ w.to_numpy()))
        ratio = realized_vol_daily / predicted_vol_daily if predicted_vol_daily > 0 else np.nan
        rows.append((name, realized_vol_daily, predicted_vol_daily, ratio))

    stat = "; ".join(f"{n}: realized/predicted={r:.2f}" for n, _, _, r in rows)
    bad = [n for n, _, _, r in rows if not (0.5 <= r <= 2.0)]

    return {
        "name": "Realized vs. predicted volatility (bias check)",
        "statistic": stat,
        "threshold": f"ratio should be roughly 0.5-2.0x over the trailing {lookback_days} days (static weights, not rebalanced)",
        "verdict": _verdict(not bad),
        "detail": (
            f"outside the sanity band: {bad} - the risk model may be under/over-stating risk for "
            "this portfolio, or the trailing window includes a regime the EWMA hasn't adapted to yet"
            if bad
            else "risk model's volatility prediction is in a reasonable range for all portfolios"
        ),
    }


def test_beta_neutrality(
    portfolios: dict[str, pd.Series], exposures: pd.DataFrame, target_name: str = "Beta-Neutral", tolerance: float = 1e-6
) -> dict:
    """Confirms the named beta-neutral portfolio's net exposure to the z-scored `beta` design
    column is actually ~0 - a correctness check on the optimizer's neutrality constraint, not a
    modeling judgment call.
    """
    if target_name not in portfolios:
        return {
            "name": "Beta neutrality",
            "statistic": f"no '{target_name}' portfolio present",
            "threshold": f"net beta exposure within +/-{tolerance}",
            "verdict": "PASS",
            "detail": "check skipped - no beta-neutral portfolio was constructed",
        }
    weights = portfolios[target_name]
    net_beta = float(weights.reindex(exposures.index).to_numpy() @ exposures.loc[weights.index, "beta"].to_numpy())
    return {
        "name": "Beta neutrality",
        "statistic": f"{target_name}: net beta exposure = {net_beta:.6f}",
        "threshold": f"net beta exposure within +/-{tolerance}",
        "verdict": _verdict(abs(net_beta) <= tolerance),
        "detail": "" if abs(net_beta) <= tolerance else "the neutrality equality constraint was not met to tolerance - check optimizer convergence",
    }


def test_expected_return_vs_realized_stock_returns(
    mu_stock: pd.Series, returns_wide: pd.DataFrame, lookback_days: int = config.PORTFOLIO_BACKTEST_LOOKBACK_DAYS
) -> dict:
    """Spearman rank correlation between the optimizer's projected per-stock expected return
    (mu_stock, built purely from factor exposures x significance-shrunk factor premia) and what
    each stock actually returned - full-sample (the same window the factor premia were
    estimated over, so a positive, significant correlation here is a basic sanity floor, not a
    strong claim) and separately the trailing lookback_days window, since the two can diverge
    when a factor's edge isn't regime-stable (see the 2018-21/2021-23/2023-present sub-period
    analysis, where only `size` was robust across regimes). A weak/negative recent-window
    correlation is flagged for visibility (WARN) rather than FAIL, since it's a known, already-
    disclosed limitation of a full-sample-estimated expected-return model, not a new defect.
    """
    symbols = [s for s in mu_stock.index if s in returns_wide.columns]
    projected = mu_stock.reindex(symbols)
    realized_full = realized_mean_daily_return(returns_wide[symbols])
    realized_recent = realized_mean_daily_return(returns_wide[symbols], lookback_days)

    corr_full, p_full = spearmanr(projected, realized_full)
    corr_recent, p_recent = spearmanr(projected, realized_recent)

    fail = corr_full <= 0 or p_full >= 0.05
    warn = not fail and (corr_recent <= 0 or p_recent >= 0.05)

    verdict = "FAIL" if fail else ("WARN" if warn else "PASS")
    return {
        "name": "Expected returns vs. realized stock returns (cross-sectional consistency)",
        "statistic": f"full-sample: rho={corr_full:.3f} (p={p_full:.3f}, n={len(symbols)}); trailing {lookback_days}d: rho={corr_recent:.3f} (p={p_recent:.3f})",
        "threshold": "full-sample rho must be positive and significant (p<0.05); trailing-window rho positive and significant for PASS, else WARN",
        "verdict": verdict,
        "detail": (
            "projected expected returns are not positively/significantly correlated with what "
            "stocks actually returned over the very sample the factor premia came from - the "
            "expected-return model may be structurally disconnected from realized outcomes"
            if fail
            else (
                "full-sample correlation holds (expected, since mu_stock is built from this same "
                "history) but the trailing-window correlation is weak or negative - consistent with "
                "the model's known regime-dependency (only `size` was stable across the three "
                "sub-period sub-models), not a new finding"
                if warn
                else "projected expected returns are positively and significantly correlated with "
                "realized returns both over the full sample and the trailing window"
            )
        ),
    }
