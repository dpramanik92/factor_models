"""Long-only mean-variance optimization: GMV, an efficient frontier, and a max-Sharpe portfolio
picked from that frontier.

Variance minimization subject to a target-return equality constraint is a convex QP (Sigma is
PSD by construction - see risk_model.py), so scipy's SLSQP reliably finds the global optimum for
each frontier point. Deliberately avoided: maximizing Sharpe ratio directly, which is
non-convex and can converge to a local optimum - tracing the frontier via the convex target-
return formulation and then picking the frontier point with the best realized Sharpe ratio is
both more robust and gives the frontier plot for free.

Position limits (max_stock_weight, max_industry_weight) are optional, off by default (1.0 = no
cap) so the low-level analytical-solution unit tests keep validating the *unconstrained* GMV
formula - the real pipeline (portfolio/pipeline.py) passes config.MAX_STOCK_WEIGHT/
MAX_INDUSTRY_WEIGHT explicitly. Industry caps use each stock's fractional industry exposure
(may span multiple industries - see factors/industry.py), so a stock at 50/50 between two
industries contributes half its weight to each industry's cap sum.

`max_stock_weight` may be a single float (broadcast to every stock, the common case) or a
length-n array/Series aligned to Sigma's symbol order, giving each stock its own cap - used by
the segmented portfolio (portfolio/segmented_risk_model.py) to try a looser single-name cap on
the "rest" (non-top100) segment than the top100 segment, since a symbol-array cap plugs straight
into the same bounds/penalty-term machinery a scalar cap does (numpy broadcasting handles both).

The single-stock cap has two enforcement modes, chosen by whether `stock_weight_penalty` is
given:
  - Hard (default, stock_weight_penalty=None): max_stock_weight is a literal upper bound in the
    optimizer's `bounds` - no solution can ever exceed it.
  - Soft (stock_weight_penalty > 0): bounds allow up to 100%, and a quadratic penalty
    `stock_weight_penalty * sum(relu(w - max_stock_weight)^2)` is added to the variance
    objective instead. Exceeding the cap is discouraged, not forbidden - how far a stock ends up
    over depends on how much variance/return is at stake relative to the penalty. Requested
    after the hard 10% cap was found to force GMV/Max-Sharpe away from otherwise-attractive
    concentrated positions with no way to trade off - see config.STOCK_WEIGHT_SOFT_PENALTY_COEF.
The industry cap has no soft mode; it stays a hard bound in both cases.

`group_exposure`/`group_max_weight`/`group_weight_penalty` add one more, independent soft
penalty: `group_weight_penalty * relu(w @ group_exposure - group_max_weight)^2`, discouraging
(never forbidding) the total weight placed in an arbitrary group of stocks (e.g. "the whole
non-top100 segment") from exceeding a target - the group-level analogue of the single-stock soft
cap, but for a linear combination of weights instead of one element. `group_exposure` is a
Symbol-indexed 0/1 (or fractional) membership vector, reindexed to Sigma's symbol order here;
unlike the (hard) per-industry cap, this has no feasibility-LP/bounds involvement at all, since a
soft penalty never changes what's feasible - it only nudges the objective.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from factor_model import config

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    weights: pd.Series  # Symbol-indexed, sums to 1, all >= 0
    expected_return: float  # daily, excess
    volatility: float  # daily
    converged: bool
    message: str


def _long_only_constraints(n: int, max_stock_weight: float | np.ndarray) -> tuple[list[dict], list[tuple[float, float]]]:
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    caps = np.broadcast_to(max_stock_weight, n)
    bounds = [(0.0, float(cap)) for cap in caps]
    return constraints, bounds


def _industry_cap_constraints(industry_exposures: np.ndarray, max_industry_weight: float) -> list[dict]:
    """One inequality constraint per industry column: cap - w @ exposure_column >= 0."""
    return [
        {"type": "ineq", "fun": lambda w, col=industry_exposures[:, j]: max_industry_weight - w @ col}
        for j in range(industry_exposures.shape[1])
    ]


def _neutrality_constraints(neutralize_exposures: np.ndarray) -> list[dict]:
    """One equality constraint per column to neutralize: w @ exposure_column == 0 - e.g. a
    beta-neutral portfolio zeroes out net exposure to the z-scored `beta` design column.
    """
    return [
        {"type": "eq", "fun": lambda w, col=neutralize_exposures[:, j]: w @ col}
        for j in range(neutralize_exposures.shape[1])
    ]


def _lp_equality_rows(n: int, neutralize_exposures: np.ndarray | None) -> tuple[list[np.ndarray], list[float]]:
    """The sum-to-1 row plus, if given, one zero-net-exposure row per neutralized column -
    shared between the feasibility LP and the max-achievable-return LP so both respect the same
    equality constraints the real QP will be solved under.
    """
    A_eq = [np.ones(n)]
    b_eq = [1.0]
    if neutralize_exposures is not None:
        for j in range(neutralize_exposures.shape[1]):
            A_eq.append(neutralize_exposures[:, j])
            b_eq.append(0.0)
    return A_eq, b_eq


def _feasible_starting_point(
    n: int,
    max_stock_weight: float | np.ndarray,
    industry_exposures: np.ndarray | None,
    max_industry_weight: float,
    neutralize_exposures: np.ndarray | None = None,
) -> np.ndarray:
    """A phase-1-style feasibility LP (zero objective) to find a starting point that already
    satisfies the long-only/sum-to-1/position-cap/neutrality constraints. SLSQP handles the
    uniform 1/n starting point fine when unconstrained, but with tight industry caps (or a
    neutrality constraint, which 1/n essentially never satisfies) 1/n can itself already violate
    a constraint, which was causing spurious non-convergence ("positive directional derivative").
    """
    bounds = [(0.0, float(cap)) for cap in np.broadcast_to(max_stock_weight, n)]
    A_ub, b_ub = None, None
    if industry_exposures is not None and max_industry_weight < 1.0:
        A_ub = industry_exposures.T
        b_ub = np.full(industry_exposures.shape[1], max_industry_weight)
    A_eq, b_eq = _lp_equality_rows(n, neutralize_exposures)
    res = linprog(c=np.zeros(n), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.success:
        return res.x
    logger.warning("Feasibility LP for starting point failed (%s); falling back to uniform 1/n", res.message)
    return np.ones(n) / n


def _max_achievable_return(
    mu: np.ndarray,
    industry_exposures: np.ndarray | None,
    max_stock_weight: float | np.ndarray,
    max_industry_weight: float,
    neutralize_exposures: np.ndarray | None = None,
) -> float:
    """Highest expected return reachable under the long-only + position-cap + neutrality
    constraints, via a small LP - needed because with caps (or a neutrality constraint) in
    place, no portfolio can get anywhere near a single high-mu stock's own return, so the
    frontier's target-return range must respect them too (using mu.max() as the upper bound, as
    the uncapped code did, would make most of the upper frontier infeasible).
    """
    n = len(mu)
    bounds = [(0.0, float(cap)) for cap in np.broadcast_to(max_stock_weight, n)]
    A_ub, b_ub = None, None
    if industry_exposures is not None and max_industry_weight < 1.0:
        A_ub = industry_exposures.T
        b_ub = np.full(industry_exposures.shape[1], max_industry_weight)
    A_eq, b_eq = _lp_equality_rows(n, neutralize_exposures)
    res = linprog(c=-mu, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        logger.warning("LP for max achievable return under position caps did not converge (%s); falling back to mu.max()", res.message)
        return float(mu.max())
    return float(-res.fun)


def minimize_variance(
    Sigma: pd.DataFrame,
    target_return: float | None = None,
    mu: pd.Series | None = None,
    industry_exposures: pd.DataFrame | None = None,
    max_stock_weight: float | pd.Series = 1.0,
    max_industry_weight: float = 1.0,
    stock_weight_penalty: float | None = None,
    neutralize_exposures: pd.DataFrame | None = None,
    group_exposure: pd.Series | None = None,
    group_max_weight: float = 1.0,
    group_weight_penalty: float | None = None,
) -> OptimizationResult:
    """Global minimum-variance long-only portfolio, or (if target_return/mu given) the
    minimum-variance long-only portfolio achieving exactly that expected return - one point on
    the efficient frontier. `industry_exposures` (Symbol x industry columns, aligned to Sigma's
    symbols) is required for max_industry_weight < 1.0 to have any effect. `max_stock_weight` may
    be a Symbol-indexed pd.Series instead of a single float, giving each stock its own cap
    (reindexed to Sigma's symbol order here) - see the module docstring for the max_stock_weight
    hard-vs-soft (stock_weight_penalty) distinction. `neutralize_exposures` (Symbol x
    one-or-more DESIGN_COLS, e.g. just `beta`) adds a hard equality constraint forcing the
    portfolio's net exposure to each given column to exactly zero - e.g. a beta-neutral
    portfolio. Long-only weights can still zero out a z-scored exposure (roughly half the
    universe sits above/below the cross-sectional mean each day), no shorting required.
    `group_exposure`/`group_max_weight`/`group_weight_penalty` add the group-level soft cap
    described in the module docstring - e.g. capping total weight in the "rest" segment.
    """
    symbols = list(Sigma.index)
    n = len(symbols)
    Sigma_np = Sigma.to_numpy()

    max_stock_weight_np = (
        max_stock_weight.reindex(symbols).to_numpy() if isinstance(max_stock_weight, pd.Series) else max_stock_weight
    )
    industry_np = industry_exposures.reindex(symbols).to_numpy() if industry_exposures is not None else None
    neutralize_np = neutralize_exposures.reindex(symbols).to_numpy() if neutralize_exposures is not None else None
    group_np = group_exposure.reindex(symbols).to_numpy() if group_exposure is not None else None
    soft_cap = stock_weight_penalty is not None and stock_weight_penalty > 0
    soft_group_cap = group_np is not None and group_weight_penalty is not None and group_weight_penalty > 0
    stock_bound = 1.0 if soft_cap else max_stock_weight_np

    constraints, bounds = _long_only_constraints(n, stock_bound)
    if target_return is not None:
        mu_np = mu.reindex(symbols).to_numpy()
        constraints.append({"type": "eq", "fun": lambda w, m=mu_np, t=target_return: w @ m - t})
    if industry_np is not None and max_industry_weight < 1.0:
        constraints += _industry_cap_constraints(industry_np, max_industry_weight)
    if neutralize_np is not None:
        constraints += _neutrality_constraints(neutralize_np)

    needs_feasible_start = (
        bool(np.any(np.asarray(stock_bound) < 1.0))
        or (industry_np is not None and max_industry_weight < 1.0)
        or neutralize_np is not None
    )
    if needs_feasible_start:
        x0 = _feasible_starting_point(n, stock_bound, industry_np, max_industry_weight, neutralize_np)
    else:
        x0 = np.ones(n) / n

    def objective(w: np.ndarray) -> float:
        val = w @ Sigma_np @ w
        if soft_cap:
            excess = np.clip(w - max_stock_weight_np, 0.0, None)
            val = val + stock_weight_penalty * np.sum(excess ** 2)
        if soft_group_cap:
            group_excess = max(0.0, w @ group_np - group_max_weight)
            val = val + group_weight_penalty * group_excess ** 2
        return val

    def grad(w: np.ndarray) -> np.ndarray:
        g = 2 * Sigma_np @ w
        if soft_cap:
            excess = np.clip(w - max_stock_weight_np, 0.0, None)
            g = g + 2 * stock_weight_penalty * excess
        if soft_group_cap:
            group_excess = max(0.0, w @ group_np - group_max_weight)
            g = g + 2 * group_weight_penalty * group_excess * group_np
        return g

    res = minimize(
        objective, x0, jac=grad, bounds=bounds, constraints=constraints, method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    weights = pd.Series(np.clip(res.x, 0, None), index=symbols)
    weights = weights / weights.sum()  # renormalize away tiny constraint-tolerance slack

    ret = float(weights.to_numpy() @ mu.reindex(symbols).to_numpy()) if mu is not None else float("nan")
    vol = float(np.sqrt(weights.to_numpy() @ Sigma_np @ weights.to_numpy()))

    return OptimizationResult(weights=weights, expected_return=ret, volatility=vol, converged=res.success, message=res.message)


def trace_efficient_frontier(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    n_points: int = config.FRONTIER_N_POINTS,
    industry_exposures: pd.DataFrame | None = None,
    max_stock_weight: float | pd.Series = 1.0,
    max_industry_weight: float = 1.0,
    stock_weight_penalty: float | None = None,
    neutralize_exposures: pd.DataFrame | None = None,
    group_exposure: pd.Series | None = None,
    group_max_weight: float = 1.0,
    group_weight_penalty: float | None = None,
) -> tuple[pd.DataFrame, OptimizationResult]:
    """Returns (frontier_df, gmv_result). frontier_df has one row per traced point:
    target_return, volatility, sharpe, converged. `max_stock_weight` may be a Symbol-indexed
    pd.Series instead of a single float - see minimize_variance. See minimize_variance for
    neutralize_exposures and the group_exposure/group_max_weight/group_weight_penalty soft cap.
    """
    symbols = list(Sigma.index)
    mu_aligned = mu.reindex(symbols)
    max_stock_weight_np = (
        max_stock_weight.reindex(symbols).to_numpy() if isinstance(max_stock_weight, pd.Series) else max_stock_weight
    )
    industry_np = industry_exposures.reindex(symbols).to_numpy() if industry_exposures is not None else None
    neutralize_np = neutralize_exposures.reindex(symbols).to_numpy() if neutralize_exposures is not None else None

    gmv = minimize_variance(
        Sigma, target_return=None, mu=mu_aligned, industry_exposures=industry_exposures,
        max_stock_weight=max_stock_weight, max_industry_weight=max_industry_weight,
        stock_weight_penalty=stock_weight_penalty, neutralize_exposures=neutralize_exposures,
        group_exposure=group_exposure, group_max_weight=group_max_weight, group_weight_penalty=group_weight_penalty,
    )

    # Always use max_stock_weight (not the soft-mode 1.0 optimizer bound) here: this LP has no
    # notion of the soft penalty, so bounding it at the true per-stock cap keeps the frontier's
    # target-return range at a level the penalized QP can actually reach without extreme,
    # numerically degenerate concentration - letting this LP assume 100% per stock (as the
    # optimizer's own bounds do in soft mode) produced an unreachable top-of-frontier target that
    # made the last point fail to converge. Same reasoning applies to neutralize_np: the LP must
    # respect it too, or the top of the frontier again targets an unreachable return.
    max_mu_return = _max_achievable_return(mu_aligned.to_numpy(), industry_np, max_stock_weight_np, max_industry_weight, neutralize_np)
    if max_mu_return <= gmv.expected_return:
        logger.warning("Highest achievable expected return under position caps is not above the GMV portfolio's - frontier will be degenerate")
        max_mu_return = gmv.expected_return + 1e-8

    targets = np.linspace(gmv.expected_return, max_mu_return, n_points)
    rows = []
    weights_by_target = {}
    for t in targets:
        result = minimize_variance(
            Sigma, target_return=t, mu=mu_aligned, industry_exposures=industry_exposures,
            max_stock_weight=max_stock_weight, max_industry_weight=max_industry_weight,
            stock_weight_penalty=stock_weight_penalty, neutralize_exposures=neutralize_exposures,
            group_exposure=group_exposure, group_max_weight=group_max_weight, group_weight_penalty=group_weight_penalty,
        )
        sharpe = result.expected_return / result.volatility if result.volatility > 0 else np.nan
        rows.append(
            {
                "target_return": t,
                "realized_return": result.expected_return,
                "volatility": result.volatility,
                "sharpe": sharpe,
                "converged": result.converged,
            }
        )
        weights_by_target[t] = result.weights

    frontier_df = pd.DataFrame(rows)
    frontier_df.attrs["weights_by_target"] = weights_by_target
    return frontier_df, gmv


def _split_variable_lp(
    n: int, industry_exposures: np.ndarray | None, max_stock_weight: float, max_industry_weight: float, max_leverage: float
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[float], list[tuple[float, float]]]:
    """Shared LP setup for the long-short helper LPs below: split each w_i into (w_i^+, w_i^-)
    >= 0 with w_i = w_i^+ - w_i^-, so the non-smooth sum(|w|) <= max_leverage constraint becomes
    the plain linear inequality sum(w^+) + sum(w^-) <= max_leverage. This 2n-dim split is only
    ever used inside these LP helpers (to find a feasible start / the max achievable return) -
    the actual QP solved by minimize_variance_long_short works in the original n-dim w directly.
    Returns (A_ub, b_ub, A_eq, b_eq, bounds) for the 2n-dim [w+; w-] LP variable.
    """
    bounds = [(0.0, max_stock_weight)] * (2 * n)
    A_eq = [np.concatenate([np.ones(n), -np.ones(n)])]  # sum(w+) - sum(w-) = 1
    b_eq = [1.0]
    A_ub_rows = [np.concatenate([np.ones(n), np.ones(n)])]  # sum(w+) + sum(w-) <= max_leverage
    b_ub = [max_leverage]
    if industry_exposures is not None and max_industry_weight < 1.0:
        for j in range(industry_exposures.shape[1]):
            col = industry_exposures[:, j]
            A_ub_rows.append(np.concatenate([col, -col]))  # net exposure <= cap
            b_ub.append(max_industry_weight)
            A_ub_rows.append(np.concatenate([-col, col]))  # net exposure >= -cap
            b_ub.append(max_industry_weight)
    return np.array(A_ub_rows), np.array(b_ub), A_eq, b_eq, bounds


def _feasible_starting_point_long_short(
    n: int, industry_exposures: np.ndarray | None, max_stock_weight: float, max_industry_weight: float, max_leverage: float
) -> np.ndarray:
    A_ub, b_ub, A_eq, b_eq, bounds = _split_variable_lp(n, industry_exposures, max_stock_weight, max_industry_weight, max_leverage)
    res = linprog(c=np.zeros(2 * n), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.success:
        return res.x[:n] - res.x[n:]
    logger.warning("Long-short feasibility LP failed (%s); falling back to uniform 1/n", res.message)
    return np.ones(n) / n


def _max_achievable_return_long_short(
    mu: np.ndarray, industry_exposures: np.ndarray | None, max_stock_weight: float, max_industry_weight: float, max_leverage: float
) -> float:
    n = len(mu)
    A_ub, b_ub, A_eq, b_eq, bounds = _split_variable_lp(n, industry_exposures, max_stock_weight, max_industry_weight, max_leverage)
    c = np.concatenate([-mu, mu])  # maximize mu@(w+ - w-) == minimize -mu@w+ + mu@w-
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        logger.warning("LP for max achievable long-short return did not converge (%s); falling back to mu.max()", res.message)
        return float(mu.max())
    return float(-res.fun)


def minimize_variance_long_short(
    Sigma: pd.DataFrame,
    target_return: float | None = None,
    mu: pd.Series | None = None,
    industry_exposures: pd.DataFrame | None = None,
    max_stock_weight: float = 1.0,
    max_industry_weight: float = 1.0,
    max_leverage: float = 1.0,
) -> OptimizationResult:
    """Long-short variant of minimize_variance: net exposure still sums to 1 (fully invested, net
    long - unchanged from the long-only formulation), but individual weights may now be negative
    (short) within +/-max_stock_weight, subject to a gross-leverage cap
    sum(|w_i|) <= max_leverage (e.g. 1.2 -> up to 10% of capital short, funding 10% extra long,
    for 120% gross / 100% net). Industry caps apply symmetrically to net (signed) industry
    exposure. No soft-cap mode here (unlike the long-only minimize_variance) - kept simple; the
    leverage constraint itself already does most of the work of bounding concentration.

    Solved as a smooth QP in the split variable x = [w+; w-] >= 0 with w = w+ - w- (standard
    L1-constrained-QP reformulation), NOT directly in w with a sum(|w|) constraint - an earlier
    version tried that with an analytic subgradient for the abs() kink, which made convergence
    *worse* (9/25 frontier points, down from 21/25 with no jac at all): a hard sign() jump is
    genuinely unfriendly to SLSQP's local linearization regardless of how the gradient at the
    kink is defined. The split-variable form has no kink anywhere - fully linear constraints, a
    smooth quadratic objective - and converges reliably (26/26).
    """
    symbols = list(Sigma.index)
    n = len(symbols)
    Sigma_np = Sigma.to_numpy()
    industry_np = industry_exposures.reindex(symbols).to_numpy() if industry_exposures is not None else None

    bounds = [(0.0, max_stock_weight)] * (2 * n)

    def w_of(x: np.ndarray) -> np.ndarray:
        return x[:n] - x[n:]

    constraints = [
        {
            "type": "eq",
            "fun": lambda x: np.sum(x[:n]) - np.sum(x[n:]) - 1.0,
            "jac": lambda x: np.concatenate([np.ones(n), -np.ones(n)]),
        },
        {
            "type": "ineq",
            "fun": lambda x: max_leverage - np.sum(x),
            "jac": lambda x: -np.ones(2 * n),
        },
    ]
    if target_return is not None:
        mu_np = mu.reindex(symbols).to_numpy()
        constraints.append(
            {
                "type": "eq",
                "fun": lambda x, m=mu_np, t=target_return: w_of(x) @ m - t,
                "jac": lambda x, m=mu_np: np.concatenate([m, -m]),
            }
        )
    if industry_np is not None and max_industry_weight < 1.0:
        for j in range(industry_np.shape[1]):
            col = industry_np[:, j]
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, c=col: max_industry_weight - w_of(x) @ c,
                    "jac": lambda x, c=col: np.concatenate([-c, c]),
                }
            )
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, c=col: w_of(x) @ c + max_industry_weight,
                    "jac": lambda x, c=col: np.concatenate([c, -c]),
                }
            )

    w0 = _feasible_starting_point_long_short(n, industry_np, max_stock_weight, max_industry_weight, max_leverage)
    x0 = np.concatenate([np.clip(w0, 0.0, None), np.clip(-w0, 0.0, None)])

    def objective(x: np.ndarray) -> float:
        w = w_of(x)
        return w @ Sigma_np @ w

    def grad(x: np.ndarray) -> np.ndarray:
        g = 2 * Sigma_np @ w_of(x)
        return np.concatenate([g, -g])

    res = minimize(
        objective, x0, jac=grad, bounds=bounds, constraints=constraints, method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    w_final = w_of(res.x)
    weights = pd.Series(w_final, index=symbols)
    weights = weights / weights.sum()  # renormalize net exposure to exactly 1.0

    ret = float(weights.to_numpy() @ mu.reindex(symbols).to_numpy()) if mu is not None else float("nan")
    vol = float(np.sqrt(weights.to_numpy() @ Sigma_np @ weights.to_numpy()))

    return OptimizationResult(weights=weights, expected_return=ret, volatility=vol, converged=res.success, message=res.message)


def trace_efficient_frontier_long_short(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    n_points: int = config.FRONTIER_N_POINTS,
    industry_exposures: pd.DataFrame | None = None,
    max_stock_weight: float = 1.0,
    max_industry_weight: float = 1.0,
    max_leverage: float = 1.0,
) -> tuple[pd.DataFrame, OptimizationResult]:
    """Long-short analogue of trace_efficient_frontier - same shape of return (frontier_df,
    gmv_result), consumable by the same pick_max_sharpe.
    """
    symbols = list(Sigma.index)
    mu_aligned = mu.reindex(symbols)
    industry_np = industry_exposures.reindex(symbols).to_numpy() if industry_exposures is not None else None

    gmv = minimize_variance_long_short(
        Sigma, target_return=None, mu=mu_aligned, industry_exposures=industry_exposures,
        max_stock_weight=max_stock_weight, max_industry_weight=max_industry_weight, max_leverage=max_leverage,
    )

    max_mu_return = _max_achievable_return_long_short(mu_aligned.to_numpy(), industry_np, max_stock_weight, max_industry_weight, max_leverage)
    if max_mu_return <= gmv.expected_return:
        logger.warning("Highest achievable long-short expected return is not above the GMV portfolio's - frontier will be degenerate")
        max_mu_return = gmv.expected_return + 1e-8

    targets = np.linspace(gmv.expected_return, max_mu_return, n_points)
    rows = []
    weights_by_target = {}
    for t in targets:
        result = minimize_variance_long_short(
            Sigma, target_return=t, mu=mu_aligned, industry_exposures=industry_exposures,
            max_stock_weight=max_stock_weight, max_industry_weight=max_industry_weight, max_leverage=max_leverage,
        )
        sharpe = result.expected_return / result.volatility if result.volatility > 0 else np.nan
        rows.append(
            {
                "target_return": t,
                "realized_return": result.expected_return,
                "volatility": result.volatility,
                "sharpe": sharpe,
                "converged": result.converged,
            }
        )
        weights_by_target[t] = result.weights

    frontier_df = pd.DataFrame(rows)
    frontier_df.attrs["weights_by_target"] = weights_by_target
    return frontier_df, gmv


def pick_max_sharpe(frontier_df: pd.DataFrame) -> OptimizationResult:
    """The frontier point with the best realized Sharpe ratio, among converged points."""
    converged = frontier_df[frontier_df["converged"]]
    if converged.empty:
        raise RuntimeError("No frontier point converged - cannot pick a max-Sharpe portfolio")
    best = converged.loc[converged["sharpe"].idxmax()]
    weights = frontier_df.attrs["weights_by_target"][best["target_return"]]
    return OptimizationResult(
        weights=weights,
        expected_return=best["realized_return"],
        volatility=best["volatility"],
        converged=True,
        message="picked from efficient frontier",
    )
