import numpy as np
import pandas as pd
import pytest

from factor_model.portfolio.analytics import backtest_performance_metrics, benchmark_relative_metrics, concentration_stats, risk_decomposition
from factor_model.portfolio.backtest import _month_end_rebalance_dates
from factor_model.portfolio.expected_returns import compute_significance_shrunk_expected_returns, project_to_stock_returns
from factor_model.portfolio.optimize import (
    minimize_variance,
    minimize_variance_long_short,
    trace_efficient_frontier,
    trace_efficient_frontier_long_short,
)
from factor_model.portfolio.risk_model import build_stock_covariance, get_latest_exposures
from factor_model.portfolio.timing_signals import (
    factor_price_index,
    macd_position,
    momentum_position,
    moving_average_position,
    test_timing_alpha as check_timing_alpha,
)
from factor_model.portfolio.validation import test_expected_return_vs_realized_stock_returns as check_expected_return_vs_realized
from factor_model.regression.cross_sectional import DESIGN_COLS


def test_expected_returns_zero_for_insignificant_factor():
    # No drift, pure noise -> some (not necessarily all) columns land under t_min by chance;
    # whichever do must get exactly zero expected return.
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=300)
    factor_returns = pd.DataFrame(
        {col: rng.normal(loc=0.0, scale=0.01, size=len(dates)) for col in DESIGN_COLS}, index=dates
    )

    expected, detail = compute_significance_shrunk_expected_returns(factor_returns, t_min=1.0, t_full=3.0)

    insignificant = detail[detail["NW_t"].abs() <= 1.0]
    assert not insignificant.empty  # sanity: pure noise across 20 columns should yield some
    assert np.allclose(expected.reindex(insignificant.index).to_numpy(), 0.0)


def test_expected_returns_full_weight_for_significant_factor():
    # Strong, consistent drift -> |NW_t| should land well above t_full -> shrinkage 1, expected
    # return equal to the EWMA-recent mean (not the flat full-sample mean).
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2024-01-01", periods=500)
    factor_returns = pd.DataFrame(
        {col: rng.normal(loc=0.002, scale=0.001, size=len(dates)) for col in DESIGN_COLS}, index=dates
    )

    expected, detail = compute_significance_shrunk_expected_returns(factor_returns, t_min=1.0, t_full=3.0, ewma_halflife=90)

    assert (detail["NW_t"].abs() > 3.0).all()
    ewma_mean = factor_returns.ewm(halflife=90, adjust=True).mean().iloc[-1]
    for col in DESIGN_COLS:
        assert np.isclose(expected[col], ewma_mean[col])
        # Sanity: with a strong, stable drift the EWMA mean should still be close to the true
        # generating mean, just not necessarily bit-identical to the flat sample mean.
        assert abs(expected[col] - 0.002) < 0.0005


def test_expected_returns_partial_shrinkage_is_between_zero_and_ewma_mean():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2024-01-01", periods=120)
    factor_returns = pd.DataFrame(
        {col: rng.normal(loc=0.0006, scale=0.01, size=len(dates)) for col in DESIGN_COLS}, index=dates
    )

    expected, detail = compute_significance_shrunk_expected_returns(factor_returns, t_min=0.0, t_full=100.0, ewma_halflife=90)

    # With t_min=0, shrinkage = |NW_t|/100, strictly between 0 and 1 for any finite, nonzero t.
    for col in DESIGN_COLS:
        ewma_mean_col = detail.loc[col, "EWMA_recent_mean"]
        assert 0.0 < abs(expected[col]) < abs(ewma_mean_col)


def test_project_to_stock_returns():
    exposures = pd.DataFrame(0.0, index=["AAA", "BBB"], columns=DESIGN_COLS)
    exposures.loc[:, "beta"] = 1.0
    exposures.loc["BBB", "value"] = 2.0
    mu_factor = pd.Series(0.0, index=DESIGN_COLS)
    mu_factor["beta"] = 0.001
    mu_factor["value"] = 0.0005

    mu_stock = project_to_stock_returns(mu_factor, exposures)

    assert np.isclose(mu_stock["AAA"], 0.001)
    assert np.isclose(mu_stock["BBB"], 0.001 + 2.0 * 0.0005)


def test_build_stock_covariance_matches_manual_decomposition():
    exposures = pd.DataFrame(0.0, index=["AAA", "BBB"], columns=DESIGN_COLS)
    exposures.loc["AAA", "beta"] = 1.0
    exposures.loc["BBB", "beta"] = 0.5

    factor_covariance = pd.DataFrame(0.0, index=DESIGN_COLS, columns=DESIGN_COLS)
    factor_covariance.loc["beta", "beta"] = 0.0004

    specific_risk = pd.DataFrame(
        {"Symbol": ["AAA", "BBB"], "specific_risk_annualized": [0.20, 0.30]}  # annualized std
    )

    Sigma = build_stock_covariance(exposures, factor_covariance, specific_risk)

    expected_aaa_var = 1.0**2 * 0.0004 + (0.20**2) / 252
    expected_bbb_var = 0.5**2 * 0.0004 + (0.30**2) / 252
    expected_cov = 1.0 * 0.5 * 0.0004

    assert np.isclose(Sigma.loc["AAA", "AAA"], expected_aaa_var)
    assert np.isclose(Sigma.loc["BBB", "BBB"], expected_bbb_var)
    assert np.isclose(Sigma.loc["AAA", "BBB"], expected_cov)


def test_get_latest_exposures_drops_incomplete_symbols():
    panel = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "Symbol": ["AAA", "BBB", "AAA", "BBB"],
            **{col: [1.0, 1.0, 1.0, np.nan] for col in DESIGN_COLS},
        }
    )
    result = get_latest_exposures(panel)
    assert list(result.index) == ["AAA"]


def test_gmv_matches_analytical_solution_for_uncorrelated_assets():
    # With zero correlation, GMV weight for asset i is proportional to 1/variance_i.
    symbols = ["AAA", "BBB", "CCC"]
    variances = [0.0001, 0.0004, 0.0009]
    Sigma = pd.DataFrame(np.diag(variances), index=symbols, columns=symbols)

    result = minimize_variance(Sigma)

    inv_var = 1.0 / np.array(variances)
    expected_weights = inv_var / inv_var.sum()

    np.testing.assert_allclose(result.weights.to_numpy(), expected_weights, atol=1e-4)
    assert result.converged
    assert np.isclose(result.weights.sum(), 1.0)
    assert (result.weights >= -1e-9).all()


def test_efficient_frontier_volatility_is_monotonic_in_target_return():
    rng = np.random.default_rng(0)
    n = 15
    symbols = [f"S{i}" for i in range(n)]
    A = rng.normal(size=(n, n))
    Sigma_np = A @ A.T * 1e-4 + np.eye(n) * 1e-5  # guaranteed PSD
    Sigma = pd.DataFrame(Sigma_np, index=symbols, columns=symbols)
    mu = pd.Series(rng.normal(loc=0.0005, scale=0.0003, size=n), index=symbols)

    frontier_df, gmv = trace_efficient_frontier(mu, Sigma, n_points=10)

    assert gmv.converged
    converged = frontier_df[frontier_df["converged"]].sort_values("target_return")
    assert len(converged) >= 8  # allow a couple of edge failures, not the whole frontier
    vol_diffs = converged["volatility"].diff().dropna()
    assert (vol_diffs >= -1e-6).all()


def test_minimize_variance_respects_max_stock_weight():
    n = 12
    symbols = [f"S{i}" for i in range(n)]
    # First asset has far lower variance - the uncapped GMV would concentrate heavily in it.
    variances = np.array([0.0001] + [0.01] * (n - 1))
    Sigma = pd.DataFrame(np.diag(variances), index=symbols, columns=symbols)

    result = minimize_variance(Sigma, max_stock_weight=0.10)

    assert result.converged
    assert result.weights.max() <= 0.10 + 1e-6
    assert np.isclose(result.weights.sum(), 1.0)


def test_minimize_variance_soft_cap_allows_but_discourages_excess():
    n = 12
    symbols = [f"S{i}" for i in range(n)]
    # First asset has far lower variance than the rest - GMV would want to concentrate heavily
    # in it, well past 10%, if left unconstrained.
    variances = np.array([0.0001] + [0.01] * (n - 1))
    Sigma = pd.DataFrame(np.diag(variances), index=symbols, columns=symbols)

    hard = minimize_variance(Sigma, max_stock_weight=0.10)
    soft = minimize_variance(Sigma, max_stock_weight=0.10, stock_weight_penalty=0.01)
    unconstrained = minimize_variance(Sigma)

    assert hard.converged and soft.converged and unconstrained.converged
    assert np.isclose(hard.weights.max(), 0.10, atol=1e-4)
    # Soft cap should sit strictly between the hard cap's 10% and the fully unconstrained
    # solution - discouraged, not forbidden, from exceeding the target.
    assert hard.weights.max() < soft.weights.max() < unconstrained.weights.max()
    assert np.isclose(soft.weights.sum(), 1.0)


def test_minimize_variance_group_soft_cap_discourages_group_concentration():
    # A "rest" group of 4 low-variance assets that GMV would want to load up on if unconstrained -
    # the group soft cap should pull total group weight down toward (not necessarily to) 20%,
    # without ever making it infeasible to exceed (soft, not hard).
    n = 12
    symbols = [f"S{i}" for i in range(n)]
    variances = np.array([0.0001] * 4 + [0.01] * (n - 4))
    Sigma = pd.DataFrame(np.diag(variances), index=symbols, columns=symbols)
    group = pd.Series(0.0, index=symbols)
    group[["S0", "S1", "S2", "S3"]] = 1.0

    unconstrained = minimize_variance(Sigma)
    capped = minimize_variance(Sigma, group_exposure=group, group_max_weight=0.20, group_weight_penalty=0.05)

    assert unconstrained.converged and capped.converged
    unconstrained_group_weight = float(unconstrained.weights[["S0", "S1", "S2", "S3"]].sum())
    capped_group_weight = float(capped.weights[["S0", "S1", "S2", "S3"]].sum())
    # Unconstrained GMV concentrates heavily in the low-variance group, well past 20%.
    assert unconstrained_group_weight > 0.20
    # The soft cap pulls it down but doesn't hard-clip it to exactly 20%.
    assert capped_group_weight < unconstrained_group_weight
    assert np.isclose(capped.weights.sum(), 1.0)


def test_minimize_variance_group_soft_cap_is_a_noop_when_penalty_none():
    n = 8
    symbols = [f"S{i}" for i in range(n)]
    variances = np.array([0.0001] * 2 + [0.01] * (n - 2))
    Sigma = pd.DataFrame(np.diag(variances), index=symbols, columns=symbols)
    group = pd.Series(0.0, index=symbols)
    group[["S0", "S1"]] = 1.0

    no_group = minimize_variance(Sigma)
    group_none_penalty = minimize_variance(Sigma, group_exposure=group, group_max_weight=0.20, group_weight_penalty=None)

    pd.testing.assert_series_equal(no_group.weights, group_none_penalty.weights, atol=1e-8)


def test_minimize_variance_per_symbol_max_stock_weight_series():
    # Two low-variance assets that would each want to concentrate heavily if uncapped - one is
    # given a tighter cap (10%) than the other (20%), via a Symbol-indexed Series instead of a
    # single float.
    n = 12
    symbols = [f"S{i}" for i in range(n)]
    variances = np.array([0.0001, 0.0001] + [0.01] * (n - 2))
    Sigma = pd.DataFrame(np.diag(variances), index=symbols, columns=symbols)
    caps = pd.Series(0.20, index=symbols)
    caps["S0"] = 0.10

    result = minimize_variance(Sigma, max_stock_weight=caps)

    assert result.converged
    assert result.weights["S0"] <= 0.10 + 1e-6
    assert result.weights["S1"] <= 0.20 + 1e-6
    # The looser-capped asset should be allowed to take more weight than the tighter-capped one,
    # since both have identical (low) variance and would otherwise be treated identically.
    assert result.weights["S1"] > result.weights["S0"]
    assert np.isclose(result.weights.sum(), 1.0)


def test_trace_efficient_frontier_accepts_per_symbol_max_stock_weight():
    n = 12
    symbols = [f"S{i}" for i in range(n)]
    rng = np.random.default_rng(4)
    variances = np.array([0.0001, 0.0001] + [0.01] * (n - 2))
    Sigma = pd.DataFrame(np.diag(variances), index=symbols, columns=symbols)
    mu = pd.Series(rng.normal(loc=0.0005, scale=0.0002, size=n), index=symbols)
    caps = pd.Series(0.20, index=symbols)
    caps["S0"] = 0.10

    frontier_df, gmv = trace_efficient_frontier(mu, Sigma, n_points=5, max_stock_weight=caps)

    assert gmv.converged
    assert gmv.weights["S0"] <= 0.10 + 1e-6
    assert frontier_df["converged"].any()


def test_minimize_variance_respects_beta_neutrality():
    n = 10
    symbols = [f"S{i}" for i in range(n)]
    variances = np.linspace(0.0001, 0.001, n)
    Sigma = pd.DataFrame(np.diag(variances), index=symbols, columns=symbols)
    beta_exposure = pd.DataFrame({"beta": [1.0] * 5 + [-1.0] * 5}, index=symbols)

    result = minimize_variance(Sigma, neutralize_exposures=beta_exposure)

    assert result.converged
    net_beta = float(result.weights.to_numpy() @ beta_exposure["beta"].to_numpy())
    assert abs(net_beta) < 1e-6
    assert np.isclose(result.weights.sum(), 1.0)
    assert (result.weights >= -1e-9).all()


def test_minimize_variance_long_short_respects_leverage_cap():
    n = 6
    symbols = [f"S{i}" for i in range(n)]
    rng = np.random.default_rng(7)
    A = rng.normal(size=(n, n))
    Sigma_np = A @ A.T * 1e-4 + np.eye(n) * 1e-5
    Sigma = pd.DataFrame(Sigma_np, index=symbols, columns=symbols)
    mu = pd.Series([0.002, 0.0015, 0.001, -0.001, -0.0015, -0.002], index=symbols)

    result = minimize_variance_long_short(Sigma, mu=mu, max_stock_weight=1.0, max_leverage=1.2)

    assert result.converged
    assert np.isclose(result.weights.sum(), 1.0)
    assert result.weights.abs().sum() <= 1.2 + 1e-6


def test_minimize_variance_long_short_uses_shorts_to_reach_high_target():
    n = 6
    symbols = [f"S{i}" for i in range(n)]
    Sigma = pd.DataFrame(np.eye(n) * 0.0001, index=symbols, columns=symbols)
    mu = pd.Series([0.002, 0.0015, 0.001, -0.001, -0.0015, -0.002], index=symbols)

    frontier_df, gmv = trace_efficient_frontier_long_short(mu, Sigma, n_points=5, max_leverage=1.2)

    assert gmv.converged
    top_target = frontier_df["target_return"].max()
    top_target_weights = frontier_df.attrs["weights_by_target"][top_target]
    # Reaching the top of the (leverage-expanded) frontier requires shorting a bad-mu name.
    assert (top_target_weights < -1e-6).any()
    assert top_target_weights.abs().sum() <= 1.2 + 1e-4


def test_minimize_variance_industry_cap():
    # 6 industries x 3 stocks, cap 20% each -> 120% total capacity, comfortably feasible at
    # sum-to-1 (unlike 4 industries x 20% = 80% max, which is infeasible outright).
    n = 18
    symbols = [f"S{i}" for i in range(n)]
    # First 3 stocks ("A") have far lower variance - GMV would want to overweight that industry.
    variances = np.array([0.0001, 0.0001, 0.0001] + [0.01] * (n - 3))
    Sigma = pd.DataFrame(np.diag(variances), index=symbols, columns=symbols)

    industries = ["A"] * 3 + ["B"] * 3 + ["C"] * 3 + ["D"] * 3 + ["E"] * 3 + ["F"] * 3
    industry_exposures = pd.get_dummies(pd.Series(industries, index=symbols)).astype(float)

    result = minimize_variance(Sigma, industry_exposures=industry_exposures, max_stock_weight=0.5, max_industry_weight=0.20)

    assert result.converged
    industry_weights = industry_exposures.T @ result.weights
    assert industry_weights.max() <= 0.20 + 1e-6
    assert np.isclose(result.weights.sum(), 1.0)


def test_concentration_stats_equal_weight_has_max_effective_n():
    n = 10
    weights = pd.Series(1.0 / n, index=[f"S{i}" for i in range(n)])
    stats = concentration_stats(weights)
    assert np.isclose(stats["effective_n_holdings"], n)
    assert np.isclose(stats["hhi"], 1.0 / n)


def test_concentration_stats_single_name_has_effective_n_one():
    weights = pd.Series({"AAA": 1.0, "BBB": 0.0, "CCC": 0.0})
    stats = concentration_stats(weights)
    assert np.isclose(stats["effective_n_holdings"], 1.0)
    assert np.isclose(stats["top1_weight"], 1.0)


def test_expected_return_vs_realized_positive_correlation_passes():
    rng = np.random.default_rng(0)
    symbols = [f"S{i}" for i in range(40)]
    signal = rng.normal(size=40)
    mu_stock = pd.Series(signal * 0.001, index=symbols)
    dates = pd.bdate_range("2023-01-01", periods=300)
    returns_wide = pd.DataFrame(
        {s: rng.normal(loc=signal[i] * 0.0008, scale=0.01, size=len(dates)) for i, s in enumerate(symbols)}, index=dates
    )

    result = check_expected_return_vs_realized(mu_stock, returns_wide, lookback_days=100)

    assert result["verdict"] in ("PASS", "WARN")


def test_expected_return_vs_realized_no_relationship_fails():
    rng = np.random.default_rng(1)
    symbols = [f"S{i}" for i in range(40)]
    mu_stock = pd.Series(rng.normal(size=40) * 0.001, index=symbols)
    dates = pd.bdate_range("2023-01-01", periods=300)
    returns_wide = pd.DataFrame({s: rng.normal(loc=0.0, scale=0.01, size=len(dates)) for s in symbols}, index=dates)

    result = check_expected_return_vs_realized(mu_stock, returns_wide, lookback_days=100)

    assert result["verdict"] == "FAIL"


def test_risk_decomposition_sums_to_total_variance():
    exposures = pd.DataFrame(0.0, index=["AAA", "BBB"], columns=DESIGN_COLS)
    exposures.loc["AAA", "beta"] = 1.0
    exposures.loc["BBB", "beta"] = 1.0

    factor_covariance = pd.DataFrame(0.0, index=DESIGN_COLS, columns=DESIGN_COLS)
    factor_covariance.loc["beta", "beta"] = 0.0004

    specific_risk = pd.DataFrame({"Symbol": ["AAA", "BBB"], "specific_risk_annualized": [0.2, 0.2]})

    weights = pd.Series({"AAA": 0.5, "BBB": 0.5})
    decomp = risk_decomposition(weights, exposures, factor_covariance, specific_risk)

    assert np.isclose(decomp["factor_variance_daily"] + decomp["specific_variance_daily"], decomp["total_variance_daily"])
    assert np.isclose(decomp["factor_pct"] + decomp["specific_pct"], 1.0)


def test_month_end_rebalance_dates_picks_last_trading_day_per_month():
    # Business days for Jan-Mar 2025, with Jan 31 (Fri) and Feb 28 (Fri) as the true month ends,
    # and Mar truncated early (Mar 20) to simulate "latest available data mid-month".
    index = pd.bdate_range("2025-01-02", "2025-01-31").append(pd.bdate_range("2025-02-03", "2025-02-28")).append(
        pd.bdate_range("2025-03-03", "2025-03-20")
    )

    dates = _month_end_rebalance_dates(index, "2025-01-01")

    assert dates == [pd.Timestamp("2025-01-31"), pd.Timestamp("2025-02-28"), pd.Timestamp("2025-03-20")]


def test_month_end_rebalance_dates_respects_start_filter():
    index = pd.bdate_range("2024-11-01", "2025-02-28")
    dates = _month_end_rebalance_dates(index, "2025-01-01")
    assert all(d >= pd.Timestamp("2025-01-01") for d in dates)
    assert dates[0].month == 1 and dates[0].year == 2025


def test_backtest_performance_metrics_matches_hand_computation():
    dates = pd.bdate_range("2025-01-01", periods=252)
    # Constant daily return of 0.03% -> exactly known annualized return/vol.
    r = pd.Series(0.0003, index=dates)
    return_series = {"Const": r}
    turnover = pd.DataFrame({"Date": [dates[0]], "Portfolio": ["Const"], "Turnover": [0.5]})

    metrics = backtest_performance_metrics(return_series, turnover)
    row = metrics.iloc[0]

    expected_total_return = 1.0003 ** 252 - 1.0
    assert np.isclose(row["Total_return"], expected_total_return)
    assert np.isclose(row["Annualized_vol"], 0.0)  # zero variance, constant returns
    assert row["Max_drawdown"] == 0.0  # monotonically increasing, no drawdown
    assert np.isclose(row["Avg_monthly_turnover"], 0.5)


def test_benchmark_relative_metrics_identical_series_is_beta_one_zero_alpha():
    dates = pd.bdate_range("2025-01-01", periods=252)
    rng = np.random.default_rng(3)
    bench = pd.Series(rng.normal(0.0004, 0.01, size=len(dates)), index=dates)
    return_series = {"Clone": bench.copy()}

    metrics = benchmark_relative_metrics(return_series, bench)
    row = metrics.iloc[0]

    assert np.isclose(row["Beta"], 1.0, atol=1e-9)
    assert np.isclose(row["Alpha_annualized"], 0.0, atol=1e-9)
    assert np.isclose(row["Tracking_error_annualized"], 0.0, atol=1e-9)
    assert np.isclose(row["Correlation"], 1.0, atol=1e-9)
    assert np.isclose(row["Up_capture"], 1.0, atol=1e-9)
    assert np.isclose(row["Down_capture"], 1.0, atol=1e-9)


def test_benchmark_relative_metrics_double_beta_leveraged_series():
    dates = pd.bdate_range("2025-01-01", periods=252)
    rng = np.random.default_rng(4)
    bench = pd.Series(rng.normal(0.0004, 0.01, size=len(dates)), index=dates)
    levered = bench * 2.0  # a pure 2x-leveraged clone: beta should be ~2

    metrics = benchmark_relative_metrics({"2x": levered}, bench)
    row = metrics.iloc[0]

    assert np.isclose(row["Beta"], 2.0, atol=1e-9)
    assert row["Up_capture"] > 1.5
    assert row["Down_capture"] > 1.5


def test_moving_average_position_matches_price_vs_ma():
    # Flat at 1.0 for 30 days (MA warms up at exactly 1.0), then a sustained jump up, then a
    # sustained drop below the still-adjusting MA.
    flat = pd.Series(1.0, index=pd.bdate_range("2024-01-01", periods=30))
    up = pd.Series(2.0, index=pd.bdate_range(flat.index[-1] + pd.tseries.offsets.BDay(1), periods=40))
    price = pd.concat([flat, up])

    position = moving_average_position(price, window=30)

    assert position.iloc[30:].eq(1.0).all()  # price (2.0) is above its rising MA once above 1.0 and steady


def test_macd_position_is_positive_in_a_sustained_uptrend():
    price = pd.Series(np.linspace(1.0, 2.0, 150), index=pd.bdate_range("2024-01-01", periods=150))
    position = macd_position(price)
    # After the initial EMA warm-up, a sustained uptrend should leave MACD above its signal line.
    assert position.iloc[-1] == 1.0


def test_timing_alpha_is_zero_excess_when_always_long():
    dates = pd.bdate_range("2024-01-01", periods=100)
    rng = np.random.default_rng(5)
    factor_returns = pd.Series(rng.normal(0.0003, 0.01, size=len(dates)), index=dates)
    always_long = pd.Series(1.0, index=dates)

    result = check_timing_alpha(factor_returns, always_long, "always-long control")

    assert np.isclose(result["excess_mean_daily"], 0.0, atol=1e-12)


def test_momentum_position_matches_sign_of_trailing_return():
    # Flat at 1.0 for the lookback+skip warm-up, then a clean, sustained uptrend.
    warmup = pd.Series(1.0, index=pd.bdate_range("2024-01-01", periods=30))
    uptrend = pd.Series(
        np.linspace(1.0, 2.0, 50), index=pd.bdate_range(warmup.index[-1] + pd.tseries.offsets.BDay(1), periods=50)
    )
    price = pd.concat([warmup, uptrend])

    position = momentum_position(price, lookback=20, skip=5)

    # Once the uptrend has been running long enough to be visible through the 5-day skip and
    # 20-day lookback window, the trailing return is unambiguously positive.
    assert position.iloc[-1] == 1.0


def test_momentum_position_degenerate_when_trend_never_reverses():
    # A monotonic price series never has a negative trailing return at any lookback/skip -
    # momentum_position should be +1 every single day once past the warm-up (the real-data
    # analogue: the size factor's 252-day-ex-1-month momentum signal was +1/-1 on 0% of days
    # across the whole sample, degenerating to a constant position).
    price = pd.Series(np.linspace(1.0, 2.0, 300), index=pd.bdate_range("2024-01-01", periods=300))
    position = momentum_position(price, lookback=252, skip=21)
    valid = position.dropna()
    assert (valid == 1.0).all()


def test_factor_price_index_starts_at_one_and_compounds():
    dates = pd.bdate_range("2024-01-01", periods=3)
    factor_returns = pd.Series([0.01, -0.02, 0.03], index=dates)
    price = factor_price_index(factor_returns)
    assert np.isclose(price.iloc[0], 1.01)
    assert np.isclose(price.iloc[-1], 1.01 * 0.98 * 1.03)
