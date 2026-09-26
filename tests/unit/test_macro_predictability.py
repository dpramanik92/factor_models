import numpy as np
import pandas as pd

from factor_model.regression.macro_predictability import (
    macro_predictive_regression,
    monthly_factor_returns,
    monthly_macro_change,
    summarize_macro_predictability,
)


def test_monthly_factor_returns_compounds_within_month():
    dates = pd.to_datetime(["2024-01-05", "2024-01-15", "2024-02-05"])
    daily = pd.DataFrame({"beta": [0.01, -0.02, 0.03]}, index=dates)

    monthly = monthly_factor_returns(daily)

    expected_jan = 1.01 * 0.98 - 1.0
    assert np.isclose(monthly.loc[monthly.index[0], "beta"], expected_jan)
    assert np.isclose(monthly.loc[monthly.index[1], "beta"], 0.03)


def test_monthly_macro_change_aligns_with_monthly_factor_returns():
    # monthly_macro_change (resample-based) and monthly_factor_returns (to_period-based) must
    # produce directly alignable indices - a real bug (DatetimeIndex vs PeriodIndex mismatch
    # silently dropping every row on concat) was only caught by running this against real
    # fetched data, not synthetic PeriodIndex-only fixtures.
    daily_dates = pd.bdate_range("2024-01-02", "2024-04-30")
    rng = np.random.default_rng(3)
    prices = pd.Series(100 * (1 + rng.normal(0, 0.01, size=len(daily_dates))).cumprod(), index=daily_dates, name="macro")
    factor_daily = pd.DataFrame({"beta": rng.normal(0, 0.01, size=len(daily_dates))}, index=daily_dates)

    macro_monthly = monthly_macro_change(prices)
    factor_monthly = monthly_factor_returns(factor_daily)

    assert type(macro_monthly.index) is type(factor_monthly.index)
    aligned = pd.concat([factor_monthly["beta"], macro_monthly.shift(1)], axis=1).dropna()
    assert len(aligned) > 0


def test_macro_predictive_regression_recovers_known_slope():
    rng = np.random.default_rng(0)
    n = 60
    periods = pd.period_range("2018-01", periods=n, freq="M")
    macro = pd.Series(rng.normal(0, 0.02, size=n), index=periods, name="macro")
    true_slope = 0.5
    # factor_t depends on macro_{t-1}, plus noise - macro's own contemporaneous value is
    # irrelevant to the lagged regression, only its shift(1) is used.
    factor = pd.Series(np.nan, index=periods, dtype=float)
    factor.iloc[1:] = true_slope * macro.shift(1).iloc[1:].to_numpy() + rng.normal(0, 0.001, size=n - 1)
    factor_df = pd.DataFrame({"beta": factor})

    result = macro_predictive_regression(macro, factor_df, lag=1)

    assert len(result) == 1
    assert np.isclose(result.iloc[0]["Slope"], true_slope, atol=0.05)
    assert abs(result.iloc[0]["NW_t"]) > 2.0


def test_macro_predictive_regression_no_relationship_gives_small_slope():
    rng = np.random.default_rng(1)
    n = 60
    periods = pd.period_range("2018-01", periods=n, freq="M")
    macro = pd.Series(rng.normal(0, 0.02, size=n), index=periods, name="macro")
    factor_df = pd.DataFrame({"beta": rng.normal(0, 0.01, size=n)}, index=periods)

    result = macro_predictive_regression(macro, factor_df, lag=1)

    assert len(result) == 1
    assert abs(result.iloc[0]["Slope"]) < 0.5


def test_summarize_macro_predictability_bonferroni_is_stricter():
    rng = np.random.default_rng(2)
    n = 60
    periods = pd.period_range("2018-01", periods=n, freq="M")
    macros = {f"M{i}": pd.Series(rng.normal(0, 0.02, size=n), index=periods, name=f"M{i}") for i in range(3)}
    factor_df = pd.DataFrame({f"F{j}": rng.normal(0, 0.01, size=n) for j in range(5)}, index=periods)

    combined = summarize_macro_predictability(macros, factor_df, lag=1)

    assert len(combined) == 3 * 5
    assert combined["Significant_bonferroni"].sum() <= combined["Significant_uncorrected_p05"].sum()
