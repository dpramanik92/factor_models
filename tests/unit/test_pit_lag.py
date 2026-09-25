import pandas as pd

from factor_model.factors.pit import apply_reporting_lag, expand_to_daily


def test_apply_reporting_lag_shifts_forward():
    fy_ends = pd.Series(pd.to_datetime(["2023-03-31", "2024-03-31"]))
    shifted = apply_reporting_lag(fy_ends, lag_days=60)
    assert shifted.iloc[0] == pd.Timestamp("2023-05-30")
    assert shifted.iloc[1] == pd.Timestamp("2024-05-30")


def test_expand_to_daily_no_lookahead():
    annual = pd.Series(
        [0.10, 0.20],
        index=pd.to_datetime(["2023-03-31", "2024-03-31"]),
    )
    business_days = pd.date_range("2023-01-01", "2024-12-31", freq="B")
    daily = expand_to_daily(annual, business_days, lag_days=60)

    # Before the FY2023 value becomes available, must be NaN.
    assert pd.isna(daily.loc["2023-05-25"])
    # On/after FY2023 + 60 days, the FY2023 value holds.
    assert daily.loc["2023-05-30"] == 0.10
    assert daily.loc["2023-12-01"] == 0.10
    # The FY2024 value must not appear before its own lag clears.
    assert daily.loc["2024-05-29"] == 0.10
    assert daily.loc["2024-06-03"] == 0.20


def test_expand_to_daily_forward_fill_only_never_backfills():
    annual = pd.Series([0.50], index=pd.to_datetime(["2024-03-31"]))
    business_days = pd.date_range("2024-01-01", "2024-12-31", freq="B")
    daily = expand_to_daily(annual, business_days, lag_days=60)
    assert daily.loc["2024-01-15"] != 0.50
    assert pd.isna(daily.loc["2024-01-15"])


def test_expand_to_daily_empty_series():
    annual = pd.Series([], dtype=float)
    business_days = pd.date_range("2024-01-01", "2024-01-10", freq="B")
    daily = expand_to_daily(annual, business_days)
    assert daily.isna().all()
    assert len(daily) == len(business_days)
