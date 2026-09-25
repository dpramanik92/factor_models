import pandas as pd

from factor_model.factors.industry import expand_industry_exposure_daily


def test_expand_industry_exposure_daily_broadcasts_static_weights():
    industry_by_symbol = pd.DataFrame(
        {"Tech": [1.0, 0.5], "Finance": [0.0, 0.5]}, index=["AAA", "BBB"]
    )
    business_days = pd.date_range("2024-01-01", periods=3, freq="B")

    long = expand_industry_exposure_daily(industry_by_symbol, business_days)

    assert len(long) == len(business_days) * 2
    assert set(long.columns) == {"Date", "Symbol", "Tech", "Finance"}

    row = long[(long["Date"] == business_days[1]) & (long["Symbol"] == "BBB")].iloc[0]
    assert row["Tech"] == 0.5
    assert row["Finance"] == 0.5

    # Weights are static across every date for a given symbol.
    aaa_tech = long[long["Symbol"] == "AAA"]["Tech"]
    assert (aaa_tech == 1.0).all()
