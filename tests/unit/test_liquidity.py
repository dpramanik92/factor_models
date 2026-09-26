import numpy as np
import pandas as pd

from factor_model.factors import liquidity as liquidity_module
from factor_model.factors.liquidity import compute_liquidity


def _annual_ratios() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Symbol": ["AAA", "BBB"],
            "FiscalYearEnd": pd.to_datetime(["2023-03-31", "2023-03-31"]),
            "No. of Equity Shares": [10000.0, 100000.0],
        }
    )


def test_compute_liquidity_matches_manual_turnover(monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=260)  # well past FY23 + 60d reporting lag
    universe = ["AAA", "BBB"]
    volume = pd.DataFrame({"AAA": 1000.0, "BBB": 2000.0}, index=dates)
    monkeypatch.setattr(liquidity_module, "load_volume", lambda u: volume[u])

    result = compute_liquidity(universe, _annual_ratios(), dates, lookback_days=252)
    last_date = dates[-1]
    row_aaa = result[(result["Date"] == last_date) & (result["Symbol"] == "AAA")].iloc[0]
    row_bbb = result[(result["Date"] == last_date) & (result["Symbol"] == "BBB")].iloc[0]

    # Constant volume and constant (already-available) shares outstanding -> turnover is
    # constant over the trailing window -> liquidity = log(turnover) exactly.
    assert np.isclose(row_aaa["liquidity"], np.log(1000.0 / 10000.0))
    assert np.isclose(row_bbb["liquidity"], np.log(2000.0 / 100000.0))
    # AAA turns over 10%/day vs BBB's 2%/day -> AAA is more liquid.
    assert row_aaa["liquidity"] > row_bbb["liquidity"]


def test_compute_liquidity_nan_before_lookback_window_fills(monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=100)  # fewer than the 252-day lookback
    universe = ["AAA"]
    volume = pd.DataFrame({"AAA": 1000.0}, index=dates)
    monkeypatch.setattr(liquidity_module, "load_volume", lambda u: volume[u])

    result = compute_liquidity(universe, _annual_ratios(), dates, lookback_days=252)

    assert result["liquidity"].isna().all()


def test_compute_liquidity_forward_fills_unflagged_holiday(monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=260)
    universe = ["AAA", "BBB"]
    volume = pd.DataFrame({"AAA": 1000.0, "BBB": 2000.0}, index=dates)
    # An unflagged holiday: the entire cross-section shows zero volume on one date, right before
    # the last date - without forward-filling, this would NaN out the last ~lookback_days worth
    # of the rolling average, not just this one day.
    holiday = dates[-2]
    volume.loc[holiday] = 0.0
    monkeypatch.setattr(liquidity_module, "load_volume", lambda u: volume[u])

    result = compute_liquidity(universe, _annual_ratios(), dates, lookback_days=252)
    last_date = dates[-1]
    row_aaa = result[(result["Date"] == last_date) & (result["Symbol"] == "AAA")].iloc[0]

    # Forward-filled holiday -> turnover unaffected -> same result as the no-holiday case.
    assert np.isclose(row_aaa["liquidity"], np.log(1000.0 / 10000.0))


def test_compute_liquidity_nan_safe_for_zero_volume(monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=260)
    universe = ["AAA"]
    volume = pd.DataFrame({"AAA": 0.0}, index=dates)
    monkeypatch.setattr(liquidity_module, "load_volume", lambda u: volume[u])

    result = compute_liquidity(universe, _annual_ratios(), dates, lookback_days=252)

    # log(0) is undefined - must come out NaN, not -inf or an error.
    assert result["liquidity"].notna().sum() == 0
