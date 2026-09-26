import numpy as np
import pandas as pd

from factor_model.factors.top100 import compute_top100_flag


def test_compute_top100_flag_ranks_correctly():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    symbols = [f"S{i}" for i in range(5)]
    # Market caps 10..50 - top 2 by cap should be flagged when n=2.
    market_cap = pd.DataFrame([[10.0, 20.0, 30.0, 40.0, 50.0]], index=dates, columns=symbols)

    result = compute_top100_flag(market_cap, n=2)
    result = result.set_index("Symbol")

    assert result.loc["S4", "top100_flag"] == 1  # largest
    assert result.loc["S3", "top100_flag"] == 1  # second largest
    assert result.loc["S2", "top100_flag"] == 0
    assert result.loc["S1", "top100_flag"] == 0
    assert result.loc["S0", "top100_flag"] == 0


def test_compute_top100_flag_nan_stays_nan_not_zero():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    symbols = ["A", "B", "C"]
    market_cap = pd.DataFrame([[10.0, np.nan, 30.0]], index=dates, columns=symbols)

    result = compute_top100_flag(market_cap, n=1)
    result = result.set_index("Symbol")

    assert pd.isna(result.loc["B", "top100_flag"])
    assert result.loc["C", "top100_flag"] == 1
    assert result.loc["A", "top100_flag"] == 0


def test_compute_top100_flag_per_date_independent():
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    symbols = ["A", "B"]
    # A is bigger on day 1, B is bigger on day 2 - the flag should flip accordingly.
    market_cap = pd.DataFrame([[100.0, 50.0], [50.0, 100.0]], index=dates, columns=symbols)

    result = compute_top100_flag(market_cap, n=1)
    day1 = result[result["Date"] == dates[0]].set_index("Symbol")
    day2 = result[result["Date"] == dates[1]].set_index("Symbol")

    assert day1.loc["A", "top100_flag"] == 1
    assert day1.loc["B", "top100_flag"] == 0
    assert day2.loc["A", "top100_flag"] == 0
    assert day2.loc["B", "top100_flag"] == 1
