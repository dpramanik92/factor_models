import numpy as np
import pandas as pd

from factor_model.factors import reversal as reversal_module
from factor_model.factors.reversal import REVERSAL_LOOKBACK_DAYS, compute_reversal


def test_compute_reversal_negates_recent_return(monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=REVERSAL_LOOKBACK_DAYS + 5)
    universe = ["AAA", "BBB"]
    # AAA rallied over the lookback window, BBB fell - reversal should flip the sign of each.
    aaa = np.linspace(100.0, 150.0, len(dates))
    bbb = np.linspace(100.0, 60.0, len(dates))
    prices = pd.DataFrame({"AAA": aaa, "BBB": bbb}, index=dates)
    monkeypatch.setattr(reversal_module, "load_prices", lambda u: prices[u])

    result = compute_reversal(universe)
    last_date = dates[-1]
    row_aaa = result[(result["Date"] == last_date) & (result["Symbol"] == "AAA")].iloc[0]
    row_bbb = result[(result["Date"] == last_date) & (result["Symbol"] == "BBB")].iloc[0]

    raw_return_aaa = aaa[-1] / aaa[-1 - REVERSAL_LOOKBACK_DAYS] - 1.0
    assert np.isclose(row_aaa["reversal"], -raw_return_aaa)
    # AAA (recent winner) gets a negative reversal exposure, BBB (recent loser) a positive one.
    assert row_aaa["reversal"] < 0
    assert row_bbb["reversal"] > 0


def test_compute_reversal_nan_before_lookback_window(monkeypatch):
    universe = ["AAA"]
    dates = pd.bdate_range("2024-01-01", periods=5)  # far fewer than the lookback window
    prices = pd.DataFrame({"AAA": [100.0] * len(dates)}, index=dates)
    monkeypatch.setattr(reversal_module, "load_prices", lambda u: prices[u])

    result = compute_reversal(universe)

    assert result["reversal"].isna().all()
