import numpy as np
import pandas as pd

from factor_model.factors.size import compute_size


def test_size_quintiles_assigned_correctly(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    # 10 symbols with clearly increasing market cap -> quintiles should be monotonic.
    symbols = [f"S{i}" for i in range(10)]
    market_cap = pd.DataFrame(
        [[10.0 * (i + 1) for i in range(10)]], index=dates, columns=symbols
    )

    monkeypatch.setattr("factor_model.factors.size.load_market_cap", lambda universe: market_cap)

    result = compute_size(symbols)
    result = result.sort_values("Symbol").reset_index(drop=True)

    # Smallest two symbols (S0, S1) should be in quintile 1, largest two (S8, S9) in quintile 5.
    assert result.loc[result["Symbol"] == "S0", "size_quintile"].iloc[0] == 1
    assert result.loc[result["Symbol"] == "S9", "size_quintile"].iloc[0] == 5

    # 'size' column is raw log(market_cap), not yet z-scored.
    expected_log = np.log(10.0)
    assert np.isclose(result.loc[result["Symbol"] == "S0", "size"].iloc[0], expected_log)
