import numpy as np
import pandas as pd

from factor_model.factors.size import add_size_decile_dummies, compute_size


def test_size_deciles_assigned_correctly():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    # 10 symbols with clearly increasing market cap -> one per decile, monotonic.
    symbols = [f"S{i}" for i in range(10)]
    market_cap = pd.DataFrame(
        [[10.0 * (i + 1) for i in range(10)]], index=dates, columns=symbols
    )

    result = compute_size(market_cap)
    result = result.sort_values("Symbol").reset_index(drop=True)

    # Smallest symbol (S0) should be decile 1, largest (S9) decile 10.
    assert result.loc[result["Symbol"] == "S0", "size_decile"].iloc[0] == 1
    assert result.loc[result["Symbol"] == "S9", "size_decile"].iloc[0] == 10

    # 'size' column is raw log(market_cap), not yet z-scored.
    expected_log = np.log(10.0)
    assert np.isclose(result.loc[result["Symbol"] == "S0", "size"].iloc[0], expected_log)


def test_size_decile_nan_when_fewer_than_ten_symbols():
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    symbols = [f"S{i}" for i in range(5)]
    market_cap = pd.DataFrame([[10.0 * (i + 1) for i in range(5)]], index=dates, columns=symbols)

    result = compute_size(market_cap)

    assert result["size_decile"].isna().all()


def test_add_size_decile_dummies_drops_d1_as_reference():
    panel = pd.DataFrame({"Symbol": [f"S{i}" for i in range(1, 11)], "size_decile": list(range(1, 11))})

    result = add_size_decile_dummies(panel)

    dummy_cols = [f"size_decile_{i}" for i in range(2, 11)]
    assert list(result.columns[-9:]) == dummy_cols
    # D1 gets all-zero dummies (the dropped reference category).
    row_d1 = result[result["size_decile"] == 1].iloc[0]
    assert (row_d1[dummy_cols] == 0.0).all()
    # D5 has exactly its own dummy set to 1, all others 0.
    row_d5 = result[result["size_decile"] == 5].iloc[0]
    assert row_d5["size_decile_5"] == 1.0
    assert (row_d5[[c for c in dummy_cols if c != "size_decile_5"]] == 0.0).all()


def test_add_size_decile_dummies_nan_stays_nan_not_zero():
    panel = pd.DataFrame({"Symbol": ["A", "B"], "size_decile": [3, np.nan]})

    result = add_size_decile_dummies(panel)

    row_b = result[result["Symbol"] == "B"].iloc[0]
    dummy_cols = [f"size_decile_{i}" for i in range(2, 11)]
    assert row_b[dummy_cols].isna().all()
