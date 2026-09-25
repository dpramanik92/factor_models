import numpy as np
import pandas as pd

from factor_model.regression.newey_west import newey_west_lag, summarize_factor_returns


def test_newey_west_lag_formula():
    assert newey_west_lag(100) == 4
    assert newey_west_lag(1) == 1  # never below 1


def test_summarize_factor_returns_recovers_mean_and_sign():
    rng = np.random.default_rng(0)
    n = 2000
    # Factor A has a clear positive mean, Factor B is centered at zero.
    series_a = pd.Series(rng.normal(loc=0.001, scale=0.01, size=n))
    series_b = pd.Series(rng.normal(loc=0.0, scale=0.01, size=n))
    factor_returns = pd.DataFrame({"A": series_a, "B": series_b})

    summary = summarize_factor_returns(factor_returns)

    row_a = summary[summary["Factor"] == "A"].iloc[0]
    row_b = summary[summary["Factor"] == "B"].iloc[0]

    assert row_a["Mean_coef"] > 0
    assert abs(row_a["NW_t"]) > 2  # clearly significant
    assert row_a["N_days"] == n
    assert 0.0 <= row_a["Pct_days_positive"] <= 1.0
    assert row_a["NW_SE"] > 0
    # B has no true mean, so it's not guaranteed significant, but the machinery must run cleanly.
    assert not pd.isna(row_b["NW_t"])


def test_summarize_factor_returns_handles_nans():
    series = pd.Series([0.01, np.nan, 0.02, 0.015, np.nan, 0.005])
    factor_returns = pd.DataFrame({"A": series})
    summary = summarize_factor_returns(factor_returns)
    assert summary.iloc[0]["N_days"] == 4
