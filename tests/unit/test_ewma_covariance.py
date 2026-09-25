import numpy as np
import pandas as pd

from factor_model.risk.ewma_covariance import compute_ewma_factor_covariance
from factor_model.risk.specific_risk import compute_ewma_specific_risk


def test_ewma_covariance_shape_and_symmetry():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=300)
    factor_returns = pd.DataFrame(
        rng.normal(scale=0.01, size=(len(dates), 3)), index=dates, columns=["F1", "F2", "F3"]
    )

    cov = compute_ewma_factor_covariance(factor_returns, halflife=30)

    assert cov.shape == (3, 3)
    np.testing.assert_allclose(cov.to_numpy(), cov.to_numpy().T, atol=1e-12)
    # Diagonal (variances) must be non-negative.
    assert (np.diag(cov.to_numpy()) >= 0).all()


def test_ewma_covariance_detects_correlation():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=300)
    base = rng.normal(scale=0.01, size=len(dates))
    # F2 is a noisy copy of F1 -> should show strong positive covariance/correlation.
    f1 = base
    f2 = base + rng.normal(scale=0.001, size=len(dates))
    f3 = rng.normal(scale=0.01, size=len(dates))  # independent
    factor_returns = pd.DataFrame({"F1": f1, "F2": f2, "F3": f3}, index=dates)

    cov = compute_ewma_factor_covariance(factor_returns, halflife=30)
    corr_12 = cov.loc["F1", "F2"] / np.sqrt(cov.loc["F1", "F1"] * cov.loc["F2", "F2"])
    corr_13 = cov.loc["F1", "F3"] / np.sqrt(cov.loc["F1", "F1"] * cov.loc["F3", "F3"])

    assert corr_12 > 0.9
    assert abs(corr_13) < 0.5


def test_ewma_specific_risk_returns_one_row_per_symbol():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=200)
    symbols = ["AAA", "BBB", "CCC"]
    rows = []
    for d in dates:
        for symbol in symbols:
            rows.append({"Date": d, "Symbol": symbol, "Residual": rng.normal(scale=0.02)})
    residuals_long = pd.DataFrame(rows)

    specific_risk = compute_ewma_specific_risk(residuals_long, halflife=30)

    assert set(specific_risk["Symbol"]) == set(symbols)
    assert (specific_risk["specific_risk_annualized"] > 0).all()
