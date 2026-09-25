import numpy as np
import pandas as pd

from factor_model.validation.statistical_tests import (
    test_residual_alpha_significance as compute_residual_alpha_significance,
)


def _synthetic_residuals(n_days: int, n_symbols: int, mean_shift: float, rng) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rows = []
    for d in dates:
        for i in range(n_symbols):
            rows.append({"Date": d, "Symbol": f"S{i}", "Residual": mean_shift + rng.normal(scale=0.01)})
    return pd.DataFrame(rows)


def test_alpha_significance_flags_clear_nonzero_mean():
    rng = np.random.default_rng(0)
    residuals = _synthetic_residuals(n_days=500, n_symbols=50, mean_shift=0.002, rng=rng)
    result = compute_residual_alpha_significance(residuals)
    assert result["verdict"] == "FAIL"
    assert "significant leftover alpha" in result["detail"]


def test_alpha_significance_passes_for_zero_mean_residuals():
    rng = np.random.default_rng(1)
    residuals = _synthetic_residuals(n_days=500, n_symbols=50, mean_shift=0.0, rng=rng)
    result = compute_residual_alpha_significance(residuals)
    assert result["verdict"] == "PASS"
    assert "no significant unexplained alpha" in result["detail"]
