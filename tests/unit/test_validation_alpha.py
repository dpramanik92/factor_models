import numpy as np
import pandas as pd

from factor_model.validation.statistical_tests import (
    test_residual_alpha_by_cap_segment as compute_residual_alpha_by_cap_segment,
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


def _segmented_residuals_and_panel(n_days: int, n_symbols: int, top100_shift: float, rest_shift: float, rng):
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    residual_rows, panel_rows = [], []
    for i in range(n_symbols):
        is_top100 = 1.0 if i < n_symbols // 2 else 0.0
        shift = top100_shift if is_top100 else rest_shift
        for d in dates:
            residual_rows.append({"Date": d, "Symbol": f"S{i}", "Residual": shift + rng.normal(scale=0.01)})
            panel_rows.append({"Date": d, "Symbol": f"S{i}", "top100_flag": is_top100})
    return pd.DataFrame(residual_rows), pd.DataFrame(panel_rows)


def test_alpha_by_cap_segment_catches_offsetting_biases_pooled_check_would_miss():
    rng = np.random.default_rng(2)
    residuals, panel = _segmented_residuals_and_panel(
        n_days=500, n_symbols=60, top100_shift=0.002, rest_shift=-0.002, rng=rng
    )
    # The pooled check should see roughly zero (offsetting biases) and PASS...
    pooled = compute_residual_alpha_significance(residuals)
    assert pooled["verdict"] == "PASS"
    # ...but the segmented check must still catch both real, opposite-signed biases.
    segmented = compute_residual_alpha_by_cap_segment(residuals, panel)
    assert segmented["verdict"] == "FAIL"
    assert "top100" in segmented["detail"] and "rest" in segmented["detail"]


def test_alpha_by_cap_segment_passes_when_neither_segment_biased():
    rng = np.random.default_rng(3)
    residuals, panel = _segmented_residuals_and_panel(
        n_days=500, n_symbols=60, top100_shift=0.0, rest_shift=0.0, rng=rng
    )
    result = compute_residual_alpha_by_cap_segment(residuals, panel)
    assert result["verdict"] == "PASS"
