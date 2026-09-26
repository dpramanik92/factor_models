import numpy as np
import pandas as pd
import pytest

from factor_model import config
from factor_model.portfolio.segmented_risk_model import (
    build_group_exposure,
    build_joint_exposures,
    build_joint_factor_covariance,
    build_joint_specific_risk,
    build_max_stock_weight,
    build_segmented_stock_covariance,
    get_latest_segment_exposures,
    joint_design_cols,
)
from factor_model.risk.ewma_covariance import compute_ewma_factor_covariance


def test_joint_design_cols_suffixes_each_column_per_segment():
    cols = joint_design_cols(["beta", "size"])
    assert cols == ["beta__top100", "size__top100", "beta__rest", "size__rest"]


def test_get_latest_segment_exposures_partitions_by_top100_flag():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    rows = []
    for d in dates:
        for symbol, flag in [("BIG1", 1.0), ("BIG2", 1.0), ("SML1", 0.0), ("SML2", 0.0)]:
            rows.append({"Date": d, "Symbol": symbol, "top100_flag": flag, "a": 1.0, "b": 2.0})
    panel = pd.DataFrame(rows)

    top100 = get_latest_segment_exposures(panel, "top100", design_cols=["a", "b"])
    rest = get_latest_segment_exposures(panel, "rest", design_cols=["a", "b"])

    assert sorted(top100.index) == ["BIG1", "BIG2"]
    assert sorted(rest.index) == ["SML1", "SML2"]
    # Only the latest date's rows are used.
    assert len(top100) == 2


def test_get_latest_segment_exposures_drops_incomplete_rows():
    dates = pd.to_datetime(["2024-01-01"])
    panel = pd.DataFrame(
        [
            {"Date": dates[0], "Symbol": "BIG1", "top100_flag": 1.0, "a": 1.0, "b": 2.0},
            {"Date": dates[0], "Symbol": "BIG2", "top100_flag": 1.0, "a": np.nan, "b": 2.0},
        ]
    )
    top100 = get_latest_segment_exposures(panel, "top100", design_cols=["a", "b"])
    assert list(top100.index) == ["BIG1"]


def test_build_joint_exposures_is_block_diagonal():
    design_cols = ["a", "b"]
    top100 = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=["BIG1", "BIG2"])
    rest = pd.DataFrame({"a": [5.0], "b": [6.0]}, index=["SML1"])

    joint = build_joint_exposures({"top100": top100, "rest": rest}, design_cols=design_cols)

    assert list(joint.columns) == ["a__top100", "b__top100", "a__rest", "b__rest"]
    # BIG1 loads only on the top100 block.
    assert joint.loc["BIG1", ["a__top100", "b__top100"]].tolist() == [1.0, 3.0]
    assert joint.loc["BIG1", ["a__rest", "b__rest"]].tolist() == [0.0, 0.0]
    # SML1 loads only on the rest block.
    assert joint.loc["SML1", ["a__rest", "b__rest"]].tolist() == [5.0, 6.0]
    assert joint.loc["SML1", ["a__top100", "b__top100"]].tolist() == [0.0, 0.0]


def test_build_joint_exposures_raises_on_symbol_in_both_segments():
    design_cols = ["a"]
    top100 = pd.DataFrame({"a": [1.0]}, index=["DUP"])
    rest = pd.DataFrame({"a": [2.0]}, index=["DUP"])
    with pytest.raises(ValueError):
        build_joint_exposures({"top100": top100, "rest": rest}, design_cols=design_cols)


def test_build_joint_factor_covariance_matches_standalone_within_block():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=200)
    top100_returns = pd.DataFrame({"a": rng.normal(scale=0.01, size=len(dates))}, index=dates)
    rest_returns = pd.DataFrame({"a": rng.normal(scale=0.01, size=len(dates))}, index=dates)
    top100_returns.index.name = "Date"
    rest_returns.index.name = "Date"

    joint_cov = build_joint_factor_covariance(
        {"top100": top100_returns, "rest": rest_returns}, design_cols=["a"], halflife=20
    )
    standalone_top100 = compute_ewma_factor_covariance(top100_returns, halflife=20)

    assert joint_cov.shape == (2, 2)
    assert list(joint_cov.columns) == ["a__top100", "a__rest"]
    np.testing.assert_allclose(joint_cov.loc["a__top100", "a__top100"], standalone_top100.loc["a", "a"])
    # Cross-segment term is genuinely estimated (not assumed zero) - just check it's finite here.
    assert np.isfinite(joint_cov.loc["a__top100", "a__rest"])


def test_build_joint_specific_risk_uses_current_segment_assignment_only():
    top100_spec = pd.DataFrame({"Symbol": ["BIG1", "FORMERLY_SMALL"], "specific_risk_annualized": [0.2, 0.9]})
    rest_spec = pd.DataFrame({"Symbol": ["SML1", "FORMERLY_SMALL"], "specific_risk_annualized": [0.3, 0.5]})
    symbols_by_segment = {"top100": ["BIG1", "FORMERLY_SMALL"], "rest": ["SML1"]}

    joint = build_joint_specific_risk({"top100": top100_spec, "rest": rest_spec}, symbols_by_segment)

    # FORMERLY_SMALL is currently top100 - only its top100-segment row (0.9) should survive.
    assert joint.loc[joint["Symbol"] == "FORMERLY_SMALL", "specific_risk_annualized"].tolist() == [0.9]
    assert sorted(joint["Symbol"]) == ["BIG1", "FORMERLY_SMALL", "SML1"]


def test_build_segmented_stock_covariance_matches_manual_calculation():
    cols = ["a__top100", "a__rest"]
    exposures = pd.DataFrame({"a__top100": [2.0, 0.0], "a__rest": [0.0, 3.0]}, index=["BIG1", "SML1"])
    F = pd.DataFrame([[0.0004, 0.0001], [0.0001, 0.0009]], index=cols, columns=cols)
    specific_risk = pd.DataFrame({"Symbol": ["BIG1", "SML1"], "specific_risk_annualized": [0.2, 0.3]})

    Sigma = build_segmented_stock_covariance(exposures, F, specific_risk)

    X = exposures.loc[["BIG1", "SML1"], cols].to_numpy()
    expected_factor_risk = X @ F.to_numpy() @ X.T
    daily_var = (np.array([0.2, 0.3]) ** 2) / 252
    expected = expected_factor_risk + np.diag(daily_var)

    np.testing.assert_allclose(Sigma.to_numpy(), expected)


def test_build_group_exposure_marks_only_named_segment():
    symbols_by_segment = {"top100": ["BIG1", "BIG2"], "rest": ["SML1"]}
    group = build_group_exposure(symbols_by_segment, "rest")
    assert group["SML1"] == 1.0
    assert group["BIG1"] == 0.0
    assert group["BIG2"] == 0.0


def test_build_max_stock_weight_returns_scalar_default_when_none():
    symbols_by_segment = {"top100": ["BIG1"], "rest": ["SML1"]}
    result = build_max_stock_weight(symbols_by_segment, None)
    assert result == config.MAX_STOCK_WEIGHT


def test_build_max_stock_weight_assigns_per_segment_cap():
    symbols_by_segment = {"top100": ["BIG1", "BIG2"], "rest": ["SML1"]}
    result = build_max_stock_weight(symbols_by_segment, {"top100": 0.10, "rest": 0.20})
    assert result["BIG1"] == 0.10
    assert result["BIG2"] == 0.10
    assert result["SML1"] == 0.20


def test_build_segmented_stock_covariance_drops_symbols_missing_specific_risk():
    cols = ["a__top100"]
    exposures = pd.DataFrame({"a__top100": [1.0, 2.0]}, index=["BIG1", "BIG2"])
    F = pd.DataFrame([[0.0004]], index=cols, columns=cols)
    specific_risk = pd.DataFrame({"Symbol": ["BIG1"], "specific_risk_annualized": [0.2]})

    Sigma = build_segmented_stock_covariance(exposures, F, specific_risk)

    assert list(Sigma.index) == ["BIG1"]
