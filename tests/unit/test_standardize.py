import numpy as np
import pandas as pd

from factor_model.standardize import cross_sectional_winsorize_zscore, winsorize, winsorize_zscore, zscore


def test_zscore_basic():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = zscore(s)
    assert abs(z.mean()) < 1e-9
    assert abs(z.std(ddof=1) - 1.0) < 1e-9


def test_zscore_nan_safe():
    s = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
    z = zscore(s)
    assert z.isna().sum() == 2
    assert not z.dropna().isna().any()


def test_zscore_zero_std_returns_nan():
    s = pd.Series([5.0, 5.0, 5.0])
    z = zscore(s)
    assert z.isna().all()


def test_winsorize_percentile_clips_outliers():
    s = pd.Series(list(range(1, 101)) + [10000.0])
    w = winsorize(s, method="percentile", bounds=(0.01, 0.99))
    assert w.max() < 10000.0


def test_winsorize_preserves_nan():
    s = pd.Series([1.0, np.nan, 3.0])
    w = winsorize(s)
    assert w.isna().sum() == 1


def test_winsorize_zscore_order_matters():
    # A single huge outlier inflates the std used for plain z-scoring, which crushes the
    # z-scores of the other 5 points toward zero. Winsorizing first removes the outlier's
    # influence on that scale estimate, so the other 5 points keep a meaningful spread.
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0])
    wz = winsorize_zscore(s, method="percentile", bounds=(0.0, 0.8))
    plain_z = zscore(s)
    assert wz.iloc[:5].std() > plain_z.iloc[:5].std()


def test_cross_sectional_winsorize_zscore_per_date():
    panel = pd.DataFrame(
        {
            "Date": ["d1", "d1", "d1", "d2", "d2", "d2"],
            "Symbol": ["A", "B", "C", "A", "B", "C"],
            "value": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )
    z = cross_sectional_winsorize_zscore(panel, "Date", "value")
    # Each date group is standardized independently, so both groups produce the same
    # relative z-score pattern despite very different raw scales.
    d1 = z[panel["Date"] == "d1"].to_numpy()
    d2 = z[panel["Date"] == "d2"].to_numpy()
    np.testing.assert_allclose(d1, d2, atol=1e-9)
