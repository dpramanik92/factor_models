import numpy as np
import pandas as pd

from factor_model import config
from factor_model.portfolio.macro_timing import compute_macro_timing_tilt


def _synthetic_setup(n_months: int = 30, seed: int = 0):
    """A synthetic universe where the Oil group's daily return is deliberately built to have a
    real, negative, lagged relationship with oil-price momentum - the same shape the real
    alpha-research finding has (rising oil momentum -> Oil group underperforms next month) - so
    the regression has something genuine to detect rather than pure noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_months * 21)

    oil_symbols = config.OIL_GROUP_SYMBOLS[:3]
    it_symbols = config.IT_EXPORTER_SYMBOLS[:3]
    rest_symbols = [f"REST{i}" for i in range(10)]
    universe = oil_symbols + it_symbols + rest_symbols

    # A slow, persistent oil-price random walk with real month-to-month trend, so momentum is
    # meaningfully non-zero and autocorrelated (not white noise, which would have no momentum).
    oil_daily_ret = rng.normal(loc=0.0003, scale=0.01, size=len(dates))
    oil_price = pd.Series(100.0 * (1.0 + oil_daily_ret).cumprod(), index=dates)

    usdinr_daily_ret = rng.normal(loc=0.0001, scale=0.003, size=len(dates))
    usdinr = pd.Series(80.0 * (1.0 + usdinr_daily_ret).cumprod(), index=dates)

    returns = pd.DataFrame(rng.normal(scale=0.01, size=(len(dates), len(universe))), index=dates, columns=universe)

    # Inject the real relationship: Oil-group return this month tracks NEGATIVE of oil price
    # momentum over the trailing 3 months (lagged), on top of the baseline noise.
    oil_price_monthly = oil_price.resample("ME").last()
    oil_momentum_by_month = (oil_price_monthly.pct_change().add(1.0)).rolling(3).apply(lambda x: x.prod() - 1.0, raw=True)
    for i, date in enumerate(dates):
        month_period = date.to_period("M") - 1  # lagged by one month
        if month_period in oil_momentum_by_month.index.to_period("M"):
            m = oil_momentum_by_month[oil_momentum_by_month.index.to_period("M") == month_period]
            if not m.empty and pd.notna(m.iloc[0]):
                returns.loc[date, oil_symbols] += -0.5 * m.iloc[0] / 21

    returns.index.name = "Date"
    return returns, universe, oil_price, usdinr, oil_symbols, it_symbols


def test_compute_macro_timing_tilt_nonzero_with_enough_history():
    returns, universe, oil_price, usdinr, oil_symbols, it_symbols = _synthetic_setup(n_months=30)

    tilt = compute_macro_timing_tilt(returns, universe, oil_price, usdinr)

    assert isinstance(tilt, pd.Series)
    assert set(tilt.index) == set(universe)
    # The injected relationship should be picked up as a genuine, nonzero Oil-group tilt.
    assert (tilt.loc[oil_symbols] != 0.0).any()
    # A stock outside both groups gets no tilt.
    assert (tilt.loc[[s for s in universe if s not in oil_symbols and s not in it_symbols]] == 0.0).all()


def test_compute_macro_timing_tilt_zero_with_insufficient_history():
    # Fewer than config.MACRO_TIMING_MIN_MONTHS (24) months of overlapping data - not enough for
    # macro_predictive_regression to fit anything, so every tilt should fall back to 0.0.
    returns, universe, oil_price, usdinr, oil_symbols, it_symbols = _synthetic_setup(n_months=10)

    tilt = compute_macro_timing_tilt(returns, universe, oil_price, usdinr)

    assert (tilt == 0.0).all()


def test_compute_macro_timing_tilt_as_of_truncates_history():
    returns, universe, oil_price, usdinr, oil_symbols, it_symbols = _synthetic_setup(n_months=30)

    early_cutoff = returns.index[10 * 21]  # only ~10 months in - not enough history yet
    tilt_early = compute_macro_timing_tilt(returns, universe, oil_price, usdinr, as_of=early_cutoff)

    assert (tilt_early == 0.0).all()


def test_compute_macro_timing_tilt_symbols_outside_universe_ignored():
    returns, universe, oil_price, usdinr, oil_symbols, it_symbols = _synthetic_setup(n_months=30)
    small_universe = oil_symbols + it_symbols  # drop the REST symbols entirely

    tilt = compute_macro_timing_tilt(returns, small_universe, oil_price, usdinr)

    assert set(tilt.index) == set(small_universe)
