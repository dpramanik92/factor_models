"""Synthetic recovery test for the WLS cross-sectional regression engine
(.claude/skills/statistics/SKILL.md): simulate returns from known exposures
and known factor returns plus noise, fit, and assert recovered coefficients
are close to the truth - independent of any real-data quirks.
"""

import numpy as np
import pandas as pd

from factor_model.regression.cross_sectional import DESIGN_COLS, run_cross_sectional_regression


def test_wls_regression_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n_symbols = 60
    n_dates = 40
    symbols = [f"S{i}" for i in range(n_symbols)]
    dates = pd.bdate_range("2024-01-01", periods=n_dates + 1)

    true_coef = rng.normal(scale=0.001, size=len(DESIGN_COLS))

    exposure_rows = []
    return_rows = {}
    for i, d in enumerate(dates[:-1]):
        X = rng.normal(size=(n_symbols, len(DESIGN_COLS)))
        market_cap = rng.uniform(1e9, 1e11, size=n_symbols)
        noise = rng.normal(scale=0.0005, size=n_symbols)
        y = X @ true_coef + noise

        for j, symbol in enumerate(symbols):
            row = {"Date": d, "Symbol": symbol, "market_cap": market_cap[j]}
            row.update(dict(zip(DESIGN_COLS, X[j])))
            exposure_rows.append(row)

        return_rows[dates[i + 1]] = pd.Series(y, index=symbols)

    exposure_panel = pd.DataFrame(exposure_rows)
    returns_wide = pd.DataFrame(return_rows).T
    returns_wide.index.name = "Date"

    result = run_cross_sectional_regression(exposure_panel, returns_wide, min_n=10, rf_annual=0.0)

    recovered_mean = result.factor_returns[DESIGN_COLS].mean().to_numpy()
    np.testing.assert_allclose(recovered_mean, true_coef, atol=5e-4)
    assert len(result.factor_returns) == n_dates - 1
    assert not result.skipped_low_n_dates


def test_design_cols_parameter_restricts_design_matrix():
    # A segment-specific regression (pipeline.run_segment) passes a smaller design_cols list
    # (dropping top100_flag/interactions, constant within a single segment) - verify the
    # regression actually uses that list, not the module-level DESIGN_COLS default.
    rng = np.random.default_rng(1)
    n_symbols = 40
    n_dates = 30
    symbols = [f"S{i}" for i in range(n_symbols)]
    dates = pd.bdate_range("2024-01-01", periods=n_dates + 1)
    custom_cols = ["beta", "size", "psu_flag"]
    true_coef = rng.normal(scale=0.001, size=len(custom_cols))

    exposure_rows = []
    return_rows = {}
    for i, d in enumerate(dates[:-1]):
        X = rng.normal(size=(n_symbols, len(custom_cols)))
        market_cap = rng.uniform(1e9, 1e11, size=n_symbols)
        noise = rng.normal(scale=0.0005, size=n_symbols)
        y = X @ true_coef + noise

        for j, symbol in enumerate(symbols):
            row = {"Date": d, "Symbol": symbol, "market_cap": market_cap[j]}
            row.update(dict(zip(custom_cols, X[j])))
            exposure_rows.append(row)

        return_rows[dates[i + 1]] = pd.Series(y, index=symbols)

    exposure_panel = pd.DataFrame(exposure_rows)
    returns_wide = pd.DataFrame(return_rows).T
    returns_wide.index.name = "Date"

    result = run_cross_sectional_regression(
        exposure_panel, returns_wide, min_n=10, rf_annual=0.0, design_cols=custom_cols
    )

    assert list(result.factor_returns.columns) == custom_cols
    recovered_mean = result.factor_returns[custom_cols].mean().to_numpy()
    np.testing.assert_allclose(recovered_mean, true_coef, atol=5e-4)


def test_low_n_dates_are_skipped_while_others_still_fit():
    # date_to_target is built purely from exposure_panel's own unique dates: with K unique
    # exposure dates you get K-1 mappings (exposure[i] -> target dates[i+1]), and the *last*
    # unique exposure date is always left unmapped. So to get both
    # exposure[dates[0]] -> target dates[1] and exposure[dates[1]] -> target dates[2] mapped,
    # a 3rd exposure date (dates[2], content irrelevant - it stays unmapped itself) must exist
    # purely so dates[1] isn't the last one anymore.
    dates = pd.bdate_range("2024-01-01", periods=3)
    symbols = [f"S{i}" for i in range(50)]
    rng = np.random.default_rng(1)
    min_n = 20

    exposure_rows = []
    for exposure_date in (dates[0], dates[2]):
        # Full coverage for all 50 symbols.
        for symbol in symbols:
            row = {"Date": exposure_date, "Symbol": symbol, "market_cap": 1e9}
            row.update({col: rng.normal() for col in DESIGN_COLS})
            exposure_rows.append(row)
    # Exposure at dates[1]: only 5 of 50 symbols have valid data -> target dates[2] should be skipped.
    for i, symbol in enumerate(symbols):
        row = {"Date": dates[1], "Symbol": symbol, "market_cap": 1e9}
        if i < 5:
            row.update({col: rng.normal() for col in DESIGN_COLS})
        else:
            row.update({col: np.nan for col in DESIGN_COLS})
        exposure_rows.append(row)
    exposure_panel = pd.DataFrame(exposure_rows)

    returns_wide = pd.DataFrame(
        rng.normal(scale=0.01, size=(len(dates), len(symbols))), index=dates, columns=symbols
    )
    returns_wide.index.name = "Date"

    result = run_cross_sectional_regression(exposure_panel, returns_wide, min_n=min_n)

    assert dates[1] in result.factor_returns.index  # from exposure[0], N=50
    assert dates[2] in result.skipped_low_n_dates  # from exposure[1], N=5


def test_corporate_action_observations_are_flagged_and_excluded():
    rng = np.random.default_rng(2)
    n_symbols = 60
    dates = pd.bdate_range("2024-01-01", periods=3)
    symbols = [f"S{i}" for i in range(n_symbols)]

    exposure_rows = []
    for d in dates[:-1]:
        for symbol in symbols:
            row = {"Date": d, "Symbol": symbol, "market_cap": 1e9}
            row.update({col: rng.normal(scale=0.01) for col in DESIGN_COLS})
            exposure_rows.append(row)
    exposure_panel = pd.DataFrame(exposure_rows)

    returns = rng.normal(scale=0.01, size=(len(dates), n_symbols))
    returns[1, 0] = 0.90  # a clear corporate-action-sized single-day move for S0
    returns_wide = pd.DataFrame(returns, index=dates, columns=symbols)
    returns_wide.index.name = "Date"

    result = run_cross_sectional_regression(exposure_panel, returns_wide, min_n=10)

    assert len(result.flagged_corporate_actions) == 1
    assert result.flagged_corporate_actions.iloc[0]["Symbol"] == "S0"
    assert "S0" not in result.residuals_long.loc[
        result.residuals_long["Date"] == dates[1], "Symbol"
    ].to_numpy()


def test_degenerate_zero_variance_day_is_skipped_not_fit():
    # Regression tests a real bug: a holiday day where every stock has an identical (often
    # exactly 0.0) return produces a zero/near-zero centered TSS, which blows up R2 to -inf or
    # an extreme negative number rather than a genuine fit. Such a day should be skipped, not
    # regressed - see config.DEGENERATE_RETURN_STD_THRESHOLD. Needs 3 exposure dates for the
    # same reason as test_low_n_dates_are_skipped_while_others_still_fit: exposure[i] only gets
    # a target while a later exposure date exists.
    rng = np.random.default_rng(3)
    n_symbols = 60
    dates = pd.bdate_range("2024-01-01", periods=4)
    symbols = [f"S{i}" for i in range(n_symbols)]

    exposure_rows = []
    for d in (dates[0], dates[1], dates[2]):  # 3 exposure dates needed - see comment above
        for symbol in symbols:
            row = {"Date": d, "Symbol": symbol, "market_cap": 1e9}
            row.update({col: rng.normal(scale=0.01) for col in DESIGN_COLS})
            exposure_rows.append(row)
    exposure_panel = pd.DataFrame(exposure_rows)

    returns = rng.normal(scale=0.01, size=(len(dates), n_symbols))
    returns[2, :] = 0.0  # every stock identical (a stale/holiday day) for the dates[2] target
    returns_wide = pd.DataFrame(returns, index=dates, columns=symbols)
    returns_wide.index.name = "Date"

    result = run_cross_sectional_regression(exposure_panel, returns_wide, min_n=10)

    assert dates[1] in result.factor_returns.index  # from exposure[dates[0]], normal returns
    assert dates[2] in result.skipped_degenerate_dates  # from exposure[dates[1]], all-zero returns
    assert dates[2] not in result.factor_returns.index
    assert not (result.r2_daily["R2"].isin([float("-inf"), float("inf")])).any()
