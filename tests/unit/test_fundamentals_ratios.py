import numpy as np
import pandas as pd

from factor_model.factors.fundamentals_ratios import compute_annual_ratios
from factor_model.factors.known_corporate_actions import KNOWN_CORPORATE_ACTIONS, exclude_known_corporate_actions


def _synthetic_annual_long() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Symbol": ["AAA", "AAA", "AAA"],
            "FiscalYearEnd": pd.to_datetime(["2022-03-31", "2023-03-31", "2024-03-31"]),
            "Sales": [100.0, 120.0, 150.0],
            "Net profit": [10.0, 15.0, 18.0],
            "Total Assets": [500.0, 550.0, 600.0],
            "Equity Share Capital": [10.0, 10.0, 10.0],
            "Reserves": [90.0, 100.0, 115.0],
            "Borrowings": [50.0, 55.0, 60.0],
            "Net Block": [200.0, 220.0, 250.0],
            "Depreciation": [10.0, 11.0, 12.0],
        }
    )


def test_ratio_formulas():
    df = compute_annual_ratios(_synthetic_annual_long())
    row_2023 = df[df["FiscalYearEnd"] == "2023-03-31"].iloc[0]

    assert np.isclose(row_2023["Leverage"], 55.0 / (10.0 + 100.0))
    assert np.isclose(row_2023["ROA"], 15.0 / 550.0)
    assert np.isclose(row_2023["ROE"], 15.0 / 110.0)
    assert np.isclose(row_2023["NetMargin"], 15.0 / 120.0)
    assert np.isclose(row_2023["RevenueGrowth"], 120.0 / 100.0 - 1.0)
    assert np.isclose(row_2023["Capex"], (220.0 - 200.0 + 11.0) / 500.0)


def test_zero_denominator_produces_nan_not_error():
    df = _synthetic_annual_long()
    df.loc[df["FiscalYearEnd"] == "2023-03-31", ["Equity Share Capital", "Reserves"]] = 0.0
    result = compute_annual_ratios(df)
    row = result[result["FiscalYearEnd"] == "2023-03-31"].iloc[0]
    assert pd.isna(row["Leverage"])
    assert pd.isna(row["ROE"])


def test_non_annual_gap_excluded_from_growth_and_capex():
    df = _synthetic_annual_long()
    # Make the 2023 row a partial-year figure by shifting its date close to 2022's.
    df.loc[df["FiscalYearEnd"] == "2023-03-31", "FiscalYearEnd"] = pd.Timestamp("2022-09-30")
    result = compute_annual_ratios(df)
    partial_row = result[result["FiscalYearEnd"] == "2022-09-30"].iloc[0]
    assert pd.isna(partial_row["RevenueGrowth"])
    assert pd.isna(partial_row["Capex"])


def test_known_corporate_action_exclusion_applies_to_registered_entry():
    symbol, fy = next(iter(KNOWN_CORPORATE_ACTIONS.keys()))
    df = pd.DataFrame(
        {
            "Symbol": [symbol],
            "FiscalYearEnd": pd.to_datetime([fy]),
            "RevenueGrowth": [0.661],
            "Capex": [0.30],
        }
    )
    result = exclude_known_corporate_actions(df)
    assert pd.isna(result.iloc[0]["RevenueGrowth"])
    assert pd.isna(result.iloc[0]["Capex"])


def test_known_corporate_action_exclusion_leaves_other_rows_alone():
    df = pd.DataFrame(
        {
            "Symbol": ["UNRELATED"],
            "FiscalYearEnd": pd.to_datetime(["2024-03-31"]),
            "RevenueGrowth": [0.10],
            "Capex": [0.05],
        }
    )
    result = exclude_known_corporate_actions(df)
    assert result.iloc[0]["RevenueGrowth"] == 0.10
    assert result.iloc[0]["Capex"] == 0.05
