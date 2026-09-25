from pathlib import Path

import pandas as pd
import pytest

from factor_model.io.fundamentals import (
    _find_all_label_rows,
    _find_label_row,
    _find_section_bounds,
    parse_fundamentals_file,
)


def _build_synthetic_data_sheet() -> pd.DataFrame:
    """A minimal synthetic Screener.in Data Sheet, mirroring the real row/label structure
    but with fewer years and no bank-style blank rows, for a fast unit test.
    """
    rows: list[list] = [[None] * 5 for _ in range(30)]

    def set_row(i, label, *values):
        rows[i][0] = label
        for j, v in enumerate(values):
            rows[i][1 + j] = v

    set_row(0, "COMPANY NAME", "TEST COMPANY LTD")
    set_row(4, "META")
    set_row(8, "Market Capitalization", 1000.0)

    set_row(10, "PROFIT & LOSS")
    set_row(11, "Report Date", pd.Timestamp("2022-03-31"), pd.Timestamp("2023-03-31"))
    set_row(12, "Sales", 100.0, 120.0)
    set_row(13, "Other Income", 5.0, 6.0)
    set_row(14, "Depreciation", 2.0, 2.5)
    set_row(15, "Interest", 1.0, 1.2)
    set_row(16, "Profit before tax", 20.0, 25.0)
    set_row(17, "Tax", 5.0, 6.0)
    set_row(18, "Net profit", 15.0, 19.0)
    set_row(19, "Dividend Amount", 3.0, 4.0)

    set_row(20, "BALANCE SHEET")
    set_row(21, "Report Date", pd.Timestamp("2022-03-31"), pd.Timestamp("2023-03-31"))
    set_row(22, "Equity Share Capital", 10.0, 10.0)
    set_row(23, "Reserves", 50.0, 65.0)
    set_row(24, "Borrowings", 20.0, 22.0)
    set_row(25, "Other Liabilities", 10.0, 11.0)
    set_row(26, "Total", 90.0, 108.0)  # liabilities + equity side
    set_row(27, "Net Block", 40.0, 45.0)
    set_row(28, "Total", 90.0, 108.0)  # assets side (same label, second occurrence)

    return pd.DataFrame(rows)


def test_find_section_bounds_locates_known_sections():
    df = _build_synthetic_data_sheet()
    bounds = _find_section_bounds(df[0])
    assert "PROFIT & LOSS" in bounds
    assert "BALANCE SHEET" in bounds
    assert bounds["PROFIT & LOSS"][0] == 11  # row after the header


def test_find_label_row_and_duplicate_total_disambiguation():
    df = _build_synthetic_data_sheet()
    bounds = _find_section_bounds(df[0])
    bs_start, bs_end = bounds["BALANCE SHEET"]

    totals = _find_all_label_rows(df[0], "Total", bs_start, bs_end)
    assert totals == [26, 28]

    net_block_row = _find_label_row(df[0], "Net Block", bs_start, bs_end)
    assert net_block_row == 27


@pytest.fixture
def synthetic_xlsx(tmp_path: Path) -> Path:
    df = _build_synthetic_data_sheet()
    path = tmp_path / "Test Company.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Data Sheet", header=False, index=False)
    return path


def test_parse_fundamentals_file_end_to_end(synthetic_xlsx: Path):
    result = parse_fundamentals_file(synthetic_xlsx)
    assert result.company_name == "TEST COMPANY LTD"

    annual = result.annual
    assert len(annual) == 2
    assert annual.loc[pd.Timestamp("2023-03-31"), "Sales"] == 120.0
    assert annual.loc[pd.Timestamp("2023-03-31"), "Net profit"] == 19.0
    # Both "Total" rows resolved correctly and distinctly.
    assert annual.loc[pd.Timestamp("2023-03-31"), "Total Liabilities & Equity"] == 108.0
    assert annual.loc[pd.Timestamp("2023-03-31"), "Total Assets"] == 108.0
