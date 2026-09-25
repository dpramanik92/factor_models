---
name: fundamental_analysis
description: How to parse the Screener.in fundamentals Excel exports in fundamentals/, derive ROA/ROE/leverage/capex-proxy/revenue-growth/net-margin from raw line items, apply the point-in-time reporting lag, and map company-name filenames to NSE tickers. Use whenever touching fundamentals/, io/fundamentals.py, mapping/, or factors/{leverage,quality,capex,growth,profitability}.py.
---

# Fundamentals parsing and derived ratios

## Source format

`fundamentals/*.xlsx` are Screener.in exports, one per company, **named by company name, not
ticker** (e.g. `Eternal.xlsx` is Zomato's current legal name). Each file has 6 sheets; only
**`Data Sheet`** is reliable - the display sheets (`Profit & Loss`, `Balance Sheet`, etc.) are
formula-linked back to it and read back empty via pandas/openpyxl.

`Data Sheet` layout (approximate row positions - **do not hardcode row indices**, they shift
slightly across files, especially for banks/NBFCs which have blank sector-specific rows):

- META: `COMPANY NAME`, `Number of shares`, `Face Value`, `Current Price`, `Market Capitalization`
- PROFIT & LOSS (annual, `Report Date` = FY-end, ~10 years back): `Sales`, `Raw Material Cost`,
  `Employee Cost`, `Other Expenses`, `Other Income`, `Depreciation`, `Interest`,
  `Profit before tax`, `Tax`, `Net profit`, `Dividend Amount`
- Quarters (last ~10 quarters): `Sales`, `Expenses`, `Net profit`, `Operating Profit`, ...
- BALANCE SHEET (annual, same FY dates): `Equity Share Capital`, `Reserves`, `Borrowings`,
  `Other Liabilities`, `Total` (liabilities+equity side), `Net Block`, `Capital Work in Progress`,
  `Investments`, `Other Assets`, `Total` (assets side - **the label appears twice**; disambiguate
  by position: the assets-side `Total` follows `Net Block`/`Investments`/`Other Assets`, the
  liabilities-side `Total` follows `Borrowings`/`Other Liabilities`)
- CASH FLOW (annual): `Cash from Operating Activity`, `Cash from Investing Activity`,
  `Cash from Financing Activity`, `Net Cash Flow`
- PRICE: year-end close aligned to the annual `Report Date` columns

**Parse by label-anchoring**, not fixed row numbers: scan column A for exact label strings and
build a `label -> row_index` map per file, so the parser survives row-count differences between
companies (e.g. banks have blank raw-material/power-fuel rows).

**Data-quality check**: `Total (assets) ≈ Total (liabilities+equity)` per FY column, tolerance
~0.5%. Log/flag mismatches - don't hard-fail the whole file over one bad column.

**Partial/TTM latest FY**: the most recent `Report Date` column may not be a clean 12-month
period. Cross-check against the trailing-4-quarters sum before using it in a YoY ratio; exclude
it from growth calculations if it doesn't corroborate.

## Derived ratios (none of these exist pre-computed in the source)

- `ROA = Net profit / TotalAssets`
- `ROE = Net profit / (Equity Share Capital + Reserves)`
- `Leverage (D/E) = Borrowings / (Equity Share Capital + Reserves)`
- `Capex proxy = (NetBlock_t - NetBlock_{t-1} + Depreciation_t) / TotalAssets_{t-1}` (no explicit
  capex line exists anywhere in the source; this is an approximation, not ground truth)
- `Revenue growth = Sales_t / Sales_{t-1} - 1` (YoY, FY-over-FY)
- `Net margin = Net profit / Sales`

## Point-in-time discipline

Annual fundamentals are only "known" to the market `REPORTING_LAG_DAYS` (default 60 calendar
days, from SEBI LODR's annual-results filing deadline) after the fiscal year-end. When
forward-filling an FY's fundamentals to a daily panel, the value must not appear before
`FY_end + REPORTING_LAG_DAYS`. Getting this wrong silently introduces look-ahead bias into the
regression - this is exactly what `model_reviewer`'s look-ahead check tests for.

## Company-name -> ticker mapping

Match `fundamentals/*.xlsx` filenames (and the `COMPANY NAME` cell inside, which is sometimes
cleaner) against `data/nse_200_industry_exposure.xlsx`'s company names via `difflib` fuzzy
matching. Known special cases (seed `mapping/overrides.py` with these and add more as found):
`Eternal.xlsx` -> `ZOMATO` (renamed), `M &amp; M.xlsx` -> `M&amp;M`, duplicate-download files like
`Asian Paints (1).xlsx` -> same symbol as `Asian Paints.xlsx` (dedupe, keep newest). Anything
below a confidence threshold gets logged as `needs_review` and excluded from the locked universe
rather than guessed at - `equity_researcher` reviews these by hand.
