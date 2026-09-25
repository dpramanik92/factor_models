# factor_models — Barra-style NSE equity factor risk model

A from-scratch Barra-style cross-sectional equity factor risk model for a 96-stock NSE (India)
universe: factor exposures -> daily cross-sectional WLS regression -> factor return time series
-> EWMA factor covariance matrix.

## Data layout (three-tier, all git-ignored)

- `data/` — raw, read-only in the ordinary build. Yahoo-Finance-sourced NSE price/factor time
  series for 210 tickers (198 original + 12 backfilled — see below), 2020-09-22 to present, plus
  an industry-exposure workbook. Not written to by the automatic pipeline; the two deliberate,
  explicit-invocation exceptions are `refresh-prices` (io/yahoo_refresh.py) and the one-off
  backfill described below. (Leftover files here named `fama_macbeth_*`/`cross_sectional_regression_*`
  are outputs from an earlier, unrelated modeling attempt — referenced only for naming-convention
  continuity, not used as inputs.)
  **One-off backfill**: 12 fundamentals-covered companies were originally missing from both the
  price panel and the industry-exposure workbook entirely (not just below the fuzzy-match
  threshold). Their price history and a precedent-matched industry classification (e.g. Siemens
  Energy India classified 80% Energy & Utilities / 20% Industrials, matching BHEL's exact split
  for the same power-equipment business) were fetched from Yahoo Finance and added directly to
  `data/nse200_nifty50_daily_prices.parquet`/`.csv.gz`, `data/nse_200_timeseries_availability.csv`,
  and `data/nse_200_industry_exposure.xlsx` as a manual, one-time data-repair action — not
  something `refresh-prices` or the pipeline does automatically. Several of these (Hyundai Motor
  India, Jio Financial, Lodha/Macrotech, Siemens Energy India, Tata Capital) are recent
  IPOs/demerger listings, so their price history starts after 2020-09-22, not at the panel's
  start date - their factor coverage will correspondingly start later than older names.
- `fundamentals/` — raw, read-only. 99 Screener.in Excel exports, one per company, named by
  company name not ticker. Only the `Data Sheet` sheet is reliably parseable — see
  `.claude/skills/fundamental_analysis/SKILL.md`.
- `model_data/` — pipeline-built intermediates: the locked universe/ticker mapping, parsed
  fundamentals, and the full factor exposure panel.
- `output/` — final deliverables: daily factor returns, regression diagnostics, the EWMA
  covariance matrix, and the validation report.

## Universe

**96 tickers** — the intersection of the price universe (210 tickers after the one-off Yahoo
Finance backfill described above) and the 99 companies with fundamentals coverage. Locked in
`model_data/universe_99.csv` (filename kept for continuity even though the locked count is 96)
after review by the `equity_researcher` agent. The 3 remaining exclusions are all genuine: `LTM`
(unresolved company identity, not found on Yahoo Finance either), and `Tata Motors`/`Jindal Steel`
(matched correctly but `NO_DATA` in the price panel, likely due to 2024 corporate restructuring).
See `.claude/rules/data-boundaries.md`.

## Factors

Beta, idiosyncratic volatility, 1-year momentum (252-trading-day return excluding the most
recent 21 days), size (log market cap, z-scored — plus a diagnostic-only size quintile column),
value (composite of E/P, B/P, dividend yield), leverage (debt/equity), quality (composite of ROA
and ROE), capex (proxy: ΔNet Block + Depreciation, scaled by prior-period total assets), growth
(revenue YoY), profitability (net margin), and 10-bucket weighted industry exposures. All
continuous factors are winsorized (1st/99th percentile) then z-scored cross-sectionally per date
— see `.claude/skills/statistics/SKILL.md` for the exact spec and the two exceptions (size,
industry).

Fundamentals-derived factors (leverage, quality, capex, growth, profitability) are annual and
forward-filled to daily frequency only after a 60-calendar-day reporting lag (SEBI LODR's annual
filing deadline), to avoid look-ahead bias — see `.claude/skills/fundamental_analysis/SKILL.md`.

## Regression and risk model

Daily cross-sectional WLS: `y = excess return`, `X = [10 industry weights, no separate
intercept] + [10 z-scored style factors]`, `weight = sqrt(market_cap)`, exposures dated `t-1`.
Newey-West HAC t-stats alongside naive t-stats. EWMA factor covariance (90-trading-day half-life
default) and per-stock EWMA specific-risk variance from residuals. Full spec:
`.claude/skills/statistics/SKILL.md`.

## Running it

```
PYTHONPATH=src ./.venv/bin/python -m factor_model.cli run-all      # full pipeline
PYTHONPATH=src ./.venv/bin/python -m factor_model.cli validate      # statistical validation only
./.venv/bin/pytest tests/                                            # unit tests, synthetic fixtures only
```

(`PYTHONPATH=src` works around an editable-install quirk seen in this environment where
`pip install -e .` doesn't reliably put `src/` on `sys.path`; try plain `factor-model run-all`
first - if it resolves `factor_model` cleanly for you, the prefix isn't needed.)

## Agents

- `equity_researcher` — fundamentals/mapping data sanity, reviews the ticker mapping.
- `Quant_analyst` — factor-formula and regression/EWMA methodology.
- `Progammar` — general Python engineering (pipeline, CLI, tests).
- `model_reviewer` — independent statistical validation of the finished model; writes
  `output/validation_report.md`. Never edits model code to force a check to pass.

## Known limitations

- **Corporate actions**: fundamentals ratios (leverage/quality/capex/growth/profitability) are
  computed from balance-sheet/income-statement totals, not per-share figures, so bonus issues and
  stock splits don't distort them. Company renames are handled via
  `mapping/overrides.py` (e.g. Zomato→Eternal). Genuine mergers that distort YoY fundamentals
  comparability (e.g. HDFC Ltd merging into HDFC Bank in 2023) are handled via an explicit,
  hand-maintained exclusion list in `factors/known_corporate_actions.py` - a generic statistical
  outlier threshold can't reliably distinguish "merger-driven jump" from "genuinely fast organic
  growth" (verified against real data: Zomato/MCX show 100%+ revenue growth that's real). Price-
  based factors (beta, idio-vol, momentum, size, and the regression's return variable) rely on
  whatever adjustment Yahoo Finance already applied; single-day |return| >
  `CORPORATE_ACTION_RETURN_THRESHOLD` (40%) is excluded from that day's regression and logged to
  `output/flagged_corporate_actions.csv`, but we do not reconstruct corporate-action-adjusted
  total returns (e.g. splicing in spun-off-entity value for a demerger like ABFRL or Vedanta).
  Dividends are likely not adjusted out of the price series (a minor, uncorrected bias).
- **Universe drift**: a handful of companies present in the 200-symbol reference list have
  `NO_DATA` in the price panel (e.g. `TATAMOTORS`, `JINDALSTEEL`), likely due to corporate
  restructuring around 2024 - their fundamentals files are correctly excluded from the locked
  universe (`model_data/universe_99.csv`) since there's no return series to regress against.
- **Ticker mapping is not purely threshold-based**: fuzzy company-name matching auto-accepts only
  above a 0.90 similarity score; borderline-but-correct matches (e.g. "I R F C" vs "Indian Railway
  Finance Corporation") are confirmed by hand in `mapping/overrides.py` rather than by lowering
  the threshold, because verified testing showed correct and incorrect matches overlap in the
  0.65-0.89 score band (a lower threshold would silently admit wrong matches).

## Project conventions

- All subagents are pinned to Sonnet (`.claude/settings.json`). Agent teams are intentionally
  **not** enabled — `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` is left unset (its default), and
  should stay that way.
- `tests/` is fast and data-independent (synthetic fixtures); the `validate` command needs the
  real local `data/`/`fundamentals/` folders and is not meant to run in a CI environment that
  lacks them.
- Local git repo only — no GitHub remote. `data/`, `fundamentals/`, `model_data/`, and `output/`
  are git-ignored; only code is committed.
