# factor_models — Barra-style NSE equity factor risk model

A from-scratch Barra-style cross-sectional equity factor risk model for a 209-stock NSE (India)
universe: factor exposures -> daily cross-sectional WLS regression -> factor return time series
-> EWMA factor covariance matrix.

## Data layout (three-tier, all git-ignored)

- `data/` — raw, read-only in the ordinary build. Yahoo-Finance-sourced NSE price/factor time
  series for 212 tickers (198 original + 12 backfilled + 2 more backfilled in the 2026-09
  expansion below — see below), 2020-09-22 to present, plus an industry-exposure workbook. Not
  written to by the automatic pipeline; the two deliberate, explicit-invocation exceptions are
  `refresh-prices` (io/yahoo_refresh.py) and the one-off backfills described below. (Leftover
  files here named `fama_macbeth_*`/`cross_sectional_regression_*` are outputs from an earlier,
  unrelated modeling attempt — referenced only for naming-convention continuity, not used as
  inputs.)
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
  **History extension**: the price panel was further backfilled from Yahoo Finance back to
  2017-01-02 (for the 172 of 210 tickers that were already trading then) specifically so `beta`
  and `idio_vol` - computed in-house from price data via `factors/market_model.py`, not loaded
  from a precomputed file - would have a longer usable history instead of being bottlenecked by
  a separate precomputed input's own start date. This backfill surfaced 3 isolated NaN gaps in
  the `^NSEI` benchmark quote itself (2018-01-01, 2019-01-01, 2019-10-27 - the last is a Diwali
  "Muhurat trading" session, a real but unusual Sunday session), which were forward-filled since
  the equities themselves had valid prices on those dates - a raw single-day index-quote gap
  like this, left unfixed, poisons every rolling 252-day window that contains it (a
  `min_periods == window` rolling calculation requires *zero* nulls anywhere in the window), so
  it doesn't just affect that one day's beta - it silently delays the entire model's usable
  start date by up to a year per gap.
  **2026-09 fundamentals expansion**: the `fundamentals/` folder grew from 99 to 212 files (see
  below). Rebuilding the universe mapping against this larger set surfaced two more one-off
  backfills, same pattern as above: `ICICIAMC` (ICICI Prudential Asset Management IPO'd on NSE/BSE
  2025-12-19, so it postdated the original data build) and `LTIM` (LTIMindtree; its Yahoo Finance
  quote had been silently failing - `LTIM.NS` 404s post the company's March-2026 rename to "LTM
  Limited", the working ticker is now `LTM.NS`, backfilled under the pre-existing internal symbol
  `LTIM`). Also backfilled in this pass: **daily trading volume** for all 209 locked-universe
  symbols, 2017-01-02 onward (`data/nse198_daily_volume.parquet` - filename kept for continuity;
  previously covered only 198 symbols from 2020-09-22), feeding the new `liquidity` style factor.
- `fundamentals/` — raw, read-only. 212 Screener.in Excel exports, one per company, named by
  company name not ticker (99 from the original build, 113 added 2026-09). Only the `Data Sheet`
  sheet is reliably parseable — see `.claude/skills/fundamental_analysis/SKILL.md`. Two files
  needed a name-collision decision, not just a fuzzy-match override: `LTM.xlsx` (verified via its
  own Sales figures - the FY22 jump from Rs.12,370cr to Rs.26,109cr is exactly the L&T Infotech +
  Mindtree merger - to be LTIMindtree's current, post-merger fundamentals, mapped to symbol
  `LTIM`) vs. `Mindtree.xlsx` (the same company's stale, pre-merger-only standalone filing,
  correctly excluded rather than also mapped to `LTIM`, which would conflate two non-overlapping
  historical filings for what is now one entity).
- `model_data/` — pipeline-built intermediates: the locked universe/ticker mapping, parsed
  fundamentals, and the full factor exposure panel.
- `output/` — final deliverables: daily factor returns, regression diagnostics, the EWMA
  covariance matrix, and the validation report.

## Universe

**209 tickers** — the intersection of the price universe (212 tickers) and the 212 companies
with fundamentals coverage. Locked in `model_data/universe_99.csv` (filename kept for historical
continuity even though the locked count is now 209) after review by the `equity_researcher` agent
(both the original 96-ticker lock and the 2026-09 expansion to 209). The 3 remaining exclusions
are all genuine: `Mindtree` (superseded by `LTM.xlsx`'s post-merger fundamentals under symbol
`LTIM` - see above), and `Tata Motors`/`Jindal Steel` (matched correctly but `NO_DATA` in the
price panel, likely due to 2024 corporate restructuring). See `.claude/rules/data-boundaries.md`.

## Factors

Beta and idiosyncratic volatility (both computed in-house from a rolling 252-day market-model
regression against NIFTY50 - `factors/market_model.py` - not loaded from a precomputed file),
1-year momentum (252-trading-day return excluding the most
recent 21 days), size (log of in-house-computed market cap = price x point-in-time shares
outstanding, `factors/market_cap.py` - z-scored, plus a diagnostic-only size decile column),
value (composite of E/P, B/P, dividend yield, each computed in-house from price x point-in-time
per-share fundamentals - EPS, book value/share, dividend/share - rather than loaded from a
precomputed file), leverage (debt/equity), quality (composite of ROA
and ROE), capex (proxy: ΔNet Block + Depreciation, scaled by prior-period total assets), growth
(revenue YoY), profitability (net margin), liquidity (log of trailing-252-day average daily share
turnover = volume / point-in-time shares outstanding, `factors/liquidity.py` - computed in-house
from `data/nse198_daily_volume.parquet`, backfilled 2017-01-02 onward for all 209 universe
symbols alongside the price history; higher = more liquid, added 2026-09 alongside the
99->212-fundamentals-file universe expansion), and 10-bucket weighted industry exposures. Market
cap is also the WLS regression weight (`sqrt(market_cap)`), so it's computed once in
`factors/market_cap.py` and shared rather than recomputed per consumer. All
continuous factors are winsorized (1st/99th percentile) then z-scored cross-sectionally per date
— see `.claude/skills/statistics/SKILL.md` for the exact spec and the two exceptions (size,
industry).

**2 binary 0/1 indicator factors** (`config.INDICATOR_FACTORS`, never winsorized/z-scored - raw
dummies, same convention as the industry weight columns): `top100_flag` (1 if a stock ranks in
the top 100 by market cap that date, `factors/top100.py` - reuses the shared market-cap panel,
re-ranked per date) and `psu_flag` (1 if the company is a Government-of-India Public Sector
Undertaking, `factors/psu.py` - static, hand-classified in `model_data/psu_flags.csv`, narrowly
defined as majority direct government ownership/official CPSE status, not merely "a PSU holds a
stake" - e.g. SBI Life/SBI Cards are NOT flagged despite majority SBI ownership, matching
standard PSU-index convention; see that file's `rationale` column, including a few flagged-for-
re-verification borderline cases like CONCOR given India's ongoing disinvestment program). Added
2026-09 after observing that the factors driving the largest ~100 stocks behave differently from
the rest - each is a *level* effect only (an average return premium/discount for segment
membership); neither is individually significant on its own but both modestly improved overall
fit in the pooled model. See "Known limitations" below for the large-cap/small-cap structural
break these only partially addressed, and how it's actually handled now (separate per-segment
regressions, not a pooled-model indicator/interaction/dummy trick - three of those were tried and
none fixed it).

Fundamentals-derived factors (leverage, quality, capex, growth, profitability) are annual and
forward-filled to daily frequency only after a 60-calendar-day reporting lag (SEBI LODR's annual
filing deadline), to avoid look-ahead bias — see `.claude/skills/fundamental_analysis/SKILL.md`.

## Regression and risk model

Daily cross-sectional WLS: `y = excess return`, `X = [10 industry weights, no separate
intercept] + [10 z-scored style factors]`, `weight = sqrt(market_cap)`, exposures dated `t-1`.
Newey-West HAC t-stats alongside naive t-stats. EWMA factor covariance (90-trading-day half-life
default) and per-stock EWMA specific-risk variance from residuals. Full spec:
`.claude/skills/statistics/SKILL.md`.

## Portfolio construction

Long-only mean-variance optimization (GMV, Max-Sharpe with a soft single-name/single-industry
concentration penalty, Equal-Weight benchmark) plus a walk-forward monthly-rebalanced backtest -
`portfolio/`. Two parallel versions exist, saved to separate output directories so neither
overwrites the other:

- **Pooled** (`output/portfolio/`, `factor-model optimize-portfolio` /
  `validate-portfolio` / `backtest-portfolio`): built directly on the pooled full-window factor
  model's outputs (`output/factor_returns_daily.csv`, `output/factor_covariance_latest.csv`,
  `output/specific_risk_latest.csv`) - inherits that model's known large-cap/small-cap structural
  break (see above).
- **Segmented** (`output/portfolio_segments/`, `factor-model optimize-portfolio-segments` /
  `validate-portfolio-segments` / `backtest-portfolio-segments`): built on the two independent
  per-segment models instead (`output/segments/top100/`, `output/segments/rest/`) via
  `portfolio/segmented_risk_model.py`. Each segment fits its own daily coefficients, so its
  factor-return series is a genuinely different series even where column names match - to combine
  them into one N-stock covariance matrix without conflating the two, every design column is
  suffixed by segment (`beta__top100` / `beta__rest`, etc.) into a joint 2K-column factor space; a
  stock's exposure row is nonzero only in its own current segment's block (from `top100_flag` on
  the latest date - it cannot load on the other segment's factors); and the joint 2K x 2K factor
  covariance is estimated by ordinary EWMA covariance over both segments' factor-return series
  stacked column-wise and aligned by date, which recovers the empirical cross-segment covariance
  rather than assuming the two segments' risks are independent. Validated 8/8 on the independent
  portfolio-validation suite (PSD covariance, realized/predicted risk ratio 0.90-1.01x, positive
  significant expected-vs-realized-return correlation) against the real 189-symbol optimizable
  universe. Requires `run-segments` to have been run first (reuses its outputs; does not rebuild
  them). The walk-forward backtest re-derives both segments' EWMA covariance/specific-
  risk/expected-returns at each monthly rebalance from truncated (as-of-date) slices of each
  segment's already-point-in-time daily factor-return/residual series - the same "no need to
  re-run the cross-sectional regression itself at every rebalance date" argument
  `portfolio/backtest.py`'s docstring makes for the pooled backtest applies per segment here too.

  **Per-segment single-stock cap**: `optimize-portfolio-segments`/`validate-portfolio-segments`/
  `backtest-portfolio-segments` all take an optional `--rest-max-stock-weight` (e.g. `0.20`),
  trying a looser single-stock soft cap on the "rest" (non-top100) segment than the top100
  segment's default 10% (`config.MAX_STOCK_WEIGHT`) - `portfolio/optimize.py`'s
  `max_stock_weight` parameter now accepts a Symbol-indexed `pd.Series` (a per-stock cap) as well
  as a single float, and `segmented_risk_model.build_max_stock_weight` builds that Series from
  each stock's current segment. A non-default value writes to a distinct sibling directory
  (`output/portfolio_segments_rest<N>/`) rather than overwriting the default (uniform-cap) run, so
  multiple single-stock-cap experiments coexist and can be compared side by side.

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
- **Large-cap/small-cap structural break**: expanding the universe from 96 to 209 stocks surfaced
  a large, highly significant, opposite-signed unexplained-return bias between the original 96
  (89% top100-by-cap, ran ~+3.4%/year unexplained) and the 113 added stocks (87% not-top100, ran
  ~-2.1%/year) - `validation/statistical_tests.py`'s `test_residual_alpha_by_cap_segment` checks
  both segments separately so this can't hide in a pooled average again (the pooled number alone
  looked only borderline, +0.4-0.5% annualized, because the two biases nearly canceled out).
  Three *pooled-single-regression* fixes were tried, in order, and all failed:
  1. `top100_flag`/`psu_flag` main-effect dummies alone (a level shift) - didn't touch the
     problem, since the true issue is differential factor *loadings*, not a differential average
     level.
  2. A `style_factor x top100_flag` interaction term (letting each style factor's coefficient
     differ by cap segment) - made the segment-level bias *worse* (NW t up to ~21 from ~6), most
     likely because the shared WLS fit (weighted by `sqrt(market_cap)`) still let the largest few
     names dominate regardless of how many columns existed.
  3. `size_decile_2`..`size_decile_10` dummy factors (`factors/size.py`'s
     `add_size_decile_dummies`, letting the size effect be nonlinear across 9 finer buckets
     instead of one crude top100/rest split) - barely changed the segment bias (NW t ~19/-13,
     essentially the same severity as #2) *and* introduced severe multicollinearity (VIF 30-48 on
     `size`/`top100_flag`/the upper deciles, since all three are encodings of the same underlying
     market-cap ranking). Reverted; the function is kept (tested, working) but not called from
     `factors/panel.py`.
  4. A single, narrower `size x top100_flag` interaction column (isolating just the size
     coefficient rather than every style factor's) - improved the *pooled* residual-alpha metric
     (0.42% -> 0.07% annualized, FAIL -> PASS) but made the *segment-level* bias even worse still
     (NW t 17.3/-7.8 -> 20.2/-17.6), the same pooled-metric-masks-a-worse-split pattern the
     segment diagnostic exists to catch - a single targeted interaction isn't a lighter-weight
     version of fix #2 that avoids its failure mode, it has the same failure mode. Reverted; not
     wired into `factors/panel.py`/`config.py`.
  **Currently active fix**: fully separate per-segment regressions
  (`config.SEGMENT_DESIGN_COLS`, `pipeline.run_segment`, `factor-model run-segments`) - one
  regression for the top-100-by-cap segment, one for the rest, each with its own factor
  returns/covariance/specific-risk, rather than forcing one shared set of coefficients across
  both. `SEGMENT_DESIGN_COLS` keeps size as continuous z-scored `log(market_cap)` only (no
  decile dummies, no `top100_flag` - the latter would be constant within a single segment and
  exactly collinear with the industry weights, which already sum to 1 per row). This is the first
  fix that actually works: each segment's own `test_residual_alpha_significance` (run
  independently per segment, not the cross-segment check, which is skipped for `is_segment=True`
  runs) is a clean PASS - top100 0.06% annualized (NW t=1.05), rest 0.03% (NW t=0.23), both far
  below the pooled model's FAIL levels - and VIF is low in both segments (all factors <3, vs. the
  pooled model's `top100_flag`/`size_x_top100` inflation in fix #4). The segments also recover
  genuinely different factor loadings, confirming the original observation that motivated all of
  this: `beta` is significant only in the top100 segment (NW t=2.61 vs 0.81), `momentum` only in
  the rest segment (NW t=3.86 vs 1.33), consistent with large caps behaving more like
  systematic-risk-driven names and smaller caps more like momentum-driven names. The segmentation
  now extends into portfolio construction too - see "Portfolio construction" below.

## Project conventions

- All subagents are pinned to Sonnet (`.claude/settings.json`). Agent teams are intentionally
  **not** enabled — `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` is left unset (its default), and
  should stay that way.
- `tests/` is fast and data-independent (synthetic fixtures); the `validate` command needs the
  real local `data/`/`fundamentals/` folders and is not meant to run in a CI environment that
  lacks them.
- Local git repo only — no GitHub remote. `data/`, `fundamentals/`, `model_data/`, and `output/`
  are git-ignored; only code is committed.
