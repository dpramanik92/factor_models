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
  universe (out of the 209-stock full universe - see "Why the optimizable universe is smaller
  than 209" below). Requires `run-segments` to have been run first (reuses its outputs; does not
  rebuild them). The walk-forward backtest re-derives both segments' EWMA covariance/specific-
  risk/expected-returns at each monthly rebalance from truncated (as-of-date) slices of each
  segment's already-point-in-time daily factor-return/residual series - the same "no need to
  re-run the cross-sectional regression itself at every rebalance date" argument
  `portfolio/backtest.py`'s docstring makes for the pooled backtest applies per segment here too.

  **Why the optimizable universe is smaller than 209**: both `get_latest_exposures`
  (`portfolio/risk_model.py`) and `get_latest_segment_exposures`
  (`portfolio/segmented_risk_model.py`) drop, rather than impute, any symbol missing a complete
  factor exposure on the latest date (e.g. a recent IPO/listing without a full 252-day
  beta/idio_vol/liquidity history yet) or a specific-risk (EWMA idiosyncratic-variance) estimate.
  As of 2026-09-22: 198/209 have complete exposure; the pooled model's Equal-Weight/GMV/Max-Sharpe
  universe is **197** (1 more drops for missing specific risk). The segmented model's is smaller
  still, **189** - a *stricter* requirement, since specific risk must exist *within a stock's
  currently-assigned segment specifically* (a stock that recently crossed the top100/rest
  boundary may have plenty of overall history but only a short residual history in its *new*
  segment). This is intentional (`portfolio/risk_model.py`: "dropped rather than given a
  fabricated/imputed value"), not a bug - re-verify this reasoning holds before trusting a
  different optimizable-universe count without re-deriving it the same way.

  **Per-segment single-stock cap**: `optimize-portfolio-segments`/`validate-portfolio-segments`/
  `backtest-portfolio-segments` all take an optional `--rest-max-stock-weight` (e.g. `0.20`),
  trying a looser single-stock soft cap on the "rest" (non-top100) segment than the top100
  segment's default 10% (`config.MAX_STOCK_WEIGHT`) - `portfolio/optimize.py`'s
  `max_stock_weight` parameter now accepts a Symbol-indexed `pd.Series` (a per-stock cap) as well
  as a single float, and `segmented_risk_model.build_max_stock_weight` builds that Series from
  each stock's current segment.

  **Group-level rest-segment cap**: the same three commands also take `--no-industry-cap`
  (removes the hard 20% per-industry cap, `config.MAX_INDUSTRY_WEIGHT`, entirely) and
  `--rest-segment-max-weight` (e.g. `0.20`, plus optional `--rest-segment-weight-penalty`) - a
  *soft* cap (discouraged, not forbidden) on the total weight held across the whole "rest"
  segment as one group, independent of any individual stock's own cap.
  `portfolio/optimize.py`'s `minimize_variance`/`trace_efficient_frontier` gained a generic
  `group_exposure`/`group_max_weight`/`group_weight_penalty` mechanism for this (a 0/1 membership
  vector, penalized the same way as the single-stock soft cap but for one linear combination of
  weights); `segmented_risk_model.build_group_exposure` builds the rest-segment membership vector.
  `config.REST_SEGMENT_WEIGHT_SOFT_PENALTY_COEF` (0.01) was calibrated empirically: with *no* cap
  at all (no industry cap, no group cap), Max-Sharpe puts **88.7%** of the portfolio into the rest
  segment alone (see `output/portfolio_segments_noindcap/`) - 0.01 pulls that down to a genuinely
  soft ~22% (0.05+ is indistinguishable from a hard cap at ~20.0-20.2%).

  Any non-default combination of `--rest-max-stock-weight`/`--no-industry-cap`/
  `--rest-segment-max-weight` writes to a distinct, auto-suffixed sibling directory (e.g.
  `output/portfolio_segments_rest20/`, `output/portfolio_segments_noindcap_restcap20/`) rather
  than overwriting `output/portfolio_segments/`, so every combination tried stays available side
  by side for comparison - see `pipeline._segmented_output_paths`.

  **Macro-timing expected-return tilts** (`--macro-timing`, all six portfolio commands -
  `portfolio/macro_timing.py`): wires the two significant, Bonferroni-robust signals from "Alpha
  research" below (Oil-price momentum -> Oil group, USD/INR momentum -> IT Exporters) directly
  into the optimizer as small additive daily tilts on `mu_stock`, applied only to each signal's
  own group members. Point-in-time correct by construction: `compute_macro_timing_tilt` re-fits
  each signal's regression slope fresh from an expanding window of history up to `as_of` every
  time it's called - including at every walk-forward-backtest rebalance date - never a fixed,
  pre-calibrated constant, so it's automatically safe against look-ahead bias. Like every other
  non-default portfolio variant, `apply_macro_timing=True` writes to its own output directory
  (`config.PORTFOLIO_MACRO_TIMING_*` for the pooled model, an auto-suffixed
  `output/portfolio_segments_..._macrotiming/` for the segmented model via
  `pipeline._segmented_output_paths`) rather than overwriting the no-tilt baseline - this was a
  real bug caught mid-session (the first pooled `--macro-timing` run initially wrote to the same
  `output/portfolio/` directory as the baseline, silently overwriting it) and fixed before it did
  more than overwrite one already-captured result.

  **Long-short** (`--long-short`/`--max-leverage`, the three `*-portfolio-segments` commands only
  - `optimize.py`'s `minimize_variance_long_short`/`trace_efficient_frontier_long_short`, already
  built and tested in an earlier session but never wired past the pooled model there, where it
  underperformed long-only net of extra turnover/leverage - see `portfolio/backtest.py`'s module
  docstring): lets GMV/Max-Sharpe hold negative (short) weights within +/-`config.MAX_STOCK_WEIGHT`
  per stock, subject to a gross-leverage cap `--max-leverage` (default `config.MAX_LEVERAGE`=1.2,
  i.e. up to 10% short; `--max-leverage 1.4` allows up to 20% short, a "120/40"-style book).
  Equal-Weight is unaffected (stays long-only 1/n, the benchmark). The long-short optimizer only
  supports a uniform scalar single-stock cap and the hard industry cap - `--rest-max-stock-weight`/
  `--rest-segment-max-weight` (the per-segment cap / group-level soft cap) have no effect together
  with `--long-short` (a warning is logged). Validation reuses `test_constraint_satisfaction`'s
  pre-existing `long_short_names` parameter (sum=1 net exposure, gross exposure <= max_leverage,
  instead of the long-only sum=1-and-w>=0 check) and `plot_portfolio_weights`'s `by_abs=True` mode
  (ranks by |weight|, shorts shown in red) - both already built generically for this, just not
  previously reachable from the CLI.

  **Walk-forward backtest results across every variant tried** (Max-Sharpe Sharpe ratio, 21
  monthly rebalances, 2025-01 through 2026-09): pooled -0.10 (see "why not the pooled model"
  below - re-verify against a fresh baseline rerun before quoting, this figure predates the
  macro-timing/long-short additions); segmented with the original uniform 10%-stock/hard-20%-
  industry caps 0.54; segmented with a 20% per-stock cap on rest only 0.37; segmented with no
  industry cap + a 20% soft group cap on total rest weight 0.47; segmented + `--macro-timing`
  0.71; **segmented + `--macro-timing` + `--long-short --max-leverage 1.4` (20% max short) 0.89
  (best)**. Each addition on top of the base segmented model was a genuine incremental
  improvement (higher return, similar-or-lower volatility, shallower max drawdown) - full
  progression: pooled (structural break, no fix) -> segmented (fixes the break) -> + macro-timing
  (adds two independently-validated alpha signals) -> + long-short (lets the same signals express
  as shorts, e.g. `IGL`/`HINDPETRO` shorted when oil-momentum's tilt turns negative for that
  group). Separately, applying `--macro-timing` to the *pooled* model alone (without segmentation)
  improved it from -0.34 to +0.08 - real, but still far worse than the segmented baseline (0.54)
  with no tilts at all, confirming segmentation (fixing the structural break) matters far more
  than either alpha signal on its own; see "Known limitations" for why the break exists.

## Alpha research

Exploratory candidate-factor/signal checks, kept separate from the production model
(`config.STYLE_FACTORS`) - a factor only gets wired into `factors/panel.py` after clearing a real
significance bar (the same |NW_t| > 2 standard the regression's own factor significance uses),
not just a plausible economic story. Reported here (both positive and negative results) so a
future session doesn't re-run a check that's already been done:

- **Environmental score quintile spread** (`R_[top_env_quintile] - R_[bottom_env_quintile]`,
  `model_data/environmental_scores.csv` - a proxy sector-baseline + manual-adjustment score,
  explicitly *not* verified third-party ESG data, only covering the original 96-stock universe):
  no alpha, NW t=0.47, not significant even before controlling for the sector confound the scores
  are partly built from.
- **Environmental Good/Bad binary spread** (`R_good - R_bad`,
  `model_data/environmental_good_bad.csv` - same proxy-heuristic methodology extended to the
  full 209-stock universe as a binary call instead of a false-precision 0-100 score: 103 Good /
  55 Bad / 51 Neutral-and-excluded): still no alpha - **-3.50% annualized, NW t=-1.22**, wrong
  sign for a "clean stocks carry a premium" story (though not significant enough to claim the
  opposite either), stable in direction but never significant across either half of the sample
  (first half NW t=-0.56, second half NW t=-1.21). Confirmed to be essentially a **sector bet in
  disguise**, the same confound the continuous-score version had: the Good bucket is
  concentrated in Financial Services (40% avg weight) + Healthcare (19%) + IT (14%), the Bad
  bucket in Materials (55%) + Energy & Utilities (17%) - not independent of the industry factors
  already in the model. Size is not a confound (Good/Bad average z-scored size differs by only
  ~0.07). Neither environmental construction has produced a real, independent alpha signal.
- **Macro-variable predictability** (lagged monthly regression of oil price / USD-INR / gold /
  US 10yr yield returns against each style factor's own return): no reliably significant
  predictor survived a Bonferroni correction across the factors tested.
- **Accruals** (`factors/accruals.py`, Sloan 1996 earnings-quality anomaly:
  `(NetProfit - CashFromOperatingActivity) / TotalAssets`, from the `Cash from Operating
  Activity` line already parsed by `io/fundamentals.py`, ~99.9% coverage): a quintile-sort
  Q5(high accruals)-Q1(low accruals) forward-return spread over the full sample is
  **-5.27% annualized, NW t=-1.81** - directionally exactly what the anomaly predicts (high
  accruals underperform), but short of this project's significance bar. Splitting by cap segment
  *weakens* both halves (top100 NW t=-0.91, rest NW t=-1.44 - expected from roughly halving the
  daily cross-section, not evidence the effect concentrates in either segment). Raw accruals also
  isn't a confound of quality/leverage/value (|corr| < 0.17 with each in the fitted exposure
  panel). Not wired into `config.STYLE_FACTORS` - the closest call of everything tried so far, but
  not over the bar. The module and its test (`tests/unit/test_accruals.py`) are kept, same
  convention as `factors/size.py`'s unused `add_size_decile_dummies`.
- **Short-term reversal** (`factors/reversal.py`: the negative of the trailing 21-trading-day
  return - exactly the window `factors/momentum.py`'s 12-1 convention excludes): no alpha, Q5-Q1
  spread NW t=0.52. Low correlation with size/momentum/idio_vol (all |corr| < 0.07) rules out an
  obvious confound, but there's simply no signal to explain. Not wired in; module/test kept.
- **Governance Good/Bad** (`model_data/governance_good_bad.csv`, 209 stocks): unlike the
  environmental score, this is *not* a sector-baseline heuristic - it's built from real,
  verified news/regulator sources (SEBI/RBI orders, forced executive resignations, fraud
  investigations, found via `WebSearch` in 2026-09), since governance issues are company-
  specific, not sector-driven. 11 stocks flagged "Bad" on a documented, reasonably-recent (~last
  3 years) governance violation - `ZEEL` (active 2026 SEBI order: Zee + Subhash Chandra/Punit
  Goenka barred from the market over an undisclosed promoter-linked asset pledge), `INDUSINDBK`
  (2025 derivatives-accounting scandal, ~INR2,000cr P&L hit, CEO/deputy CEO resigned + SEBI
  insider-trading order), `PAYTM` (RBI cancelled Paytm Payments Bank's licence explicitly citing
  governance lapses), `IIFL` (RBI gold-loan business ban, assaying/LTV/cash violations),
  `MANAPPURAM` (RBI lending ban on subsidiary Asirvad), `ADANIENT`/`ADANIENSOL`/`ADANIGREEN`/
  `ADANIPORTS`/`ADANIPOWER` (Sept 2026 SEBI settlement over RPT/governance issues at 5 entities -
  separate from, and milder than, the Hindenburg allegations SEBI's investigation found no
  violation on), `BANDHANBNK` (RBI monetary penalty + a retired-RBI-official board observer).
  Explicitly checked and *not* flagged: `IDEA` (financial distress from AGR dues, not a
  governance violation), `FORTIS` (the Singh-brothers fraud was under prior ownership - IHH
  Healthcare has held board control since 2018), `RBLBANK` (a 2021 RBI-director episode, but
  2024-2025 activity reads as routine RBI-approved succession), `DLF` (only a counterparty in
  another company's fraud, not itself implicated). The remaining 198 default to "Good" (absence
  of a found flag, not a positive rating - a real issue could exist that this search pass simply
  didn't surface).

  **Two versions were tested, and the difference is the finding**: naively applying the Bad flag
  across each stock's *entire* history (2020-2026) gives a superficially larger spread (-9.29%
  annualized) that looked marginally significant in the first half of the sample (NW t=-1.84) and
  vanished in the second half (t=-0.00) - but this is **methodologically circular**: the Bad list
  itself was built with 2024-2026 hindsight, so testing pre-disclosure years against it means
  comparing the market's *actual* pre-scandal beliefs against a label the market couldn't have
  known yet. Re-run **point-in-time correctly** (each stock only counts as Bad from the date its
  issue actually became public, e.g. `ADANIENT` from the 2023-01-24 Hindenburg report,
  `INDUSINDBK` from its 2025-03-10 disclosure), the *ongoing* forward-looking Good-Bad spread from
  each flag date onward is **not significant at all** (NW t=0.29). What the point-in-time version
  does show clearly: every one of the 11 flagged stocks fell sharply in the 21 trading days
  immediately following its own flag date (-7.6% to -72.1%) - the market reacts to governance
  scandal news **immediately and severely**, consistent with market efficiency, not with a
  slow-bleed pattern a systematic factor could capture after the fact. Net conclusion: no
  tradable "avoid bad-governance stocks" alpha exists once an issue is public, because by the
  time it's public the repricing has already happened - not wired into any factor; kept as a
  reusable methodological lesson (retrospective-hindsight labels need a point-in-time flag date,
  the same discipline `factors/pit.py`'s reporting lag already enforces for fundamentals) as much
  as a factor-research result.
- **Post-Earnings-Announcement Drift (PEAD)** - a genuinely different angle from every factor
  above: an *event-time* drift test (60 trading days after each company's own earnings
  announcement) rather than a *calendar-time* daily cross-sectional sort. SUE (standardized
  unexpected earnings) proxy: YoY change in annual EPS, standardized by that company's own
  trailing rolling std of the same YoY change (Foster/Olsen/Shevlin's seasonal-random-walk
  convention, adapted to annual frequency); announcement date = `FY_end + REPORTING_LAG_DAYS`,
  same PIT convention as everywhere else in the model. Result: **no drift at all** - Q5 (biggest
  positive surprise) vs Q1 (biggest negative surprise) 60-day forward-return spread is 0.11%,
  t=0.07, on 1,374 (Symbol, FY) events; the five SUE quintiles' mean forward returns don't even
  order monotonically (11.98%/10.72%/8.69%/11.00%/12.09%). **Likely reason, not just "no
  effect"**: PEAD is a quarterly-earnings phenomenon in the literature, and this data only has
  *annual* EPS parsed (`io/fundamentals.py` recognizes the Screener "Quarters" section header but
  doesn't parse its rows yet) - by the time an *annual* report is 60 days old, an Indian company
  that reports quarterly has already told the market 4 quarters' worth of information along the
  way, so the "surprise" at the annual reporting date is mostly stale, already-priced-in news,
  not a fresh event. A real PEAD test would need quarterly EPS/surprise parsed - a real, scoped
  engineering addition (new parser section + a ~45-day quarterly reporting lag, not the 60-day
  annual one).

  **Quarterly follow-up (built next)**: added the "Quarters" block to `io/fundamentals.py`
  (`_parse_quarterly`, `ParsedFundamentals.quarterly`), `factors/fundamentals_ratios.py`'s
  `build_fundamentals_quarterly_long`, `config.QUARTERLY_REPORTING_LAG_DAYS` (45 days), and
  `factors/pead.py`'s `compute_quarterly_sue` (same-quarter-year-ago lag-4 comparison, not a
  naive lag-1, so a seasonal Q4-every-year spike isn't misread as a surprise). Re-running PEAD on
  real quarterly Net-profit SUE gives a result none of the other alpha checks did: a **directionally
  consistent, monotonic pattern that the data can't yet confirm statistically**. Pooled across 618
  (Symbol, Quarter) events, Q5-Q1 60-day forward spread = +4.42% (t=2.43, p=0.016), monotonic
  across all five quintiles (0.25%/-0.45%/1.11%/2.93%/4.67%), and growing with horizon (+1.57% at
  5 days -> +4.42% at 60 days) - the classic PEAD shape (a partial immediate reaction followed by
  continued drift), not a single-day jump. **But**: Screener only exports each company's trailing
  ~10 quarters, so the earliest quarter with enough own history for a lag-4 SUE is 2025-09-30 -
  the 618 events collapse into just **3 distinct announcement-month clusters** (2025-09, 2025-12,
  2026-03), not 618 independent observations, so the pooled t-test's significance is
  overstated - the true effective sample size for inference is closer to 3 than 618. Tested each
  cluster separately: none individually reaches significance (p=0.185, 0.100, 0.205), but all
  three point the same direction (spreads of +3.16%/+6.76%/+3.73%) - suggestive (3/3 same-sign
  is weak evidence, not proof, with a sign-test p-value of only ~0.25 at n=3), genuinely
  promising, but not confirmable with the history currently available. Would need several more
  years of quarterly cycles (or a data source with full quarterly history instead of a trailing
  ~10-quarter window) to actually test this properly - the single clearest "come back to this
  once more time has passed" result of all six alpha-research attempts, rather than a clean
  reject. Not wired into any factor; `factors/pead.py`/`build_fundamentals_quarterly_long` are
  kept as tested, reusable infrastructure for that future re-test.
- **Defence/military-tech theme, trailing 1 year** - a descriptive "did this theme actually
  outperform recently" check (not a predictive factor test), since Indian defence stocks have had
  a widely-covered re-rating narrative (Nifty India Defence Index +20%+ YTD per press coverage,
  record FY26 defence production/exports, a large FY27 defence budget). Classification: stocks
  commonly covered as defence-theme names (Nifty India Defence Index / analyst defence-stock
  lists, verified via web search, not sector-baseline guesswork) that are also in our 209-stock
  universe - `BEL`, `HAL`, `MAZDOCK`, `COCHINSHIP` (pure-play defence PSUs), `SOLARINDS` (defence
  segment ~27% of FY26 revenue, ~42% CAGR guided through FY30), `BHARATFORG` (Kalyani Strategic
  Systems defence/aerospace crossed ~Rs1,700cr FY25 revenue). `ASHOKLEY`/`BHEL` checked and
  excluded - real but minor defence-adjacent revenue, not typically defence-index-classified.

  **Result: no significant pooled pattern, but a striking split within the theme.** Equal-weight
  Defence (6 names) vs Non-Defence (203 names) daily-return spread over the trailing 252 trading
  days (2025-09-22 to 2026-09-22): +5.16% annualized-equivalent, **NW t=0.30 - not significant**,
  and Defence only "won" 45% of days. The pooled number hides real dispersion: `BHARATFORG`
  +58.5% and `SOLARINDS` +36.1% over the year, but `BEL` -2.1%, `HAL` -1.4%, `MAZDOCK` -25.6%,
  `COCHINSHIP` -26.5% - the *opposite* of a monolithic rally. Quarter-by-quarter cumulative
  spread confirms this isn't even a steady trend: Defence *underperformed* by -6.8% through the
  first quarter of the trailing year, was still behind at the halfway mark (-0.6%), and only
  pulled ahead in the most recent two quarters (+3.2%, then +3.8% today) - consistent with a
  market rotation *out of* already-re-rated PSU defence primes and *into* earlier-stage private
  defence diversifiers, not a sector-wide continuation of the widely-reported rally. Confound
  note: the Defence group skews toward higher beta/size/liquidity than the rest of the universe
  (all z-scored ~0.1-0.4 above the non-defence average), but with only 6 names the whole result is
  really a 2-stock story (`BHARATFORG`/`SOLARINDS`), not a diversified group effect - too small a
  sample to treat as a systematic, tradable factor either way. Not wired into any factor; no new
  module built (a pure descriptive check, not a factor-construction exercise).
- **Oil-price momentum predicting (Oil+Auto+Petrochem) vs. the rest** - the first outright
  statistically significant, Bonferroni-robust finding of the whole alpha-research pass, though
  not the grouping originally asked for. Method: reuses `regression/macro_predictability.py`'s
  existing lagged-monthly-regression machinery (built for the earlier oil/USD-INR/gold/US10yr
  check) with Brent crude (`BZ=F` via `yfinance`) trailing-3-month momentum as the predictor and
  a thematic portfolio spread as the target instead of a style factor: Group A = 30 oil-price-
  sensitive names (Oil: `BPCL`/`HINDPETRO`/`PETRONET`/`IGL`/`GAIL`/`ONGC`/`IOC`; Auto:
  `BAJAJ-AUTO`/`HEROMOTOCO`/`ASHOKLEY`/`M&M`/`MARUTI`/`TVSMOTOR`/`EICHERMOT`/`MRF`/`APOLLOTYRE`/
  `BALKRISIND`/`EXIDEIND`/`BHARATFORG`; Petrochem: `ASIANPAINT`/`BERGEPAINT`/`SUPREMEIND`/`SRF`/
  `UPL`/`AARTIIND`/`DEEPAKNTR`/`NAVINFLUOR`/`VINATIORGA`/`RELIANCE`/`TATACHEM`) vs. Group B (the
  other 179 stocks).

  **The combined grouping the request asked for is only marginal** (3-month oil momentum ->
  next-month spread: slope=-0.0159, NW t=-1.51, not significant; the simpler 1-month version:
  t=-1.88, p=0.061, borderline) - **but decomposing it by sub-theme finds the combined test was
  diluting a real signal**: the Oil sub-group alone is strongly significant (NW t=-3.12,
  p=0.0018, survives Bonferroni correction across the 5 group specifications tested,
  threshold p<0.01), Auto shows a same-direction but non-significant hint (t=-1.21), and
  Petrochem shows nothing at all (t=0.50, wrong sign). Splitting Oil further into upstream
  (`ONGC` alone, t=-2.52, p=0.012) and downstream refiners/gas-distribution
  (`BPCL`/`HINDPETRO`/`PETRONET`/`IGL`/`GAIL`/`IOC`, t=-2.76, p=0.0059) shows **both** move the
  same (negative) direction - counter to the naive refiner-margin-squeeze story alone (which
  would predict downstream negative but upstream *positive*, since a producer should benefit from
  higher realized prices). The more likely mechanism: India is a large net oil importer, so
  rising oil prices are a macro-risk signal for the whole listed Oil & Gas sector (currency/
  import-bill pressure) and/or reflect windfall-profit-tax policy risk on domestic crude
  producers introduced in 2022 - both channels would hit upstream and downstream together, which
  is what the data shows. N=113 overlapping months (~9.4 years) for each regression - a
  meaningfully longer, more robust sample than the defence-theme (N=6 stocks) or PEAD (3
  independent periods) checks above. Not yet wired into anything (this is a sector-timing signal,
  not a per-stock exposure factor - if it's worth using, it belongs in something like
  `portfolio/timing_signals.py`'s existing sector/factor-timing framework, a follow-up decision
  not made here); `regression/macro_predictability.py` needed no changes, it was already general
  enough to test an arbitrary portfolio spread, not just the built-in style factors.
- **Interest-rate momentum predicting Financials vs. Non-Financials** - same reused
  `macro_predictability.py` machinery as the oil-momentum check, but the Financials group is
  defined objectively from the model's own industry classification (exposure_panel's "Financial
  Services" industry weight >= 0.5 on the latest date, 43 symbols) rather than a hand-curated
  list. Rate proxy caveat: yfinance has no direct India government-bond-yield series (checked
  `^IN10Y`/`IN10Y=RR`/`INR10Y=RR` - all empty/404); used the US 10yr Treasury yield (`^TNX`)
  instead, the same proxy the original oil/USD-INR/gold/US10yr check used - a real channel (US
  Fed policy drives EM capital flows/FII positioning in Indian financials and correlates with RBI
  policy), but not a clean substitute for Indian rates directly.

  **Clean null result, unlike oil**: no significant relationship in any of 4 specifications (1-
  and 3-month horizons, both the existing `monthly_macro_change`'s %-change-of-level convention
  and a raw percentage-point-change cross-check): t=1.21, 0.46, 1.24, 0.62 respectively, none
  close to significant. Also checked whether a finer split hides something (the same concern
  that mattered for Oil/Auto/Petrochem): Banks (15 symbols: `AXISBANK`/`HDFCBANK`/`ICICIBANK`/
  `KOTAKBANK`/`SBIN`/etc., a rate-hike/-cut cycle should plausibly hit net-interest-margin
  lenders differently from fee/credit-spread businesses) vs Non-Bank Financials (28 symbols:
  NBFCs/insurers/AMCs/exchanges) against 3-month rate momentum - still nothing (Banks t=1.19,
  Non-Bank t=-0.14). Unlike the oil result, decomposing this one didn't uncover a hidden
  signal - it's a genuinely clean, consistent null across every cut tried. Not wired into
  anything; no new module needed (reused existing macro_predictability.py machinery unchanged).
- **USD/INR momentum, screened against the entire 209-stock universe individually, then against
  natural exporter/importer groupings** - the third genuinely significant, Bonferroni-robust
  finding of the session. Individual-stock screen (`macro_predictive_regression` run against
  each of the 209 stocks' own monthly return, not a single pre-chosen group): correctly, **zero**
  stocks survive a strict per-stock Bonferroni correction (threshold p<0.000239 across 209
  tests) - but the top hits by |NW t| are dominated by IT services names, all with the *same*
  negative sign: `PERSISTENT` (t=-2.63), `COFORGE` (t=-2.54), `TECHM` (t=-2.24), `MPHASIS`
  (t=-2.17), `INFY` (t=-1.98), `WIPRO` (t=-1.95/-1.77 across specs) - a consistent pattern across
  6 different IT names is a much stronger signal than any one name's individual p-value suggests,
  since pooling correlated names into one group test recovers the power the per-stock Bonferroni
  correction sacrifices.

  Pooling IT exporters into one group confirms it cleanly: **IT_Exporters (9 symbols: `COFORGE`/
  `LTIM`/`LTTS`/`MPHASIS`/`PERSISTENT`/`SONATSOFTW`/`TATAELXSI`/`KPITTECH`/`MAPMYINDIA`) minus the
  rest of the universe, vs. 3-month USD/INR momentum: slope=-0.68, NW t=-2.94, p=0.0033** -
  survives Bonferroni across the 4 group specifications tested (threshold p<0.0125). The sign is
  the **opposite of the textbook "rupee depreciation helps IT exporters" story** (more INR
  revenue per dollar billed) - rising USD/INR (rupee weakening) momentum predicts IT stocks
  *underperforming* the broader market the following month. Most likely explanation: rupee
  depreciation is typically a *symptom* of broader FII (foreign institutional investor)
  risk-off/outflow episodes from Indian equities, and IT services stocks - large, liquid, heavily
  FII-owned - are often sold hardest in exactly those episodes, swamping the smaller currency-
  translation benefit (not confirmed directly here - no FII-ownership data in this project - but
  the sign and magnitude are consistent with that mechanism and inconsistent with a naive
  currency-translation-only story).

  This did **not** generalize the way "oil sensitivity" generalized to both upstream and
  downstream: Pharma_Exporters (9 symbols, same "exporter" logic) showed nothing (t=0.99,
  p=0.32) - pharma stock moves are more FDA-approval/pipeline-driven than currency-driven, and
  export revenue share varies more across pharma names than it does across pure-play IT
  services. Import-heavy Oil (6 symbols, USD-denominated crude/gas costs) also showed nothing
  against *currency* momentum specifically (t=0.58) - contrast with the same group's real,
  significant sensitivity to *oil-price* momentum found earlier. Combining Exporters (IT+Pharma,
  18 symbols) minus Importers (6 symbols) into one spread *diluted* the clean IT-only signal back
  down to non-significance (t=-1.43) - the same "combining themes hides the real signal" pattern
  as Oil/Auto/Petrochem and Defence above. Not wired into anything; no new module needed.

  **Robustness check (was this just two coincidentally-trending series?)**: IT services has had
  a real secular relative decline since 2022 - plausibly the widely-discussed "AI disruption"
  narrative for outsourced IT services (cumulative IT-minus-Rest spread peaked at +241% in early
  2022, and has fallen steadily since: +150.7% end-2022 -> +212.9% end-2023 -> +166.1% end-2024
  -> +125.9% end-2025 -> +63.0% as of 2026-09), while USD/INR has also trended in one direction
  essentially the whole time (64.1 at end-2017 -> 95.8 by 2026-09) - raising the obvious question
  of whether the USD/INR-momentum relationship is real, or just two unrelated trending series
  correlating by coincidence. Two checks rule that out: (1) splitting the sample at 2023-01-01,
  the relationship is significant in **both** halves - pre-2023 (2017-2022, when IT was actually
  in a strong *bull* run relative to the rest of the market, the opposite of a bear market): NW
  t=-2.37, p=0.018; 2023-onward (the actual AI-narrative-era decline): NW t=-2.19, p=0.028 - a
  relationship present during IT's bull run cannot be an artifact of IT's later bear market. (2)
  Adding an explicit linear time trend to the regression leaves the USD/INR-momentum coefficient
  essentially unchanged (-0.60 vs. -0.68 without the trend, t=-2.98 vs. -2.94) - if this were
  spurious co-trending, an explicit trend term should have absorbed most of the effect, and it
  didn't. Conclusion: both things are real and separate - IT has a genuine secular AI-linked
  relative decline since 2022 (a *level/trend* story), and independently of that trend, IT's
  month-to-month returns still show a robust negative sensitivity to USD/INR momentum in both
  the bull-run and bear-market eras alike (a *timing* story on top of the trend, not instead of
  it).

  **Cleaner isolation, using a real AI-narrative proxy instead of a generic linear trend**:
  a linear time trend is a crude stand-in for "the AI effect" - it assumes a single constant
  slope for the whole sample, which the narrative almost certainly didn't follow. A sharper
  control: **Accenture (`ACN`)**, the closest global peer to TCS/Infosys/Wipro's business model
  (IT/consulting services), which discusses GenAI's impact on services demand every earnings
  call, but is US-domiciled, reports in USD, and has ~zero direct USD/INR exposure - so its own
  return is a clean proxy for "how is the global IT-services/AI narrative moving this month,"
  with the currency channel differenced out. Adding ACN's own monthly return as a control:
  - ACN's return is itself a real, strongly significant explainer of the India IT-minus-Rest
    spread (coef=0.31, t=4.87, p<0.0001, correlation 0.43) - confirming it's a meaningful AI/
    global-IT-narrative proxy, not a placebo.
  - **USD/INR momentum's coefficient barely moves**: -0.575 (t=-2.78, p=0.0054) with the ACN
    control vs. -0.683 (t=-2.94, p=0.0033) without it - a small reduction in magnitude, not a
    collapse. R² roughly quadruples (0.060 -> 0.230) once ACN is added, but essentially all of
    that extra explanatory power comes from ACN itself, not from stealing significance away from
    USD/INR.
  - Adding NIFTY 50's own return as a third control (ruling out generic India-market-beta as yet
    another confound) changes nothing further - Nifty return is not significant at all (t=-0.07),
    and USD/INR momentum stays at t=-2.76, p=0.0058.
  - USD/INR momentum and ACN's return are barely correlated with each other (-0.09) - they are
    two close-to-orthogonal predictors, each carrying independent information, not two proxies
    for the same underlying phenomenon.

  **Conclusion**: the AI-narrative effect (proxied by ACN) and the USD/INR-momentum effect are
  separate, additive, largely independent drivers of Indian IT's relative performance - not the
  same signal wearing two names. Isolating one doesn't explain away the other.

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
- Public GitHub remote: https://github.com/dpramanik92/factor_models (`origin`). `data/`,
  `fundamentals/`, `model_data/`, and `output/` are git-ignored; only code is committed - never
  push anything from those directories, and re-check for secrets/credentials before any push
  given the repo is public.
