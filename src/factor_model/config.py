"""Central configuration: paths and model parameters.

Every module reads paths from here rather than hardcoding them, per
.claude/rules/data-boundaries.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Paths -----------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
FUNDAMENTALS_DIR = REPO_ROOT / "fundamentals"
MODEL_DATA_DIR = REPO_ROOT / "model_data"
OUTPUT_DIR = REPO_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"

UNIVERSE_FILE = MODEL_DATA_DIR / "universe_99.csv"
FUNDAMENTALS_ANNUAL_FILE = MODEL_DATA_DIR / "fundamentals_annual_long.parquet"
FACTOR_EXPOSURE_PANEL_FILE = MODEL_DATA_DIR / "factor_exposure_panel.parquet"
RETURNS_PANEL_FILE = MODEL_DATA_DIR / "returns_panel.parquet"


@dataclass(frozen=True)
class OutputPaths:
    """Bundles the output/ file paths so a regression/risk/validation run can be pointed at a
    subdirectory instead of output/ directly - used for period-scoped model runs (e.g.
    output/periods/2018_2021/) that must not overwrite the full-window result.
    """

    output_dir: Path

    @property
    def factor_returns_file(self) -> Path:
        return self.output_dir / "factor_returns_daily.csv"

    @property
    def r2_daily_file(self) -> Path:
        return self.output_dir / "r2_daily.csv"

    @property
    def residuals_long_file(self) -> Path:
        return self.output_dir / "residuals_long.parquet"

    @property
    def summary_file(self) -> Path:
        return self.output_dir / "summary.csv"

    @property
    def flagged_corporate_actions_file(self) -> Path:
        return self.output_dir / "flagged_corporate_actions.csv"

    @property
    def factor_covariance_file(self) -> Path:
        return self.output_dir / "factor_covariance_latest.csv"

    @property
    def specific_risk_file(self) -> Path:
        return self.output_dir / "specific_risk_latest.csv"

    @property
    def validation_report_file(self) -> Path:
        return self.output_dir / "validation_report.md"

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / "plots"


DEFAULT_OUTPUT_PATHS = OutputPaths(OUTPUT_DIR)

# Kept as plain constants too (equivalent to DEFAULT_OUTPUT_PATHS.*) since most call sites don't
# need period-scoping and importing the constant directly is simpler.
FACTOR_RETURNS_FILE = OUTPUT_DIR / "factor_returns_daily.csv"
R2_DAILY_FILE = OUTPUT_DIR / "r2_daily.csv"
RESIDUALS_LONG_FILE = OUTPUT_DIR / "residuals_long.parquet"
SUMMARY_FILE = OUTPUT_DIR / "summary.csv"
FACTOR_COVARIANCE_FILE = OUTPUT_DIR / "factor_covariance_latest.csv"
SPECIFIC_RISK_FILE = OUTPUT_DIR / "specific_risk_latest.csv"
VALIDATION_REPORT_FILE = OUTPUT_DIR / "validation_report.md"

#: Named sub-periods for comparative regression runs (see pipeline.run_period /
#: cli.py's run-periods command). Saved under output/periods/<label>/, never overwriting the
#: full-window result at the output/ top level.
MODEL_PERIODS: dict[str, tuple[str, str]] = {
    "2018_2021": ("2018-01-01", "2021-01-01"),
    "2021_2023": ("2021-01-01", "2023-01-01"),
    "2023_present": ("2023-01-01", "2100-01-01"),
}

INDUSTRY_EXPOSURE_XLSX = DATA_DIR / "nse_200_industry_exposure.xlsx"

# --- Model parameters --------------------------------------------------------

#: Calendar days between an Indian fiscal year-end and when its audited annual
#: results become public (SEBI LODR filing deadline) - fundamentals are not
#: forward-filled into the daily panel until this many days after FY-end.
REPORTING_LAG_DAYS = 60

#: SEBI LODR's quarterly (non-Q4) financial-results filing deadline - shorter than the annual
#: REPORTING_LAG_DAYS since quarterly results are unaudited/limited-review, not a full audit.
#: Used only by the exploratory quarterly-SUE/PEAD check (factors/pead.py) - not part of the
#: main annual-frequency pipeline.
QUARTERLY_REPORTING_LAG_DAYS = 45

#: Flat annual risk-free rate used to convert raw returns to excess returns,
#: taken from data/nifty50_benchmark_stats.csv's Sharpe_rf5.5pct column.
RF_ANNUAL = 0.055
TRADING_DAYS_PER_YEAR = 252

#: Trailing-return momentum window and the most-recent-month gap excluded
#: from it (standard Barra/academic momentum convention).
MOMENTUM_LOOKBACK_DAYS = 252
MOMENTUM_LAG_DAYS = 21

#: Trailing window for the rolling market-model regression (beta and idiosyncratic vol are
#: both computed from this - see factors/market_model.py).
BETA_IDIO_VOL_LOOKBACK_DAYS = 252

#: Trailing window for the liquidity factor's average daily share turnover
#: (volume / point-in-time shares outstanding) - see factors/liquidity.py.
LIQUIDITY_LOOKBACK_DAYS = 252

#: Rank cutoff for the top100_flag indicator factor (factors/top100.py) - 1 if a stock ranks in
#: the top N by market cap that date, else 0.
TOP100_N = 100

#: A date is treated as an unflagged exchange holiday (forward-filled, not treated as a genuine
#: zero-volume trading day) if at least this fraction of the cross-section shows zero or NaN
#: volume - the same signature already used to detect degenerate return days
#: (DEGENERATE_RETURN_STD_THRESHOLD), just measured on volume instead of returns. Verified
#: against real data: known unflagged-holiday dates show exactly 100% zero/NaN across all 209
#: symbols, vs. 0% on an ordinary trading day - 0.5 is a wide margin either side of that gap.
LIQUIDITY_HOLIDAY_ZERO_FRACTION_THRESHOLD = 0.5

#: Cross-sectional winsorization bounds (percentile method) applied before
#: z-scoring every continuous style factor.
WINSOR_LOWER_PCT = 0.01
WINSOR_UPPER_PCT = 0.99

#: Minimum number of stocks with full factor coverage required to run the
#: cross-sectional regression on a given date; dates below this are skipped.
MIN_N_PER_DAY = 40

#: A regression date is skipped as "degenerate" (rather than fit) if the cross-sectional
#: standard deviation of that day's returns falls below this. This catches exchange holidays
#: that a plain business-day calendar (freq="B") doesn't know about, where the data source
#: just repeats the prior close for every ticker - every stock shows an identical 0.0% return,
#: which produces a zero (or near-zero) total-sum-of-squares and an undefined/exploding R2,
#: and contributes no genuine cross-sectional information to the regression anyway.
#: 1e-6 is ~10,000x smaller than a typical single-stock daily return std (~1-3%), so this only
#: catches genuinely flat/stale days, not just unusually quiet trading days. An earlier, much
#: stricter threshold (1e-10) missed several holiday days where floating-point noise from
#: dividing already-identical prices left a std a few orders of magnitude above exact zero but
#: still functionally degenerate (observed producing R2 as extreme as -387).
DEGENERATE_RETURN_STD_THRESHOLD = 1e-6

#: Single-day |return| threshold above which an observation is treated as a
#: likely corporate action (demerger, unadjusted split, etc.) rather than a
#: genuine return, and excluded from that day's cross-sectional regression.
#: We do not attempt to reconstruct corporate-action-adjusted total returns
#: (e.g. splicing in spun-off-entity value) - see CLAUDE.md "Known
#: limitations". Flagged observations are logged for manual review, not
#: silently dropped.
CORPORATE_ACTION_RETURN_THRESHOLD = 0.40

#: EWMA half-lives (trading days) for factor covariance and specific risk.
EWMA_HALFLIFE_FACTOR = 90
EWMA_HALFLIFE_SPECIFIC = 90

#: Industry exposure buckets, in the order used throughout the model.
INDUSTRY_FACTORS = [
    "Financial Services",
    "Consumer Discretionary",
    "Materials",
    "Energy & Utilities",
    "Healthcare",
    "Consumer Staples",
    "Industrials",
    "Information Technology",
    "Communication Services",
    "Real Estate",
]

#: Style factors, in the order used throughout the model.
STYLE_FACTORS = [
    "beta",
    "idio_vol",
    "momentum",
    "size",
    "value",
    "leverage",
    "quality",
    "capex",
    "growth",
    "profitability",
    "liquidity",
]

#: Binary 0/1 indicator factors - unlike STYLE_FACTORS, these are never winsorized/z-scored
#: (factors/panel.py's standardization loop iterates STYLE_FACTORS only), passed straight into
#: the regression as raw dummies, same convention as the industry weight columns.
#:
#: top100_flag/psu_flag: added because the factors driving the largest ~100 stocks were observed
#: to behave differently from the rest, and because government ownership (PSU status) plausibly
#: carries its own return premium/discount independent of the other factors - see
#: factors/top100.py and factors/psu.py. Each is a level effect only (average return
#: premium/discount for segment membership).
#:
#: size_decile_2..size_decile_10 (dummy-encoded from factors/size.py's size_decile column via
#: add_size_decile_dummies) was tried here and reverted: combining it with the continuous
#: z-scored size term and top100_flag in one pooled regression made VIF explode (size=30.8,
#: top100_flag=31.3, size_decile_10=48.5 - they're all encodings of the same underlying market-
#: cap ranking) *and* barely changed the segment-level residual bias this was meant to fix (NW t
#: ~19/-13, versus ~21/-19 for the interaction-term attempt before it). Three single-pooled-
#: regression fixes for the large-cap/small-cap structural break have now failed in a row
#: (main-effect dummy alone, interaction terms, decile dummies) - see CLAUDE.md's "Known
#: limitations" and pipeline.run_segment (config.SEGMENT_DESIGN_COLS) for the fully-separate-
#: per-segment-regressions approach now in active use instead.
INDICATOR_FACTORS = [
    "top100_flag",
    "psu_flag",
]

ALL_FACTORS = INDUSTRY_FACTORS + STYLE_FACTORS + INDICATOR_FACTORS

#: Both a single size x top100_flag interaction column, and the full style_factor x top100_flag
#: set (11 columns, one per style factor), were tried and rejected as fixes for the large-cap/
#: small-cap structural break (see SEGMENT_DESIGN_COLS and CLAUDE.md's "Known limitations"). The
#: full set made the segment-level residual bias *worse* (NW t up to ~21 from ~6); the single,
#: more targeted size-only interaction improved the *pooled* residual-alpha metric (0.42% -> 0.07%
#: annualized, FAIL -> PASS) but made the *segment-level* bias even worse still (NW t 17.3/-7.8 ->
#: 20.2/-17.6) - the same pooled-metric-masks-a-worse-split pattern the segment diagnostic exists
#: to catch, and a reminder that a shared WLS fit (weighted by sqrt(market_cap)) keeps letting the
#: largest few names dominate regardless of how many extra columns are added to the same
#: regression. Replaced by fully separate per-segment regressions instead - see
#: pipeline.run_segment / cli.py's `run-segments` command.
#:
#: Design columns for a *segment-specific* regression (top100-only, or non-top100-only): the
#: industry weights and style factors are unchanged, but top100_flag and any interaction with it
#: would be constant (or entirely absent) within a single segment - a constant column equal to 1
#: is exactly collinear with the industry weights (which already sum to 1 per row), so including
#: it would make the design matrix singular. psu_flag is kept: it has genuine variation within
#: both segments (17 of 100 top100 names are PSUs, 11 of 109 non-top100 names are).
SEGMENT_DESIGN_COLS = INDUSTRY_FACTORS + STYLE_FACTORS + ["psu_flag"]

#: The two segment labels `pipeline.run_segment`/`run-segments` produces, and their `top100_flag`
#: value - shared by portfolio/segmented_risk_model.py so both sides of the join always agree on
#: naming/order.
SEGMENT_LABELS = ["top100", "rest"]
SEGMENT_TOP100_FLAG_VALUE = {"top100": 1.0, "rest": 0.0}

# --- Portfolio optimization (long-only mean-variance, see portfolio/) -------

#: Number of points traced along the long-only efficient frontier (target-return
#: parameterization - see portfolio/optimize.py).
FRONTIER_N_POINTS = 25

#: Long-only MVO position limits: no single stock above this weight, no single industry
#: (aggregated via each stock's fractional industry exposure) above this weight - a known
#: mitigation for long-only MVO's tendency to over-concentrate in whichever name has the most
#: optimistic (and least reliable) expected-return estimate (see the Single-name concentration
#: validation check, which flagged Max-Sharpe holdings up to 17.7% pre-cap). The industry cap
#: stays a hard bound; the stock cap is enforced as a SOFT quadratic penalty (see
#: STOCK_WEIGHT_SOFT_PENALTY_COEF and portfolio/optimize.py's module docstring) rather than a
#: hard bound - relaxed after the hard 10% cap was found to force the optimizer away from
#: concentrated positions with no way to weigh the tradeoff.
MAX_STOCK_WEIGHT = 0.10
MAX_INDUSTRY_WEIGHT = 0.20

#: Penalty coefficient for the soft stock-weight cap: added to the (daily variance-units)
#: objective as `coef * sum(relu(w - MAX_STOCK_WEIGHT)^2)`. Calibrated empirically against the
#: real fitted model (not just synthetic data): 0.01 turned out indistinguishable from a hard
#: cap on this portfolio's actual Sigma/mu (Max-Sharpe's top holding moved from 10.0% to just
#: 10.2%); 0.001 lets the same holding reach ~11.5% - visibly relaxed but still well short of the
#: ~14%+ it would reach fully unconstrained - while GMV (which never wanted much concentration in
#: the first place) barely moves. Re-check this calibration if Sigma/mu change materially.
STOCK_WEIGHT_SOFT_PENALTY_COEF = 0.001

#: Penalty coefficient for the segmented model's group-level soft cap on total "rest"-segment
#: weight (portfolio/optimize.py's group_exposure/group_max_weight/group_weight_penalty,
#: portfolio/segmented_risk_model.py's build_group_exposure) - added to the objective as
#: `coef * relu(w @ rest_membership - REST_SEGMENT_MAX_WEIGHT)^2`. This is a *different*
#: quantity from STOCK_WEIGHT_SOFT_PENALTY_COEF (a per-stock excess vs. a whole-group excess that
#: sums many stocks' weight), so the same coefficient value doesn't bind the same way - calibrated
#: empirically against the real fitted segmented model (Max-Sharpe, no industry cap): with no
#: group cap at all, Max-Sharpe puts 88.7% of the portfolio in the rest segment (see the
#: "remove the industry cap" experiment in CLAUDE.md); coef=0.001 only pulls that down to 57%
#: (too weak to be a meaningful signal); coef=0.05-0.5 all land at ~20.0-20.2%, indistinguishable
#: from a hard cap; coef=0.01 lands at ~22.0% - a genuinely soft, visibly-relaxed-but-still-
#: binding middle ground, the same shape of tradeoff STOCK_WEIGHT_SOFT_PENALTY_COEF's own
#: calibration found. Re-check this if Sigma/mu change materially.
REST_SEGMENT_MAX_WEIGHT = 0.20
REST_SEGMENT_WEIGHT_SOFT_PENALTY_COEF = 0.01

#: Long-short portfolio variant (portfolio/optimize.py's minimize_variance_long_short /
#: trace_efficient_frontier_long_short): gross exposure cap sum(|w|) <= MAX_LEVERAGE, with net
#: exposure still fully invested at 1.0 (a "120/20"-style book: up to (MAX_LEVERAGE-1)/2 of
#: capital financed short to fund the same amount of extra long exposure). Individual stock and
#: industry caps reuse MAX_STOCK_WEIGHT/MAX_INDUSTRY_WEIGHT, applied symmetrically (a short can
#: be as large in magnitude as a long).
MAX_LEVERAGE = 1.2

#: Single-name concentration validation check thresholds (portfolio/validation.py). Since
#: MAX_STOCK_WEIGHT is now a soft target rather than a hard bound, exceeding it a little is an
#: accepted, intended outcome, not a defect - the check now WARNs between the soft target and
#: this hard-fail threshold, and only FAILs above it.
PORTFOLIO_CONCENTRATION_HARD_FAIL_WEIGHT = 0.20

#: A single stock's weight in a long-only MVO solution above this is flagged by the
#: portfolio validation suite as a concentration risk. Set equal to MAX_STOCK_WEIGHT now that
#: the optimizer enforces that cap directly - the check's job is now "did the hard cap actually
#: hold" (e.g. after the sum-to-1 renormalization step) rather than an independent soft flag.
PORTFOLIO_CONCENTRATION_FLAG_WEIGHT = MAX_STOCK_WEIGHT

PORTFOLIO_OUTPUT_DIR = OUTPUT_DIR / "portfolio"
PORTFOLIO_PLOTS_DIR = PORTFOLIO_OUTPUT_DIR / "plots"
PORTFOLIO_VALIDATION_REPORT_FILE = PORTFOLIO_OUTPUT_DIR / "portfolio_validation_report.md"
PORTFOLIO_BACKTEST_OUTPUT_DIR = PORTFOLIO_OUTPUT_DIR / "backtest"
PORTFOLIO_BACKTEST_REPORT_FILE = PORTFOLIO_BACKTEST_OUTPUT_DIR / "backtest_report.md"

#: Portfolio built against the segmented (top100/rest) risk model instead of the pooled one -
#: portfolio/segmented_risk_model.py + portfolio/pipeline.py's run_segmented_portfolio_*
#: functions. Deliberately a sibling directory, never PORTFOLIO_OUTPUT_DIR itself: the pooled-
#: model portfolio result is kept alongside this one for comparison, not replaced by it.
PORTFOLIO_SEGMENTS_OUTPUT_DIR = OUTPUT_DIR / "portfolio_segments"
PORTFOLIO_SEGMENTS_PLOTS_DIR = PORTFOLIO_SEGMENTS_OUTPUT_DIR / "plots"
PORTFOLIO_SEGMENTS_VALIDATION_REPORT_FILE = PORTFOLIO_SEGMENTS_OUTPUT_DIR / "portfolio_validation_report.md"
PORTFOLIO_SEGMENTS_BACKTEST_OUTPUT_DIR = PORTFOLIO_SEGMENTS_OUTPUT_DIR / "backtest"
PORTFOLIO_SEGMENTS_BACKTEST_REPORT_FILE = PORTFOLIO_SEGMENTS_BACKTEST_OUTPUT_DIR / "backtest_report.md"

#: Pooled-model portfolio run with portfolio/macro_timing.py's tilts enabled (`--macro-timing`) -
#: a distinct sibling directory to PORTFOLIO_OUTPUT_DIR, never the same one, so turning the flag
#: on or off never silently overwrites the other run's result (the same "add, don't overwrite"
#: convention config.PORTFOLIO_SEGMENTS_OUTPUT_DIR's siblings already follow).
PORTFOLIO_MACRO_TIMING_OUTPUT_DIR = OUTPUT_DIR / "portfolio_macro_timing"
PORTFOLIO_MACRO_TIMING_PLOTS_DIR = PORTFOLIO_MACRO_TIMING_OUTPUT_DIR / "plots"
PORTFOLIO_MACRO_TIMING_VALIDATION_REPORT_FILE = PORTFOLIO_MACRO_TIMING_OUTPUT_DIR / "portfolio_validation_report.md"
PORTFOLIO_MACRO_TIMING_BACKTEST_OUTPUT_DIR = PORTFOLIO_MACRO_TIMING_OUTPUT_DIR / "backtest"
PORTFOLIO_MACRO_TIMING_BACKTEST_REPORT_FILE = PORTFOLIO_MACRO_TIMING_BACKTEST_OUTPUT_DIR / "backtest_report.md"

#: Newey-West |t| thresholds used to shrink each factor's expected daily return toward zero
#: (portfolio/expected_returns.py). A factor with |NW_t| (computed over the full historical
#: series - "was this ever a real signal") at or below T_MIN is treated as statistically
#: indistinguishable from zero and gets zero weight; a factor at or above T_FULL gets full
#: weight; in between, the weight ramps linearly. Replaces an earlier Sharpe x current-EWMA-vol
#: formula that both used no significance filter and double-counted risk as return
#: (Quant_analyst review, 2026-09).
EXPECTED_RETURN_SHRINKAGE_T_MIN = 1.0
EXPECTED_RETURN_SHRINKAGE_T_FULL = 3.0

#: Half-life (trading days) for the EWMA-recency-weighted mean used as each factor's expected-
#: return *magnitude* (the significance shrinkage above still uses the full-sample NW t-stat -
#: recency answers "what's the premium worth right now", significance answers "was this ever a
#: real signal at all"). Kept as its own constant rather than reusing EWMA_HALFLIFE_FACTOR even
#: though it's the same value today, since the two answer different questions and may want to
#: diverge later. Added after real-data verification showed the old flat full-sample mean was
#: dominated by stale multi-year rallies (e.g. Communication Services/Industrials) that had
#: already reversed in the trailing 252 days - see portfolio/validation.py's
#: test_expected_return_vs_realized_stock_returns.
EXPECTED_RETURN_EWMA_HALFLIFE_DAYS = 90

#: Trailing window (trading days) for the static-weight portfolio backtest used in both the
#: realized-vs-predicted-risk validation check and the cumulative-performance plot.
PORTFOLIO_BACKTEST_LOOKBACK_DAYS = 252

# --- Macro-momentum timing tilts (portfolio/macro_timing.py) ----------------------------------
#: Two group-specific expected-return tilts found via alpha research (CLAUDE.md's "Alpha
#: research" section) and, unlike everything else there, actually wired into the optimizer:
#: Oil-sector returns' sensitivity to oil-price momentum, and IT-exporter returns' sensitivity to
#: USD/INR momentum. Both survived Bonferroni correction across every group specification tried,
#: and the USD/INR one specifically survived being isolated from a real "AI-narrative" proxy
#: (Accenture's own return) and a market-beta control (NIFTY 50's own return) - see CLAUDE.md.
#: The regression slope itself is *never* hardcoded from that research run - it's re-estimated
#: fresh from an expanding window of history every time compute_macro_timing_tilt is called
#: (including at every rebalance date inside the walk-forward backtest), the same point-in-time
#: discipline portfolio/expected_returns.py's own EWMA/shrinkage estimates already follow, so an
#: early backtest period never uses a slope that could only be known with later data.
OIL_TICKER = "BZ=F"  # Brent crude futures (yfinance)
USDINR_TICKER = "INR=X"  # USD/INR spot (yfinance)

OIL_GROUP_SYMBOLS = ["BPCL", "HINDPETRO", "PETRONET", "IGL", "GAIL", "ONGC", "IOC"]
IT_EXPORTER_SYMBOLS = ["COFORGE", "LTIM", "LTTS", "MPHASIS", "PERSISTENT", "SONATSOFTW", "TATAELXSI", "KPITTECH", "MAPMYINDIA"]

#: Trailing months of oil-price / USD-INR change compounded into the momentum reading fed to
#: each group's own predictive regression - matches the 3-month window the original research used.
OIL_MOMENTUM_LOOKBACK_MONTHS = 3
USDINR_MOMENTUM_LOOKBACK_MONTHS = 3

#: macro_predictive_regression's own minimum-overlapping-months requirement (regression/
#: macro_predictability.py) - reproduced here only so callers can check "will a tilt even be
#: computable yet" without importing that module's internals.
MACRO_TIMING_MIN_MONTHS = 24

#: Trading days per calendar month, used only to convert a monthly-frequency regression slope's
#: implied monthly return contribution into the daily-return units portfolio/expected_returns.py
#: works in - matches factors/momentum.py's MOMENTUM_LAG_DAYS (the same "one month" convention
#: used everywhere else in this model).
TRADING_DAYS_PER_MONTH = 21

#: Number of efficient-frontier points traced at *each monthly rebalance* of the walk-forward
#: backtest (portfolio/backtest.py) - deliberately fewer than FRONTIER_N_POINTS (25, used for
#: the single current-date optimization) purely for runtime: a multi-year monthly backtest
#: re-solves the frontier at every rebalance date, so this trades a coarser frontier for a
#: backtest that finishes in minutes instead of tens of minutes. Doesn't materially change which
#: point ends up picked as max-Sharpe.
BACKTEST_FRONTIER_N_POINTS = 8

def ensure_output_dirs() -> None:
    """Create model_data/, output/, and output/plots/ if they don't exist."""
    MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
