"""Central configuration: paths and model parameters.

Every module reads paths from here rather than hardcoding them, per
.claude/rules/data-boundaries.md.
"""

from __future__ import annotations

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

FACTOR_RETURNS_FILE = OUTPUT_DIR / "factor_returns_daily.csv"
R2_DAILY_FILE = OUTPUT_DIR / "r2_daily.csv"
RESIDUALS_LONG_FILE = OUTPUT_DIR / "residuals_long.parquet"
SUMMARY_FILE = OUTPUT_DIR / "summary.csv"
FACTOR_COVARIANCE_FILE = OUTPUT_DIR / "factor_covariance_latest.csv"
SPECIFIC_RISK_FILE = OUTPUT_DIR / "specific_risk_latest.csv"
VALIDATION_REPORT_FILE = OUTPUT_DIR / "validation_report.md"

INDUSTRY_EXPOSURE_XLSX = DATA_DIR / "nse_200_industry_exposure.xlsx"

# --- Model parameters --------------------------------------------------------

#: Calendar days between an Indian fiscal year-end and when its audited annual
#: results become public (SEBI LODR filing deadline) - fundamentals are not
#: forward-filled into the daily panel until this many days after FY-end.
REPORTING_LAG_DAYS = 60

#: Flat annual risk-free rate used to convert raw returns to excess returns,
#: taken from data/nifty50_benchmark_stats.csv's Sharpe_rf5.5pct column.
RF_ANNUAL = 0.055
TRADING_DAYS_PER_YEAR = 252

#: Trailing-return momentum window and the most-recent-month gap excluded
#: from it (standard Barra/academic momentum convention).
MOMENTUM_LOOKBACK_DAYS = 252
MOMENTUM_LAG_DAYS = 21

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
]

ALL_FACTORS = INDUSTRY_FACTORS + STYLE_FACTORS


def ensure_output_dirs() -> None:
    """Create model_data/, output/, and output/plots/ if they don't exist."""
    MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
