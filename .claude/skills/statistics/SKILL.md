---
name: statistics
description: Cross-sectional WLS regression specification, standardization (winsorize/z-score) conventions, Newey-West HAC t-stats, EWMA covariance estimation, and the model-validation statistical tests used in this repo. Use whenever touching standardize.py, regression/, risk/, or validation/.
---

# Regression, standardization, and risk-model statistics conventions

## Standardization

Always **winsorize before z-scoring**, cross-sectionally (grouped by date, never across time).
Default winsorization: 1st/99th percentile (`method="percentile"`); a std-based mode
(`method="std"`, bounds `(-3, 3)`) is available for smoother factors like beta where percentile
clipping is less necessary. z-score is `(x - mean) / std` computed on the winsorized series,
NaN-safe (missing stays missing, never imputed at this stage).

Two factors are **exceptions** to plain z-scoring:
- **Size**: z-scored `log(market_cap)` is the regression exposure; a separate `size_decile`
  (D1-D10, `pd.qcut` per date) column is retained for diagnostics only (e.g. the D1-vs-D10 spread
  test), not fed into the regression as a categorical.
- **Industry**: the 10-bucket weighted exposure matrix is used as-is (weights sum to 1 per
  stock) - it is not z-scored, since it plays the role of the regression intercept.

## Cross-sectional WLS regression

Per date `t`:
- `y`: excess stock return over `(t-1, t]` (daily return minus a flat ~5.5%/year risk-free rate,
  from `RF_ANNUAL` in `config.py`)
- `X`: `[10 industry exposure weights, no separate intercept]` + `[10 z-scored style factors:
  beta, idio_vol, momentum, size, value, leverage, quality, capex, growth, profitability]`
- `weight`: `sqrt(market_cap)` (downweights small/illiquid names, standard Barra convention)
- Exposures and weights are dated `t-1`, predicting the return realized over `(t-1, t]` - never
  use same-day-as-return exposures.
- Skip (and log) any date with fewer than `MIN_N` stocks having full factor coverage. Report the
  actual first date at which full coverage is reached rather than assuming day one of the price
  history is usable (beta/idio-vol need rolling history; fundamentals need the reporting lag to
  clear).

Fit with `statsmodels.api.WLS(y, X, weights=w).fit()` per date; stack the daily coefficient
vectors into the factor return time series.

## Newey-West HAC t-stats

On each factor's daily-coefficient time series, fit an intercept-only OLS
(`coef_series ~ 1`) with `cov_type="HAC"`, `cov_kwds={"maxlags": L}`, where
`L = floor(4*(T/100)**(2/9))` (Newey-West 1994 plug-in rule). Report both the naive t-stat
(assuming i.i.d. daily coefficients) and the NW t-stat side by side - they diverge exactly when
factor returns are autocorrelated, which is expected and informative, not a bug.

## EWMA factor covariance

`factor_returns_df.ewm(halflife=EWMA_HALFLIFE, min_periods=2*EWMA_HALFLIFE).cov()`, default
half-life 90 trading days (`config.py`). Report the latest as-of-today K x K matrix as the primary
output; a full daily time series of matrices is not persisted by default (unbounded size) unless
explicitly requested via a snapshot frequency flag.

## Specific (idiosyncratic) risk

Per-stock EWMA variance of regression residuals (`output/residuals_long.parquet`), same
half-life convention as factor covariance unless configured otherwise.

## Validation statistical tests (model_reviewer)

- R²/t-stat distribution summary (mean/median/std)
- VIF (`statsmodels.stats.outliers_influence.variance_inflation_factor`) across the 10 style
  factors; flag VIF > 10
- Jarque-Bera (`scipy.stats.jarque_bera`) normality on residuals
- Breusch-Pagan (`statsmodels.stats.diagnostic.het_breuschpagan`) heteroskedasticity
- Ljung-Box autocorrelation on residuals over time, per stock
- ADF stationarity test on each factor's daily return series
- Industry-weights-sum-to-1 check (tolerance 1e-6)
- Look-ahead-bias check: exposure values must not change before `FY_end + REPORTING_LAG_DAYS`
- Size-quintile Q1-vs-Q5 return spread t-test

## Testing this logic itself

Unit tests for the regression engine should use a **synthetic recovery test**: simulate returns
from known factor exposures and known factor returns plus noise, run the regression, and assert
the recovered coefficients are close to the true ones. This is the standard way to validate a
regression engine's correctness independent of any real data quirks.
