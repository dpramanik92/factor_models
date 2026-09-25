"""Optional yfinance-based refresh of the daily price panel.

Not run automatically by the pipeline - the existing data/ price history is
already current as of this build. This is the one deliberate exception to
"data/ is read-only" (.claude/rules/data-boundaries.md): it's an explicit,
user-invoked action to extend the raw source data itself, not something the
build pipeline does on its own. Only refreshes
nse200_nifty50_daily_prices - the other ~20 derived wide files in data/
(PE, PB, leverage, etc.) are not touched by this command.
"""

from __future__ import annotations

import logging

import pandas as pd

from factor_model import config

logger = logging.getLogger(__name__)

PRICE_PARQUET = config.DATA_DIR / "nse200_nifty50_daily_prices.parquet"
PRICE_CSV_GZ = config.DATA_DIR / "nse200_nifty50_daily_prices.csv.gz"


def refresh_prices(force: bool = False) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required for refresh-prices (pip install yfinance)") from exc

    existing = pd.read_parquet(PRICE_PARQUET)
    existing.index = pd.to_datetime(existing.index)
    last_date = existing.index.max()
    today = pd.Timestamp.now().normalize()

    if not force and last_date >= today - pd.Timedelta(days=1):
        logger.info("Price data already current through %s; nothing to refresh (use --force to override)", last_date.date())
        return existing

    avail = pd.read_csv(config.DATA_DIR / "nse_200_timeseries_availability.csv")
    tickers = avail.loc[avail["Status"] == "OK", ["Symbol", "YF_Ticker"]]

    start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info("Refreshing prices from %s to today for %d tickers", start, len(tickers))

    new_data: dict[str, pd.Series] = {}
    for symbol, yf_ticker in zip(tickers["Symbol"], tickers["YF_Ticker"]):
        try:
            hist = yf.Ticker(yf_ticker).history(start=start, auto_adjust=True)
            if not hist.empty:
                new_data[symbol] = hist["Close"]
        except Exception:
            logger.exception("Failed to refresh %s (%s)", symbol, yf_ticker)

    if not new_data:
        logger.info("No new price data retrieved")
        return existing

    new_df = pd.DataFrame(new_data)
    new_df.index = pd.to_datetime(new_df.index).tz_localize(None)
    new_df.index.name = "Date"

    combined = pd.concat([existing, new_df.reindex(columns=existing.columns)])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    combined.to_parquet(PRICE_PARQUET)
    combined.to_csv(PRICE_CSV_GZ, compression="gzip")
    logger.info("Wrote refreshed price panel: %s rows through %s", len(combined), combined.index.max().date())
    return combined
