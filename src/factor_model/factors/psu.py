"""PSU (Public Sector Undertaking - majority Indian-government-owned) indicator factor: a static,
hand-classified 0/1 flag per company, expanded to a daily panel (unlike every other factor here,
ownership status doesn't change day to day, so there's no rolling window or point-in-time lag to
apply - just broadcast the one flag across every date).

Classification (model_data/psu_flags.csv) uses a narrow, consistent rule: 1 only if the Government
of India (directly, or via a recognized Central Public Sector Enterprise holding structure) holds
majority equity and the company is a nationalized bank or officially recognized CPSE - not merely
"a PSU holds a stake in it" (e.g. SBI Life, SBI Cards, and LIC Housing Finance are majority-owned
by PSU parents but are NOT themselves classified as PSUs here, matching standard market-index
convention such as the Nifty PSE Index). See that file's `rationale` column for the reasoning
behind every single symbol, including the handful of genuinely borderline cases (CONCOR, PETRONET,
HINDZINC) called out there for possible re-verification as India's PSU disinvestment program
continues.
"""

from __future__ import annotations

import pandas as pd

from factor_model import config


def compute_psu_flag(universe: list[str], business_day_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Returns long (Date, Symbol, psu_flag) - a constant 0/1 per symbol across every date.
    Not z-scored (factors/panel.py's standardization loop only touches config.STYLE_FACTORS) -
    fed into the regression as a raw dummy, same convention as the industry weight columns.
    """
    flags = pd.read_csv(config.MODEL_DATA_DIR / "psu_flags.csv").set_index("Symbol")["is_psu"]
    flags = flags.reindex(universe).fillna(0).astype(int)

    wide = pd.DataFrame(
        {symbol: flags[symbol] for symbol in universe}, index=business_day_index
    )
    wide.index.name = "Date"
    return wide.reset_index().melt(id_vars="Date", var_name="Symbol", value_name="psu_flag")
