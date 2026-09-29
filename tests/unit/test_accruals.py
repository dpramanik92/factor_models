import numpy as np
import pandas as pd

from factor_model import config
from factor_model.factors.accruals import compute_accruals


def test_compute_accruals_expands_to_daily_after_reporting_lag():
    fy_end = pd.Timestamp("2024-03-31")
    annual_ratios = pd.DataFrame(
        {"Symbol": ["AAA"], "FiscalYearEnd": [fy_end], "Accruals": [0.05]}
    )
    business_day_index = pd.bdate_range("2024-01-01", "2024-08-31")

    result = compute_accruals(annual_ratios, business_day_index)

    before_lag = result[(result["Symbol"] == "AAA") & (result["Date"] == fy_end + pd.Timedelta(days=1))]
    after_lag = result[
        (result["Symbol"] == "AAA") & (result["Date"] == fy_end + pd.Timedelta(days=config.REPORTING_LAG_DAYS + 5))
    ]

    assert before_lag.empty or before_lag.iloc[0]["accruals"] != 0.05 or pd.isna(before_lag.iloc[0]["accruals"])
    assert np.isclose(after_lag.iloc[0]["accruals"], 0.05)
