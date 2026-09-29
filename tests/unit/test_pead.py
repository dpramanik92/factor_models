import numpy as np
import pandas as pd

from factor_model import config
from factor_model.factors.pead import compute_quarterly_sue


def test_compute_quarterly_sue_uses_same_quarter_year_ago_not_prior_quarter():
    # Two years of quarters for one company, with a strong Q4 seasonal spike every year (and
    # mild quarter-to-quarter noise elsewhere so the rolling std is non-zero) - a naive
    # prior-quarter (lag-1) comparison would flag every Q4 as a huge "surprise" purely from
    # seasonality, but the lag-4 (same-quarter-year-ago) convention should see the *repeated* Q4
    # pattern as unsurprising, since this year's Q4 barely moved versus last year's Q4.
    quarter_ends = pd.to_datetime(
        ["2022-06-30", "2022-09-30", "2022-12-31", "2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"]
    )
    net_profit = [10.0, 11.0, 30.0, 9.0, 10.5, 11.8, 30.3]  # Q4 (Dec) always spikes to ~30
    df = pd.DataFrame({"Symbol": "AAA", "QuarterEnd": quarter_ends, "Net profit": net_profit})

    result = compute_quarterly_sue(df, min_periods=1)
    with_sue = result.dropna(subset=["SUE"]).set_index("QuarterEnd")

    # A naive lag-1 (prior-quarter) comparison would see 30.3 - 11.8 = 18.5, a huge jump for the
    # Q4 row; the lag-4 (same-quarter-year-ago) comparison instead sees YoY diffs of 0.5, 0.8,
    # 0.3 for the three most recent quarters - Q4's 0.3 is the *smallest* of the three, so its
    # standardized surprise should be the smallest in magnitude too, not the largest a naive
    # same-period-last-quarter comparison would have flagged it as.
    assert abs(with_sue.loc["2023-12-31", "SUE"]) == with_sue["SUE"].abs().min()


def test_compute_quarterly_sue_announcement_date_uses_quarterly_lag():
    quarter_ends = pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31"])
    net_profit = [10.0, 11.0, 9.0, 12.0, 15.0]
    df = pd.DataFrame({"Symbol": "AAA", "QuarterEnd": quarter_ends, "Net profit": net_profit})

    result = compute_quarterly_sue(df, min_periods=1)
    row = result[result["QuarterEnd"] == "2024-03-31"].iloc[0]

    assert row["AnnouncementDate"] == pd.Timestamp("2024-03-31") + pd.Timedelta(days=config.QUARTERLY_REPORTING_LAG_DAYS)


def test_compute_quarterly_sue_nan_before_lag_and_min_periods():
    quarter_ends = pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"])
    net_profit = [10.0, 11.0, 9.0, 12.0]
    df = pd.DataFrame({"Symbol": "AAA", "QuarterEnd": quarter_ends, "Net profit": net_profit})

    # Only one quarter has a lag-4 predecessor available at all with this short history - none of
    # them, in fact, since there are only 4 quarters total and lag=4 needs a 5th, so every SUE
    # should be NaN here.
    result = compute_quarterly_sue(df)
    assert result["SUE"].isna().all()
