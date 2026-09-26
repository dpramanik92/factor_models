import pandas as pd

from factor_model.factors import psu as psu_module
from factor_model.factors.psu import compute_psu_flag


def test_compute_psu_flag_broadcasts_static_classification(monkeypatch, tmp_path):
    flags_csv = tmp_path / "psu_flags.csv"
    pd.DataFrame({"Symbol": ["AAA", "BBB", "CCC"], "is_psu": [1, 0, 1]}).to_csv(flags_csv, index=False)
    monkeypatch.setattr(psu_module.config, "MODEL_DATA_DIR", tmp_path)

    dates = pd.bdate_range("2024-01-01", periods=5)
    result = compute_psu_flag(["AAA", "BBB", "CCC"], dates)

    assert (result.loc[result["Symbol"] == "AAA", "psu_flag"] == 1).all()
    assert (result.loc[result["Symbol"] == "BBB", "psu_flag"] == 0).all()
    assert (result.loc[result["Symbol"] == "CCC", "psu_flag"] == 1).all()
    # Constant across every date - no rolling window, no point-in-time lag.
    assert result.groupby("Symbol")["psu_flag"].nunique().eq(1).all()


def test_compute_psu_flag_defaults_unknown_symbol_to_zero(monkeypatch, tmp_path):
    flags_csv = tmp_path / "psu_flags.csv"
    pd.DataFrame({"Symbol": ["AAA"], "is_psu": [1]}).to_csv(flags_csv, index=False)
    monkeypatch.setattr(psu_module.config, "MODEL_DATA_DIR", tmp_path)

    dates = pd.bdate_range("2024-01-01", periods=3)
    result = compute_psu_flag(["AAA", "ZZZ"], dates)

    assert (result.loc[result["Symbol"] == "ZZZ", "psu_flag"] == 0).all()
