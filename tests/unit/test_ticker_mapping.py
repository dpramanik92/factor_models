from factor_model.mapping.ticker_mapping import (
    _best_fuzzy_match,
    normalize_company_name,
    strip_duplicate_suffix,
)
import pandas as pd


def test_strip_duplicate_suffix():
    assert strip_duplicate_suffix("Asian Paints (1)") == "Asian Paints"
    assert strip_duplicate_suffix("Asian Paints") == "Asian Paints"
    assert strip_duplicate_suffix("HDFC Bank (12)") == "HDFC Bank"


def test_normalize_company_name_strips_legal_suffix_and_punct():
    assert normalize_company_name("Tata Consultancy Services Ltd") == "TATA CONSULTANCY SERVICES"
    assert normalize_company_name("MAHINDRA & MAHINDRA LTD") == "MAHINDRA & MAHINDRA"
    assert normalize_company_name("M & M") == "M & M"


def test_best_fuzzy_match_finds_correct_symbol():
    reference = pd.DataFrame(
        {
            "Symbol": ["TCS", "INFY", "M&M"],
            "Company Name": ["Tata Consultancy Services", "Infosys", "Mahindra & Mahindra"],
        }
    )
    symbol, score = _best_fuzzy_match("MAHINDRA & MAHINDRA LTD", reference)
    assert symbol == "M&M"
    assert score > 0.9


def test_best_fuzzy_match_low_score_for_unrelated_name():
    reference = pd.DataFrame({"Symbol": ["TCS"], "Company Name": ["Tata Consultancy Services"]})
    symbol, score = _best_fuzzy_match("Completely Unrelated Company", reference)
    assert score < 0.5
