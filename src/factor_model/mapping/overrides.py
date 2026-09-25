"""Hand-maintained overrides for fundamentals filename -> NSE symbol mapping.

Fuzzy matching (ticker_mapping.py) handles the vast majority of the 99
fundamentals/*.xlsx files, but a few need a human decision: renamed
companies, punctuation variants, or filenames whose real identity is
ambiguous. Maintained by the equity_researcher agent - see
.claude/agents/equity_researcher.md.

Keys are the fundamentals filename stem (no .xlsx extension). A value of
None means "known, but deliberately excluded from the universe" - keep the
key so the reason is documented rather than the file silently vanishing.
"""

from __future__ import annotations

OVERRIDES: dict[str, str | None] = {
    # Zomato Ltd renamed to Eternal Ltd; the reference industry-exposure sheet
    # already lists it as "Eternal (Zomato)" under symbol ZOMATO, so fuzzy
    # matching likely resolves this correctly on its own, but we pin it
    # explicitly for determinism.
    "Eternal": "ZOMATO",
    # Ampersand/spacing variant of "Mahindra & Mahindra".
    "M & M": "M&M",
    # COMPANY NAME cell reads "LTM LTD", which doesn't match any company in
    # the reference industry-exposure sheet. The closest plausible identity
    # is LTIMindtree (symbol LTIM), based on the fuzzy filename similarity
    # and a plausible market-cap magnitude, but LTIM is not present in the
    # 198-ticker price/returns panel (data/nse200_nifty50_daily_prices) at
    # all, so this file cannot enter the model universe regardless of its
    # true identity. Excluded; flagged for equity_researcher to confirm.
    "LTM": None,
    # The following all scored below the 0.90 fuzzy-match auto-accept
    # threshold (0.65-0.89) purely due to verbose/abbreviated/reordered
    # naming (e.g. "I R F C" vs "Indian Railway Finance Corporation", or
    # "Adani Ports" being a strict prefix of "Adani Ports & Special Economic
    # Zone"), but are verified-correct by manual review against
    # data/nse_200_industry_exposure.xlsx - see equity_researcher's review
    # notes. Not a sign the fuzzy matcher is broken: every other sub-0.90
    # match in this batch was independently confirmed to have NO correct
    # candidate in the 200-company reference at all (name absent from the
    # reference, e.g. Coal India, Lodha, recent IPOs like Jio Financial/Tata
    # Capital) - a fixed score threshold can't distinguish "correct but
    # verbose" from "no correct answer exists" on its own, hence manual
    # confirmation here rather than lowering the threshold globally.
    "Indian Hotels Co": "INDHOTEL",
    "Lodha Developers": "LODHA",
    "Adani Ports": "ADANIPORTS",
    "Avenue Super": "DMART",
    "Cholaman.Inv.&Fn": "CHOLAFIN",
    "HDFC AMC": "HDFCAMC",
    "HDFC Life Insur": "HDFCLIFE",
    "I R F C": "IRFC",
    "Interglobe Aviat": "INDIGO",
    "Jindal Steel": "JINDALSTEEL",
    "O N G C": "ONGC",
    "SBI Life Insuran": "SBILIFE",
    "Tata Power Co": "TATAPOWER",
}
