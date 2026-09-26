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
    # the reference industry-exposure sheet by name. Confirmed by inspecting
    # the actual filing data: sales jump from Rs.12,370cr (FY21) to
    # Rs.26,109cr (FY22) to Rs.33,183cr (FY23) - exactly the L&T Infotech +
    # Mindtree merger (effective Nov 2022) that created LTIMindtree. This is
    # LTIMindtree's (symbol LTIM) fundamentals. LTIM previously had NO_DATA
    # in the price panel (a prior Yahoo Finance fetch attempt failed, not a
    # genuine listing gap - LTIM is a real, currently-trading large-cap) and
    # was one-off backfilled alongside ICICIAMC - see CLAUDE.md.
    "LTM": "LTIM",
    # Pre-merger standalone Mindtree Ltd - only covers FY2013-FY2019, well
    # before the Nov 2022 L&T Infotech merger that created LTIMindtree. "LTM"
    # above already supplies LTIMindtree's current (post-merger) fundamentals
    # under symbol LTIM; this stale, superseded file is excluded rather than
    # also mapped to LTIM, which would conflate two non-overlapping filings
    # for what is now one entity.
    "Mindtree": None,
    # "ICICI PRUDENTIAL ASSET MANAGEMENT CO LTD" fuzzy-matched to HDFCAMC at
    # low confidence (0.63) - a different company entirely. ICICI Prudential
    # AMC only IPO'd and listed on NSE/BSE on 2025-12-19 (symbol ICICIAMC),
    # so it wasn't in the reference industry-exposure sheet or price panel
    # when those were first built. One-off backfilled - see CLAUDE.md.
    "ICICI AMC": "ICICIAMC",
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
    # Second batch of overrides, added when fundamentals coverage was expanded
    # from 99 to 212 files - all verified correct-but-verbose/abbreviated
    # matches scoring 0.60-0.89 (below the 0.90 auto-accept threshold), same
    # rationale as the batch above.
    "C.E. Info System": "MAPMYINDIA",
    "Dixon Technolog": "DIXON",
    "FSN E-Commerce": "NYKAA",
    "ICICI Lombard": "ICICIGI",
    "K P R Mill Ltd": "KPRMILL",
    "Multi Comm. Exc": "MCX",
    "One 97": "PAYTM",
    "PB Fintech": "POLICYBZR",
}
