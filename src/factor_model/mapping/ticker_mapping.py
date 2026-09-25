"""Fuzzy-match fundamentals/*.xlsx filenames to NSE tickers and lock the model universe.

The model universe is the intersection of (a) the 198-ticker price/returns
panel in data/ and (b) companies with a fundamentals/*.xlsx file, per the
project's universe decision (see CLAUDE.md). This module builds that mapping
and writes model_data/universe_99.csv, which is the single source of truth
downstream modules subset to (.claude/rules/data-boundaries.md).
"""

from __future__ import annotations

import difflib
import logging
import re
from pathlib import Path

import pandas as pd

from factor_model import config
from factor_model.io.fundamentals import parse_fundamentals_file
from factor_model.mapping.overrides import OVERRIDES

logger = logging.getLogger(__name__)

FUZZY_MATCH_THRESHOLD = 0.90

_SUFFIX_RE = re.compile(r"\s*\(\d+\)\s*$")
_LEGAL_SUFFIX_RE = re.compile(r"\b(LTD|LIMITED|LIMTED)\b", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^A-Z0-9& ]")


def strip_duplicate_suffix(filename_stem: str) -> str:
    """Strip a trailing " (1)"-style duplicate-download suffix from a filename stem."""
    return _SUFFIX_RE.sub("", filename_stem).strip()


def normalize_company_name(name: str) -> str:
    name = name.upper()
    name = _LEGAL_SUFFIX_RE.sub("", name)
    name = _PUNCT_RE.sub(" ", name)
    return re.sub(r"\s+", " ", name).strip()


def load_reference_companies() -> pd.DataFrame:
    """Symbol / Company Name reference list (200 rows) from the industry-exposure workbook."""
    df = pd.read_excel(config.INDUSTRY_EXPOSURE_XLSX, sheet_name="Industry Exposure Matrix")
    return df[["Symbol", "Company Name"]].drop_duplicates()


def load_price_universe_symbols() -> set[str]:
    """The set of tickers actually present in the price/returns panel."""
    avail = pd.read_csv(config.DATA_DIR / "nse_200_timeseries_availability.csv")
    return set(avail.loc[avail["Status"] == "OK", "Symbol"])


def _best_fuzzy_match(name: str, reference: pd.DataFrame) -> tuple[str | None, float]:
    norm_name = normalize_company_name(name)
    best_symbol, best_score = None, 0.0
    for symbol, ref_name in zip(reference["Symbol"], reference["Company Name"]):
        norm_ref = normalize_company_name(str(ref_name))
        score = difflib.SequenceMatcher(None, norm_name, norm_ref).ratio()
        if score > best_score:
            best_symbol, best_score = symbol, score
    return best_symbol, best_score


def build_universe_mapping(fundamentals_dir: Path | None = None) -> pd.DataFrame:
    """Build the fundamentals-filename -> symbol mapping table (one row per xlsx file)."""
    fundamentals_dir = fundamentals_dir or config.FUNDAMENTALS_DIR
    reference = load_reference_companies()
    price_universe = load_price_universe_symbols()

    rows: list[dict] = []
    for path in sorted(fundamentals_dir.glob("*.xlsx")):
        stem = path.stem
        dedup_key = strip_duplicate_suffix(stem)

        try:
            parsed = parse_fundamentals_file(path)
            company_name = parsed.company_name or dedup_key
        except Exception:
            logger.exception("Could not parse %s for mapping; falling back to filename", path.name)
            company_name = dedup_key

        if dedup_key in OVERRIDES:
            symbol = OVERRIDES[dedup_key]
            method = "override" if symbol is not None else "manual_exclude"
            score = 1.0
        else:
            symbol, score = _best_fuzzy_match(str(company_name), reference)
            method = "fuzzy"

        needs_review = method == "fuzzy" and score < FUZZY_MATCH_THRESHOLD
        in_price_universe = symbol in price_universe if symbol else False

        rows.append(
            {
                "fundamentals_filename": path.name,
                "mtime": path.stat().st_mtime,
                "dedup_key": dedup_key,
                "company_name": company_name,
                "matched_symbol": symbol,
                "match_score": round(score, 4),
                "match_method": method,
                "in_price_universe": in_price_universe,
                "needs_review": needs_review,
            }
        )

    mapping = pd.DataFrame(rows)

    # Resolve duplicate downloads mapping to the same symbol: keep the newest file.
    mapping["included"] = mapping["matched_symbol"].notna() & mapping["in_price_universe"] & ~mapping["needs_review"]
    mapping["exclusion_reason"] = ""
    mapping.loc[mapping["matched_symbol"].isna(), "exclusion_reason"] = "no symbol match"
    mapping.loc[
        mapping["matched_symbol"].notna() & ~mapping["in_price_universe"], "exclusion_reason"
    ] = "symbol not in price/returns panel"
    mapping.loc[mapping["needs_review"], "exclusion_reason"] = "low-confidence fuzzy match, needs manual review"

    dupe_mask = mapping["included"] & mapping.duplicated(subset=["matched_symbol"], keep=False)
    for symbol, group in mapping[dupe_mask].groupby("matched_symbol"):
        keep_idx = group["mtime"].idxmax()
        drop_idx = [i for i in group.index if i != keep_idx]
        mapping.loc[drop_idx, "included"] = False
        mapping.loc[drop_idx, "exclusion_reason"] = f"duplicate download for {symbol}, kept newer file"

    mapping = mapping.drop(columns=["mtime"]).sort_values("fundamentals_filename").reset_index(drop=True)
    return mapping


def get_locked_universe(mapping: pd.DataFrame) -> list[str]:
    return sorted(mapping.loc[mapping["included"], "matched_symbol"].unique().tolist())


def save_universe_mapping(mapping: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or config.UNIVERSE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(path, index=False)
    n_included = int(mapping["included"].sum())
    logger.info("Wrote universe mapping to %s (%d included, %d excluded)", path, n_included, len(mapping) - n_included)
    return path
