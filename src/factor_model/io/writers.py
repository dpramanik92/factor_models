"""Save helpers that write generated artifacts to output/ and mirror them into data/.

The user wants generated factor time-series and other data outputs available
in data/ alongside output/ (both folders are git-ignored, so this has no
effect on what gets committed - see .claude/rules/data-boundaries.md). Reports
that aren't time-series/tabular data (validation_report.md, plots) are not
mirrored - only CSV/parquet data artifacts are.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from factor_model import config

logger = logging.getLogger(__name__)


def _mirror_path(path: Path) -> Path | None:
    try:
        rel = path.relative_to(config.OUTPUT_DIR)
    except ValueError:
        return None
    return config.DATA_DIR / rel


def save_csv(df: pd.DataFrame, path: Path, index: bool = True, mirror_to_data: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    if mirror_to_data:
        mirror = _mirror_path(path)
        if mirror is not None:
            mirror.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(mirror, index=index)
            logger.info("Wrote %s (mirrored to %s)", path, mirror)
        else:
            logger.info("Wrote %s", path)


def save_parquet(df: pd.DataFrame, path: Path, mirror_to_data: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    if mirror_to_data:
        mirror = _mirror_path(path)
        if mirror is not None:
            mirror.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(mirror)
            logger.info("Wrote %s (mirrored to %s)", path, mirror)
        else:
            logger.info("Wrote %s", path)
