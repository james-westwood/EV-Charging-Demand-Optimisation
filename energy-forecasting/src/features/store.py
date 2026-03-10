"""Feature store: write and read feature DataFrames from data/features/."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

_FEATURES_DIR = Path("data/features")


def write_features(
    df: pd.DataFrame,
    run_date: date | None = None,
    features_dir: Path | None = None,
) -> Path:
    """Save a feature DataFrame to data/features/features_{YYYY-MM-DD}.parquet.

    Args:
        df: Feature DataFrame to persist.
        run_date: Date to use in the filename. Defaults to today (UTC).
        features_dir: Override the default features directory (useful in tests).

    Returns:
        Path to the written file.
    """
    if run_date is None:
        run_date = date.today()

    directory = features_dir if features_dir is not None else _FEATURES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"features_{run_date.isoformat()}.parquet"
    df.to_parquet(path, engine="pyarrow", index=False)
    logger.info("Wrote %d rows to %s", len(df), path)
    return path


def load_features(features_dir: Path | None = None) -> pd.DataFrame:
    """Read the most recent features_{YYYY-MM-DD}.parquet from data/features/.

    Args:
        features_dir: Override the default features directory (useful in tests).

    Returns:
        DataFrame loaded from the most recent features file.

    Raises:
        FileNotFoundError: If no feature files exist in the directory.
    """
    directory = features_dir if features_dir is not None else _FEATURES_DIR
    files = sorted(directory.glob("features_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No feature files found in {directory}")

    path = files[-1]
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info("Loaded %d rows from %s", len(df), path)
    return df
