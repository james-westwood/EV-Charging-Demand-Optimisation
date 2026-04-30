"""Feature store: write and read feature DataFrames from data/features/ or GCS."""
from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from google.cloud import storage

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


def load_features_from_gcs(bucket_name: str, prefix: str = "features/") -> pd.DataFrame:
    """Load the most recent feature DataFrame from GCS.

    Args:
        bucket_name: GCS bucket name.
        prefix: Prefix within the bucket where feature files are stored.

    Returns:
        DataFrame loaded from the most recent feature file in GCS.

    Raises:
        FileNotFoundError: If no feature files exist in the bucket/prefix.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # List blobs with the given prefix
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        raise FileNotFoundError(f"No feature files found in gs://{bucket_name}/{prefix}")
    
    # Filter to only parquet files and sort by name (lexicographic = chronological for YYYY-MM-DD)
    parquet_blobs = [b for b in blobs if b.name.endswith(".parquet")]
    if not parquet_blobs:
        raise FileNotFoundError(f"No feature files found in gs://{bucket_name}/{prefix}")
    
    # Sort by name to get the latest (YYYY-MM-DD format ensures lexicographic = chronological)
    latest_blob = sorted(parquet_blobs, key=lambda b: b.name)[-1]
    
    # Download to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
        try:
            latest_blob.download_to_filename(tmp_file.name)
            df = pd.read_parquet(tmp_file.name, engine="pyarrow")
            logger.info("Loaded %d rows from gs://%s/%s", len(df), bucket_name, latest_blob.name)
            return df
        finally:
            os.unlink(tmp_file.name)
