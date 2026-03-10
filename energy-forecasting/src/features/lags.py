"""Lag features for energy forecasting."""
from __future__ import annotations

import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

_LAGS = [1, 2, 48, 336]


def add_lag_features(df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    """Create lag columns for target series at specified intervals.

    Lags: t-1, t-2, t-48 (1 day), t-336 (1 week).
    Column names: {target}_lag_{n}.

    Args:
        df: DataFrame containing the target columns.
        target_cols: List of column names to create lags for.

    Returns:
        DataFrame with new lag columns added.
    """
    result = df.copy()

    for col in target_cols:
        if col not in df.columns:
            logger.warning("Column '%s' not found for lag feature calculation", col)
            continue

        for lag in _LAGS:
            new_col = f"{col}_lag_{lag}"
            result[new_col] = df[col].shift(lag)
            logger.debug("Computed lag %d for %s", lag, col)

    return result
