"""Rolling window features for energy forecasting."""
from __future__ import annotations

import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

_WINDOW = 336  # 7 days of 30-min periods (7 * 24 * 2)
_MIN_PERIODS = 48  # 1 day of 30-min periods


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 7-day rolling mean for key columns.

    Target columns: wind_pct, solar_pct, carbon_intensity.
    Suffix for new columns: _7d_avg.
    Rolling window: 336 periods.
    Minimum periods: 48.

    Args:
        df: DataFrame containing the target columns.

    Returns:
        DataFrame with new rolling average columns added.
    """
    cols_to_roll = ["wind_pct", "solar_pct", "carbon_intensity"]
    result = df.copy()

    for col in cols_to_roll:
        if col not in df.columns:
            logger.warning("Column '%s' not found for rolling average calculation", col)
            continue

        new_col = f"{col}_7d_avg"
        result[new_col] = (
            df[col]
            .rolling(window=_WINDOW, min_periods=_MIN_PERIODS)
            .mean()
        )
        logger.debug("Computed 7-day rolling average for %s", col)

    return result
