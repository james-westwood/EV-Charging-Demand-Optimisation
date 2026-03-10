"""Feature engineering pipeline for EV charging demand optimisation."""
from __future__ import annotations

import pandas as pd

from src.features.alignment import align_to_settlement_periods
from src.features.calendar import add_calendar_features
from src.features.lags import add_lag_features
from src.features.penetration import add_penetration_features
from src.features.rolling import add_rolling_features
from src.features.weather_join import join_weather_to_grid
from src.logging_config import get_logger

logger = get_logger(__name__)


def feature_pipeline(
    carbon_df: pd.DataFrame,
    generation_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compose all feature engineering steps into a single pipeline.

    Steps:
    1. Align carbon and generation data to 30-min settlement periods.
    2. Join weather data (interpolated to 30-min grid).
    3. Add generation penetration features (wind_pct, etc.).
    4. Add rolling average features (7-day averages).
    5. Add lag features (t-1, t-2, t-48, t-336).
    6. Add calendar features (hour, day, holiday, etc.).
    7. Clean up: rename columns and drop rows with NaNs in non-lag columns.

    Args:
        carbon_df: Raw carbon intensity data.
        generation_df: Raw generation mix data.
        weather_df: Raw long-format weather data.

    Returns:
        DataFrame with all engineered features.
    """
    logger.info("Starting feature engineering pipeline")

    # 1. Align grid data
    # alignment.py handles the merging and short-gap filling.
    # It returns a DataFrame with 'settlement_period' as a column.
    df = align_to_settlement_periods(carbon_df, generation_df)
    if df.empty:
        logger.warning("Pipeline produced empty DataFrame after alignment")
        return df

    # Standardise column names
    # rolling.py expects 'carbon_intensity' but collector provides 'intensity_actual'
    if "intensity_actual" in df.columns:
        df = df.rename(columns={"intensity_actual": "carbon_intensity"})
        logger.debug("Renamed intensity_actual to carbon_intensity")

    # 2. Join weather data
    # weather_join expects the grid as a DatetimeIndex
    grid_index = pd.DatetimeIndex(df["settlement_period"])
    weather_wide = join_weather_to_grid(weather_df, grid_index)
    
    # Merge weather features back to main DataFrame
    # Both have settlement_period (as column in df, as index in weather_wide)
    df = df.set_index("settlement_period")
    df = pd.concat([df, weather_wide], axis=1)
    df = df.reset_index()

    # 3. Generation penetration
    df = add_penetration_features(df)

    # 4. Rolling averages
    # Targets: wind_pct, solar_pct, carbon_intensity
    df = add_rolling_features(df)

    # 5. Lag features
    # Target columns for lags as per PRD/EPIC 4: carbon_intensity and wind generation (or wind_pct)
    # We'll lag the main targets.
    lag_targets = ["carbon_intensity", "wind_pct"]
    df = add_lag_features(df, lag_targets)

    # 6. Calendar features
    df = add_calendar_features(df)

    # 7. Final clean up
    # Acceptance criteria: "zero NaN in non-lag columns"
    # Lag columns have '_lag_' in their name.
    lag_cols = [c for c in df.columns if "_lag_" in c]
    non_lag_cols = [c for c in df.columns if c not in lag_cols]
    
    rows_before = len(df)
    df = df.dropna(subset=non_lag_cols)
    rows_after = len(df)
    
    if rows_before != rows_after:
        logger.info("Dropped %d rows with NaNs in non-lag columns", rows_before - rows_after)

    logger.info("Feature engineering pipeline complete. Final shape: %s", df.shape)
    return df
