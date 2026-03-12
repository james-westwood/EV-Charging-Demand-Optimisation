"""Calendar features for energy forecasting."""
from __future__ import annotations

import pandas as pd
from workalendar.europe import UnitedKingdom
import numpy as np

from src.logging_config import get_logger

logger = get_logger(__name__)

_uk_cal = UnitedKingdom()


def _get_dt_series(df: pd.DataFrame) -> pd.Series:
    """Extract a datetime Series from a DatetimeIndex or settlement_period column."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index.to_series().reset_index(drop=True)
    if "settlement_period" in df.columns:
        return pd.to_datetime(df["settlement_period"]).reset_index(drop=True)
    raise ValueError(
        "DataFrame must have a DatetimeIndex or a 'settlement_period' column."
    )


def _build_holiday_set(years: set[int]) -> set[object]:
    holidays: set[object] = set()
    for year in years:
        for date, _ in _uk_cal.holidays(year):
            holidays.add(date)
    return holidays



def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based feature columns to df.

    Adds: hour_of_day, day_of_week, month, is_weekend, is_bank_holiday_uk.

    Args:
        df: DataFrame with a DatetimeIndex or a 'settlement_period' column.

    Returns:
        DataFrame with new calendar columns appended.
    """
    result = df.copy()
    dt = _get_dt_series(df)

    result["hour_of_day"] = dt.dt.hour.values
    result["day_of_week"] = dt.dt.dayofweek.values  # 0=Mon
    result["day_of_year"] = dt.dt.dayofyear.values
    result["month"] = dt.dt.month.values
    result["is_weekend"] = (dt.dt.dayofweek >= 5).values
    result["season_sin"] = np.sin(2 * np.pi * result['day_of_year'] / 365)
    result["season_cos"] = np.cos(2 * np.pi * result['day_of_year'] / 365)

    years = set(dt.dt.year.unique())
    holidays = _build_holiday_set(years)
    result["is_bank_holiday_uk"] = dt.dt.date.isin(holidays).values

    logger.debug("Added calendar features for %d rows", len(result))
    return result
