"""Inference-time feature generation.

The feature pipeline normally runs on full DuckDB history. At inference time we only have
the last N periods cached, so we use those to compute features for the forecast horizon.
"""
from datetime import datetime, timezone, timedelta

import pandas as pd

from src.features.calendar_features import add_calendar_features
from src.features.lags import add_lag_features
from src.features.penetration import add_penetration_features
from src.features.rolling import add_rolling_features
from src.features.store import load_features
from src.logging_config import get_logger

logger = get_logger(__name__)

_SETTLEMENT_PERIOD = timedelta(minutes=30)
_LOOKBACK_PERIODS = 336  # 7 days of 30-min periods


def generate_timestamps(
    start: datetime,
    horizon: int,
) -> list[datetime]:
    """Generate settlement period timestamps for N periods ahead.

    Args:
        start: Start datetime (UTC)
        horizon: Number of 30-min periods to forecast

    Returns:
        List of timestamps
    """
    return [start + i * _SETTLEMENT_PERIOD for i in range(horizon)]


def create_future_grid(timestamps: list[datetime]) -> pd.DataFrame:
    """Create empty DataFrame with future settlement_periods.

    Args:
        timestamps: List of future timestamps

    Returns:
        DataFrame with settlement_period column
    """
    return pd.DataFrame({"settlement_period": timestamps})


def generate_inference_features(
    horizon: int,
    start: datetime | None = None,
) -> pd.DataFrame:
    """Generate features for inference.

    Loads cached features (last ~7 days) to compute rolling/lag features
    for the forecast horizon. Calendar features computed for future timestamps.

    Args:
        horizon: Number of 30-min periods to forecast (e.g., 48 = 24 hours)
        start: Start datetime. Defaults to now.

    Returns:
        DataFrame with feature columns matching training schema
    """
    if start is None:
        start = datetime.now(timezone.utc)

    timestamps = generate_timestamps(start, horizon)
    future_df = create_future_grid(timestamps)

    cached_df = load_features()

    n_lookback = min(_LOOKBACK_PERIODS, len(cached_df))
    history = cached_df.tail(n_lookback).copy()

    combined = pd.concat([history, future_df], ignore_index=True)

    combined = add_penetration_features(combined)
    combined = add_rolling_features(combined)
    combined = add_lag_features(combined, ["carbon_intensity", "wind_pct"])
    combined = add_calendar_features(combined)

    lag_cols = [c for c in combined.columns if "_lag_" in c]
    non_lag_cols = [c for c in combined.columns if c not in lag_cols]
    combined = combined.dropna(subset=non_lag_cols)

    inference_df = combined.tail(horizon).reset_index(drop=True)

    feature_cols = [c for c in inference_df.columns if c not in ["settlement_period", "carbon_intensity"]]
    return inference_df[feature_cols]