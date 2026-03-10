"""Validator for weather DataFrames."""
import pandas as pd

from src.data.validators.exceptions import ValidationError
from src.logging_config import get_logger

logger = get_logger(__name__)

_TEMPERATURE_MIN = -30.0
_TEMPERATURE_MAX = 50.0
_WIND_SPEED_MIN = 0.0
_WIND_SPEED_MAX = 150.0
_RADIATION_MIN = 0.0
_EXPECTED_CITIES = {"London", "Manchester", "Edinburgh"}


def validate_weather(df: pd.DataFrame) -> None:
    """Validate a weather DataFrame.

    Checks (in order, raises on first failure):
    1. ``temperature`` values are in the range [−30, 50] °C.
    2. ``wind_speed`` values are in the range [0, 150] km/h.
    3. ``radiation`` values are ≥ 0.
    4. All three cities (London, Manchester, Edinburgh) are present for
       every timestamp in the ``city`` column.

    Args:
        df: DataFrame with columns [timestamp, city, temperature, wind_speed, radiation].

    Raises:
        ValidationError: On first validation failure, with ``field`` set to the
            offending column name.
    """
    if df.empty:
        logger.info("Empty DataFrame passed to validate_weather — skipping")
        return

    # 1. Temperature in range [−30, 50]
    if "temperature" in df.columns:
        series = df["temperature"].dropna()
        out_of_range = series[(series < _TEMPERATURE_MIN) | (series > _TEMPERATURE_MAX)]
        if not out_of_range.empty:
            raise ValidationError(
                field="temperature",
                message=(
                    f"Values must be in [{_TEMPERATURE_MIN}, {_TEMPERATURE_MAX}] °C; "
                    f"found {out_of_range.tolist()} at indices {out_of_range.index.tolist()}"
                ),
            )

    # 2. Wind speed in range [0, 150]
    if "wind_speed" in df.columns:
        series = df["wind_speed"].dropna()
        out_of_range = series[(series < _WIND_SPEED_MIN) | (series > _WIND_SPEED_MAX)]
        if not out_of_range.empty:
            raise ValidationError(
                field="wind_speed",
                message=(
                    f"Values must be in [{_WIND_SPEED_MIN}, {_WIND_SPEED_MAX}] km/h; "
                    f"found {out_of_range.tolist()} at indices {out_of_range.index.tolist()}"
                ),
            )

    # 3. Radiation ≥ 0
    if "radiation" in df.columns:
        series = df["radiation"].dropna()
        negative = series[series < _RADIATION_MIN]
        if not negative.empty:
            raise ValidationError(
                field="radiation",
                message=(
                    f"Values must be >= {_RADIATION_MIN}; "
                    f"found {negative.tolist()} at indices {negative.index.tolist()}"
                ),
            )

    # 4. All three cities present per timestamp
    if "city" in df.columns and "timestamp" in df.columns:
        for ts, group in df.groupby("timestamp"):
            present = set(group["city"].unique())
            missing = _EXPECTED_CITIES - present
            if missing:
                raise ValidationError(
                    field="city",
                    message=(
                        f"Missing cities {sorted(missing)} for timestamp {ts}; "
                        f"expected {sorted(_EXPECTED_CITIES)}"
                    ),
                )

    logger.info("Weather validation passed (%d rows)", len(df))
