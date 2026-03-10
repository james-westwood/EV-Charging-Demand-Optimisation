"""Validator for carbon intensity DataFrames."""
import pandas as pd

from src.data.validators.exceptions import ValidationError
from src.logging_config import get_logger

logger = get_logger(__name__)

_INTENSITY_MIN = 0
_INTENSITY_MAX = 800
_EXPECTED_INTERVAL_MINUTES = 30


def validate_carbon_intensity(df: pd.DataFrame) -> None:
    """Validate a carbon intensity DataFrame.

    Checks (in order, raises on first failure):
    1. No null values in ``settlement_period``.
    2. No duplicate ``settlement_period`` timestamps.
    3. Consecutive rows are exactly 30 minutes apart.
    4. ``intensity_actual`` values are in the range [0, 800].

    Args:
        df: DataFrame with columns [settlement_period, intensity_actual, intensity_forecast].

    Raises:
        ValidationError: On first validation failure, with ``field`` set to the
            offending column name.
    """
    if df.empty:
        logger.info("Empty DataFrame passed to validate_carbon_intensity — skipping")
        return

    # 1. No null settlement_periods
    if df["settlement_period"].isna().any():
        raise ValidationError(
            field="settlement_period",
            message="DataFrame contains null settlement_period values",
        )

    # 2. No duplicate timestamps
    if df["settlement_period"].duplicated().any():
        raise ValidationError(
            field="settlement_period",
            message="DataFrame contains duplicate settlement_period timestamps",
        )

    # 3. 30-min intervals between consecutive rows
    if len(df) > 1:
        sorted_periods = df["settlement_period"].sort_values()
        diffs = sorted_periods.diff().dropna()
        expected = pd.Timedelta(minutes=_EXPECTED_INTERVAL_MINUTES)
        bad = diffs[diffs != expected]
        if not bad.empty:
            raise ValidationError(
                field="settlement_period",
                message=(
                    f"Expected {_EXPECTED_INTERVAL_MINUTES}-minute intervals between rows; "
                    f"found irregular gaps at indices {bad.index.tolist()}"
                ),
            )

    # 4. Intensity values in range [0, 800]
    for col in ("intensity_actual", "intensity_forecast"):
        if col not in df.columns:
            continue
        series = df[col].dropna()
        out_of_range = series[(series < _INTENSITY_MIN) | (series > _INTENSITY_MAX)]
        if not out_of_range.empty:
            raise ValidationError(
                field=col,
                message=(
                    f"Values must be in [{_INTENSITY_MIN}, {_INTENSITY_MAX}]; "
                    f"found {out_of_range.tolist()} at indices {out_of_range.index.tolist()}"
                ),
            )

    logger.info("Carbon intensity validation passed (%d rows)", len(df))
