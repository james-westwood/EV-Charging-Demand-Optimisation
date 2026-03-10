"""Validator for generation mix DataFrames."""
import pandas as pd

from src.data.validators.exceptions import ValidationError
from src.logging_config import get_logger

logger = get_logger(__name__)

_FUEL_COLUMNS = [
    "gas",
    "coal",
    "nuclear",
    "wind",
    "hydro",
    "imports",
    "biomass",
    "other",
    "solar",
]
_EXPECTED_INTERVAL_MINUTES = 30


def validate_generation_mix(df: pd.DataFrame) -> None:
    """Validate a generation mix DataFrame.

    Checks (in order, raises on first failure):
    1. No null settlement_period timestamps.
    2. No duplicate settlement_period timestamps.
    3. Consecutive rows are exactly 30 minutes apart.
    4. All fuel columns are non-negative.
    5. Row sums are within 95-105% of 'total' column (if it exists)
       or within 5% of the row's maximum value (if no 'total' column).

    Args:
        df: DataFrame with settlement_period and fuel columns.

    Raises:
        ValidationError: On first validation failure.
    """
    if df.empty:
        logger.info("Empty DataFrame passed to validate_generation_mix — skipping")
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

    # 4. All fuel columns are non-negative
    for col in _FUEL_COLUMNS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        negative = series[series < 0]
        if not negative.empty:
            raise ValidationError(
                field=col,
                message=(
                    f"Fuel generation values must be >= 0; "
                    f"found {negative.tolist()} at indices {negative.index.tolist()}"
                ),
            )

    # 5. Row sum consistency
    present_fuels = [c for c in _FUEL_COLUMNS if c in df.columns]
    if present_fuels:
        row_sums = df[present_fuels].sum(axis=1)
        if "total" in df.columns:
            totals = df["total"]
            # Row sums within 95-105% of stated total
            lower_bound = totals * 0.95
            upper_bound = totals * 1.05
            out_of_sync = (row_sums < lower_bound) | (row_sums > upper_bound)
            if out_of_sync.any():
                idx = out_of_sync.idxmax() if out_of_sync.any() else None
                raise ValidationError(
                    field="total",
                    message=(
                        f"Row sum {row_sums[idx]:.2f} not within 95-105% of "
                        f"stated total {totals[idx]:.2f} at index {idx}"
                    ),
                )
        else:
            # Row sums within 5% of row max (as per instructions, though weird for MW)
            row_max = df[present_fuels].max(axis=1)
            # Avoid division by zero if row_max is 0
            # If row_max is 0, sum must be 0, which is within any % of 0.
            # But the instruction says "within 5% of row max".
            # If row_max > 0: |sum - max| / max <= 0.05
            # Which is equivalent to sum <= 1.05 * max (since sum >= max for non-negative values)
            
            # Use sum <= 1.05 * row_max
            bad_sum = row_sums > (1.05 * row_max)
            if bad_sum.any():
                idx = bad_sum.idxmax()
                raise ValidationError(
                    field="multiple",
                    message=(
                        f"Row sum {row_sums[idx]:.2f} exceeds row max {row_max[idx]:.2f} "
                        f"by more than 5% at index {idx}"
                    ),
                )

    logger.info("Generation mix validation passed (%d rows)", len(df))
