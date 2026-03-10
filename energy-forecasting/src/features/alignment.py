"""Align carbon intensity and generation mix DataFrames to settlement periods."""
from __future__ import annotations

import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

_MAX_FILL_GAP = 3  # consecutive missing periods that may be forward-filled


def align_to_settlement_periods(
    carbon_df: pd.DataFrame,
    generation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join carbon intensity and generation mix DataFrames on settlement_period.

    Steps:
    1. Build a complete 30-min settlement period index spanning both inputs.
    2. Reindex both DataFrames to the full grid (missing rows become NaN).
    3. Forward-fill runs of NaN that are at most 3 periods long.
    4. Drop any remaining NaN rows (gap was > 3 periods).

    Args:
        carbon_df: DataFrame with a ``settlement_period`` column (datetime, UTC).
        generation_df: DataFrame with a ``settlement_period`` column (datetime, UTC).

    Returns:
        Merged DataFrame with ``settlement_period`` as a column plus all columns
        from both inputs.  Rows with unfillable gaps are excluded.
    """
    if "settlement_period" not in carbon_df.columns:
        raise ValueError("carbon_df must contain a 'settlement_period' column")
    if "settlement_period" not in generation_df.columns:
        raise ValueError("generation_df must contain a 'settlement_period' column")

    carbon = carbon_df.set_index("settlement_period")
    generation = generation_df.set_index("settlement_period")

    # Build a contiguous 30-min index spanning both datasets.
    all_periods = carbon.index.union(generation.index)
    full_index = pd.date_range(
        start=all_periods.min(),
        end=all_periods.max(),
        freq="30min",
        name="settlement_period",
    )

    merged = pd.concat(
        [carbon.reindex(full_index), generation.reindex(full_index)],
        axis=1,
    )

    merged = _forward_fill_short_gaps(merged, max_gap=_MAX_FILL_GAP)

    rows_before = len(merged)
    merged = merged.dropna()
    rows_dropped = rows_before - len(merged)
    if rows_dropped:
        logger.info(
            "Dropped %d row(s) with gaps longer than %d period(s)",
            rows_dropped,
            _MAX_FILL_GAP,
        )

    merged = merged.reset_index()

    logger.info(
        "Aligned %d settlement period(s) from %s to %s",
        len(merged),
        merged["settlement_period"].iloc[0] if len(merged) else "N/A",
        merged["settlement_period"].iloc[-1] if len(merged) else "N/A",
    )

    return merged


def _forward_fill_short_gaps(df: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """Forward-fill NaN runs of length <= max_gap; leave longer runs as NaN.

    Args:
        df: DataFrame that may contain NaN values.
        max_gap: Maximum consecutive NaN count to forward-fill.

    Returns:
        New DataFrame with short gaps filled.
    """
    result = df.copy()
    for col in df.columns:
        series = df[col]
        is_nan = series.isna()
        if not is_nan.any():
            continue

        # Assign a group ID that increments at each non-NaN value.
        # All NaNs following a non-NaN value share that value's group.
        run_id = (~is_nan).cumsum()

        # Count how many NaNs belong to each group.
        nan_run_length = is_nan.groupby(run_id).transform("sum")

        # Forward-fill everything then restore cells that belong to long gaps.
        filled = series.ffill()
        long_gap_mask = is_nan & (nan_run_length > max_gap)
        filled[long_gap_mask] = float("nan")

        result[col] = filled

    return result
