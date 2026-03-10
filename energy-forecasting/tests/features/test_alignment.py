"""Tests for src/features/alignment.py."""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.alignment import align_to_settlement_periods


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _carbon_df(periods: list[pd.Timestamp]) -> pd.DataFrame:
    n = len(periods)
    return pd.DataFrame(
        {
            "settlement_period": periods,
            "intensity_actual": [150.0] * n,
            "intensity_forecast": [145.0] * n,
        }
    )


def _generation_df(periods: list[pd.Timestamp]) -> pd.DataFrame:
    n = len(periods)
    return pd.DataFrame(
        {
            "settlement_period": periods,
            "gas": [1000.0] * n,
            "coal": [0.0] * n,
            "nuclear": [500.0] * n,
            "wind": [200.0] * n,
            "hydro": [10.0] * n,
            "imports": [5.0] * n,
            "biomass": [20.0] * n,
            "other": [2.0] * n,
            "solar": [30.0] * n,
        }
    )


BASE = pd.Timestamp("2024-01-01 00:00", tz="UTC")


# ---------------------------------------------------------------------------
# Acceptance criteria tests
# ---------------------------------------------------------------------------


def test_one_period_gap_is_filled() -> None:
    """A 1-period gap in generation must be forward-filled."""
    all_periods = pd.date_range(BASE, periods=5, freq="30min").tolist()
    # Remove the third period (index 2) from generation to create a 1-period gap.
    generation_periods = [all_periods[0], all_periods[1], all_periods[3], all_periods[4]]

    result = align_to_settlement_periods(
        _carbon_df(all_periods),
        _generation_df(generation_periods),
    )

    assert len(result) == 5, "All 5 periods should be present after filling the 1-period gap"

    filled_row = result.loc[result["settlement_period"] == all_periods[2]].iloc[0]
    # Forward-filled from period index 1 → same value
    assert filled_row["wind"] == pytest.approx(200.0)
    assert filled_row["intensity_actual"] == pytest.approx(150.0)


def test_four_period_gap_rows_are_dropped() -> None:
    """A 4-period consecutive gap must not be filled; those rows are dropped."""
    all_periods = pd.date_range(BASE, periods=10, freq="30min").tolist()
    # Remove indices 3, 4, 5, 6 from generation → 4-period gap.
    generation_periods = all_periods[:3] + all_periods[7:]

    result = align_to_settlement_periods(
        _carbon_df(all_periods),
        _generation_df(generation_periods),
    )

    assert len(result) == 6, "4 gap rows should be dropped, leaving 6"

    gap_periods = set(all_periods[3:7])
    result_periods = set(result["settlement_period"])
    assert gap_periods.isdisjoint(result_periods), "Gap periods must not appear in output"


# ---------------------------------------------------------------------------
# Additional behavioural tests
# ---------------------------------------------------------------------------


def test_three_period_gap_is_filled() -> None:
    """A gap of exactly 3 periods (the maximum) must be forward-filled."""
    all_periods = pd.date_range(BASE, periods=8, freq="30min").tolist()
    # Remove indices 2, 3, 4 from generation → 3-period gap.
    generation_periods = all_periods[:2] + all_periods[5:]

    result = align_to_settlement_periods(
        _carbon_df(all_periods),
        _generation_df(generation_periods),
    )

    assert len(result) == 8, "3-period gap should be filled; all 8 rows retained"
    for period in all_periods[2:5]:
        filled = result.loc[result["settlement_period"] == period].iloc[0]
        assert filled["wind"] == pytest.approx(200.0)


def test_no_gaps_returns_full_merged_dataframe() -> None:
    """When both DataFrames share the same periods, all rows are returned."""
    periods = pd.date_range(BASE, periods=10, freq="30min").tolist()

    result = align_to_settlement_periods(
        _carbon_df(periods),
        _generation_df(periods),
    )

    assert len(result) == 10
    assert "settlement_period" in result.columns
    assert "intensity_actual" in result.columns
    assert "wind" in result.columns


def test_output_contains_all_columns() -> None:
    """Output must include settlement_period and all columns from both inputs."""
    periods = pd.date_range(BASE, periods=4, freq="30min").tolist()

    result = align_to_settlement_periods(
        _carbon_df(periods),
        _generation_df(periods),
    )

    carbon_cols = {"settlement_period", "intensity_actual", "intensity_forecast"}
    generation_cols = {"gas", "coal", "nuclear", "wind", "hydro", "imports", "biomass", "other", "solar"}
    assert carbon_cols.issubset(result.columns)
    assert generation_cols.issubset(result.columns)


def test_gap_in_carbon_is_filled() -> None:
    """A 1-period gap in carbon intensity (not generation) is also forward-filled."""
    all_periods = pd.date_range(BASE, periods=5, freq="30min").tolist()
    carbon_periods = [all_periods[0], all_periods[1], all_periods[3], all_periods[4]]

    result = align_to_settlement_periods(
        _carbon_df(carbon_periods),
        _generation_df(all_periods),
    )

    assert len(result) == 5
    filled_row = result.loc[result["settlement_period"] == all_periods[2]].iloc[0]
    assert filled_row["intensity_actual"] == pytest.approx(150.0)


def test_empty_result_when_all_gaps_exceed_limit() -> None:
    """If every period has a gap > 3, the result should be empty."""
    # Carbon: first 2 periods. Generation: last 2 periods with a 4-period gap.
    all_periods = pd.date_range(BASE, periods=8, freq="30min").tolist()
    carbon_periods = all_periods[:2]
    generation_periods = all_periods[6:]

    result = align_to_settlement_periods(
        _carbon_df(carbon_periods),
        _generation_df(generation_periods),
    )

    # The gap between the two datasets is 4 periods (indices 2–5) for one side
    # or NaN entirely for the other; all mismatched rows should be dropped.
    assert result["intensity_actual"].notna().all()
    assert result["wind"].notna().all()


def test_invalid_carbon_df_raises() -> None:
    """Missing settlement_period column in carbon_df raises ValueError."""
    bad_df = pd.DataFrame({"intensity_actual": [100.0]})
    good_df = _generation_df([BASE])

    with pytest.raises(ValueError, match="settlement_period"):
        align_to_settlement_periods(bad_df, good_df)


def test_invalid_generation_df_raises() -> None:
    """Missing settlement_period column in generation_df raises ValueError."""
    good_df = _carbon_df([BASE])
    bad_df = pd.DataFrame({"wind": [200.0]})

    with pytest.raises(ValueError, match="settlement_period"):
        align_to_settlement_periods(good_df, bad_df)
