"""Tests for src/features/run_feature_pipeline.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.run_feature_pipeline import feature_pipeline


def test_feature_pipeline_expected_columns(carbon_intensity_df, generation_mix_df, weather_df) -> None:
    """Test that the pipeline produces all expected columns.

    Acceptance criteria: output has all expected columns (lags, calendar, rolling, penetration, weather).
    """
    # Since the fixtures are only 10 rows, rolling and lag features will all be NaN.
    # The pipeline drops NaNs from non-lag columns (including rolling).
    # To test that the columns are created, we can temporarily disable dropna or just check that they exist
    # before the final dropna if we were to modify the pipeline.
    # Alternatively, we can use a larger synthetic dataset for this test.

    result = feature_pipeline(carbon_intensity_df, generation_mix_df, weather_df)

    # Note: result might be empty if we drop all rows with NaNs in non-lag columns.
    # Let's verify that the columns exist even if the DataFrame is empty.

    expected_col_prefixes = [
        "carbon_intensity",
        "wind_pct",
        "solar_pct",
        "low_carbon_pct",
        "london_temperature",
        "manchester_wind_speed",
        "edinburgh_radiation",
        "hour_of_day",
        "day_of_week",
        "is_bank_holiday_uk",
        "_7d_avg",
        "_lag_1",
        "_lag_336",
    ]

    for prefix in expected_col_prefixes:
        assert any(c.startswith(prefix) or prefix in c for c in result.columns), f"Missing column for {prefix}"


def test_feature_pipeline_no_nans_in_non_lags(carbon_intensity_df, generation_mix_df, weather_df) -> None:
    """Test that non-lag columns have zero NaNs.

    Acceptance criteria: zero NaN in non-lag columns.
    """
    result = feature_pipeline(carbon_intensity_df, generation_mix_df, weather_df)

    lag_cols = [c for c in result.columns if "_lag_" in c]
    non_lag_cols = [c for c in result.columns if c not in lag_cols]

    for col in non_lag_cols:
        assert result[col].isna().sum() == 0, f"NaNs found in non-lag column: {col}"


def test_feature_pipeline_synthetic_large_data() -> None:
    """Test the pipeline with enough data to produce non-empty output and valid features."""
    # Create 400 periods (200 hours = ~8 days)
    periods = pd.date_range("2024-01-01", periods=400, freq="30min", tz="UTC")

    carbon_df = pd.DataFrame({"settlement_period": periods, "intensity_actual": [150.0] * 400, "intensity_forecast": [150.0] * 400})

    gen_df = pd.DataFrame(
        {
            "settlement_period": periods,
            "gas": [1000.0] * 400,
            "coal": [0.0] * 400,
            "nuclear": [500.0] * 400,
            "wind": [300.0] * 400,
            "hydro": [10.0] * 400,
            "imports": [5.0] * 400,
            "biomass": [20.0] * 400,
            "other": [2.0] * 400,
            "solar": [50.0] * 400,
        }
    )

    # Weather: hourly data, 3 cities
    weather_periods = pd.date_range("2024-01-01", periods=201, freq="h", tz="UTC")
    weather_rows = []
    for city in ["London", "Manchester", "Edinburgh"]:
        for p in weather_periods:
            weather_rows.append({"city": city, "timestamp": p, "temperature": 15.0, "wind_speed": 10.0, "radiation": 100.0})
    weather_df = pd.DataFrame(weather_rows)

    result = feature_pipeline(carbon_df, gen_df, weather_df)

    # We have 400 rows.
    # Rolling (min_periods=48) means rows 0-46 are NaN in rolling columns.
    # Lags (max_lag=336) means rows 0-335 are NaN in lag columns.
    # pipeline.py drops NaNs in non-lag columns (including rolling).
    # So we should have 400 - 47 = 353 rows left.
    assert len(result) == 353

    # Check that rolling columns are NOT NaN
    assert not result["carbon_intensity_7d_avg"].isna().any()

    # Check that lag columns have some values (but could be NaN if they were dropped by accident,
    # but they shouldn't be because we only drop on subset=non_lag_cols)
    # At index 336 (absolute) of the original 400, lags should be valid.
    # In the result (starting at index 47 of original), index 336-47 = 289 should be the first valid lag row.
    assert result["carbon_intensity_lag_336"].iloc[289:].notna().all()
    assert result["carbon_intensity_lag_336"].iloc[:289].isna().all()
