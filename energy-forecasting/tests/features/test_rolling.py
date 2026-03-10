"""Tests for src/features/rolling.py."""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.rolling import add_rolling_features


def test_rolling_averages_constant_values() -> None:
    """Test rolling averages with constant values.

    Acceptance criteria: 400 constant values -> rolling mean equals constant after period 48.
    """
    n = 400
    df = pd.DataFrame(
        {
            "wind_pct": [50.0] * n,
            "solar_pct": [20.0] * n,
            "carbon_intensity": [150.0] * n,
        }
    )

    result = add_rolling_features(df)

    # All expected columns should exist.
    assert "wind_pct_7d_avg" in result.columns
    assert "solar_pct_7d_avg" in result.columns
    assert "carbon_intensity_7d_avg" in result.columns

    # Rolling mean should be NaN before min_periods (48).
    # Since rolling() with min_periods=48, at index 46 (47th row) it should be NaN.
    # At index 47 (48th row), it should be 50.0.
    assert pd.isna(result["wind_pct_7d_avg"].iloc[46])
    assert result["wind_pct_7d_avg"].iloc[47] == pytest.approx(50.0)
    assert result["solar_pct_7d_avg"].iloc[47] == pytest.approx(20.0)
    assert result["carbon_intensity_7d_avg"].iloc[47] == pytest.approx(150.0)

    # After period 48, it should remain constant.
    assert result["wind_pct_7d_avg"].iloc[100] == pytest.approx(50.0)
    assert result["wind_pct_7d_avg"].iloc[399] == pytest.approx(50.0)


def test_rolling_averages_missing_column() -> None:
    """Test rolling averages when one of the target columns is missing."""
    df = pd.DataFrame({"wind_pct": [50.0] * 100})
    # solar_pct and carbon_intensity are missing.

    result = add_rolling_features(df)

    assert "wind_pct_7d_avg" in result.columns
    assert "solar_pct_7d_avg" not in result.columns
    assert "carbon_intensity_7d_avg" not in result.columns
    assert result["wind_pct_7d_avg"].iloc[47] == pytest.approx(50.0)


def test_rolling_averages_all_missing() -> None:
    """Test rolling averages when all target columns are missing."""
    df = pd.DataFrame({"other": [1.0] * 100})

    result = add_rolling_features(df)

    # Should just return the original DataFrame (or copy).
    assert list(result.columns) == ["other"]
    assert len(result) == 100


def test_rolling_averages_with_varying_data() -> None:
    """Test rolling averages with varying data to check calculation."""
    # Create 100 rows. Window is 336, min_periods is 48.
    # We should have valid rolling means from index 47 onwards.
    data = list(range(100))
    df = pd.DataFrame({"carbon_intensity": data})

    result = add_rolling_features(df)

    # At index 47, we have 48 points (0 to 47).
    # Mean should be sum(0..47) / 48 = (0+47)*48 / 2 / 48 = 23.5
    assert result["carbon_intensity_7d_avg"].iloc[47] == pytest.approx(23.5)

    # At index 99, we have 100 points (0..99).
    # Mean should be sum(0..99) / 100 = (0+99)*100 / 2 / 100 = 49.5
    assert result["carbon_intensity_7d_avg"].iloc[99] == pytest.approx(49.5)
