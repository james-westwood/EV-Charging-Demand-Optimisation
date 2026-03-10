"""Tests for src/features/weather_join.py."""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.weather_join import join_weather_to_grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE = pd.Timestamp("2024-01-01 00:00", tz="UTC")
T1 = pd.Timestamp("2024-01-01 01:00", tz="UTC")
T_HALF = pd.Timestamp("2024-01-01 00:30", tz="UTC")

CITIES = ["London", "Manchester", "Edinburgh"]


def _make_weather_df(
    timestamps: list[pd.Timestamp],
    cities: list[str] | None = None,
    temp_start: float = 10.0,
    temp_step: float = 2.0,
) -> pd.DataFrame:
    """Build a minimal long-format weather DataFrame."""
    if cities is None:
        cities = CITIES
    rows = []
    for i, ts in enumerate(timestamps):
        for city in cities:
            rows.append(
                {
                    "city": city,
                    "timestamp": ts,
                    "temperature": temp_start + i * temp_step,
                    "wind_speed": 5.0 + i * 1.0,
                    "radiation": float(i * 50),
                }
            )
    return pd.DataFrame(rows)


def _half_hour_grid(start: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="30min")


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------


def test_interpolated_value_at_half_hour_is_midpoint() -> None:
    """Hourly weather at 00:00 and 01:00 → value at 00:30 is midpoint ±0.01."""
    # Single city to keep it simple; two hourly observations.
    weather_df = pd.DataFrame(
        [
            {"city": "London", "timestamp": BASE, "temperature": 10.0, "wind_speed": 4.0, "radiation": 0.0},
            {"city": "London", "timestamp": T1, "temperature": 20.0, "wind_speed": 8.0, "radiation": 100.0},
        ]
    )

    grid = _half_hour_grid(BASE, 3)  # 00:00, 00:30, 01:00
    result = join_weather_to_grid(weather_df, grid)

    midpoint_row = result.loc[T_HALF]
    assert midpoint_row["london_temperature"] == pytest.approx(15.0, abs=0.01)
    assert midpoint_row["london_wind_speed"] == pytest.approx(6.0, abs=0.01)
    assert midpoint_row["london_radiation"] == pytest.approx(50.0, abs=0.01)


def test_original_hourly_values_preserved() -> None:
    """Values at the original hourly timestamps must be unchanged."""
    weather_df = pd.DataFrame(
        [
            {"city": "London", "timestamp": BASE, "temperature": 10.0, "wind_speed": 4.0, "radiation": 0.0},
            {"city": "London", "timestamp": T1, "temperature": 20.0, "wind_speed": 8.0, "radiation": 100.0},
        ]
    )

    grid = _half_hour_grid(BASE, 3)
    result = join_weather_to_grid(weather_df, grid)

    assert result.loc[BASE, "london_temperature"] == pytest.approx(10.0)
    assert result.loc[T1, "london_temperature"] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Wide-format column naming
# ---------------------------------------------------------------------------


def test_output_columns_wide_format_naming() -> None:
    """Output columns follow {city_slug}_{variable} convention."""
    weather_df = _make_weather_df([BASE, T1], cities=["London", "Manchester"])
    grid = _half_hour_grid(BASE, 3)
    result = join_weather_to_grid(weather_df, grid)

    expected_cols = {
        "london_temperature", "london_wind_speed", "london_radiation",
        "manchester_temperature", "manchester_wind_speed", "manchester_radiation",
    }
    assert expected_cols == set(result.columns)


def test_multi_city_interpolation() -> None:
    """Each city is interpolated independently."""
    weather_df = pd.DataFrame(
        [
            # London: temp goes 10 → 20
            {"city": "London", "timestamp": BASE, "temperature": 10.0, "wind_speed": 0.0, "radiation": 0.0},
            {"city": "London", "timestamp": T1, "temperature": 20.0, "wind_speed": 0.0, "radiation": 0.0},
            # Manchester: temp goes 5 → 15
            {"city": "Manchester", "timestamp": BASE, "temperature": 5.0, "wind_speed": 0.0, "radiation": 0.0},
            {"city": "Manchester", "timestamp": T1, "temperature": 15.0, "wind_speed": 0.0, "radiation": 0.0},
        ]
    )
    grid = _half_hour_grid(BASE, 3)
    result = join_weather_to_grid(weather_df, grid)

    assert result.loc[T_HALF, "london_temperature"] == pytest.approx(15.0, abs=0.01)
    assert result.loc[T_HALF, "manchester_temperature"] == pytest.approx(10.0, abs=0.01)


def test_city_name_with_space_slugified() -> None:
    """City names with spaces become underscore slugs (e.g. 'New York' → 'new_york')."""
    weather_df = pd.DataFrame(
        [
            {"city": "New York", "timestamp": BASE, "temperature": 10.0, "wind_speed": 3.0, "radiation": 0.0},
            {"city": "New York", "timestamp": T1, "temperature": 12.0, "wind_speed": 5.0, "radiation": 50.0},
        ]
    )
    grid = _half_hour_grid(BASE, 3)
    result = join_weather_to_grid(weather_df, grid)

    assert "new_york_temperature" in result.columns


def test_index_named_settlement_period() -> None:
    """The result index must be named 'settlement_period'."""
    weather_df = _make_weather_df([BASE, T1])
    grid = _half_hour_grid(BASE, 3)
    result = join_weather_to_grid(weather_df, grid)

    assert result.index.name == "settlement_period"


def test_result_has_same_length_as_grid() -> None:
    """Output row count must equal the grid length."""
    weather_df = _make_weather_df([BASE, T1, T1 + pd.Timedelta(hours=1)])
    grid = _half_hour_grid(BASE, 5)  # 00:00–02:00 at 30-min intervals
    result = join_weather_to_grid(weather_df, grid)

    assert len(result) == 5


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_missing_column_raises_value_error() -> None:
    """weather_df without required columns raises ValueError."""
    bad_df = pd.DataFrame({"city": ["London"], "timestamp": [BASE]})
    grid = _half_hour_grid(BASE, 2)

    with pytest.raises(ValueError, match="missing columns"):
        join_weather_to_grid(bad_df, grid)


def test_non_datetimeindex_grid_raises_type_error() -> None:
    """Passing a plain list as grid raises TypeError."""
    weather_df = _make_weather_df([BASE, T1])

    with pytest.raises(TypeError):
        join_weather_to_grid(weather_df, [BASE, T_HALF, T1])  # type: ignore[arg-type]


def test_empty_grid_raises_value_error() -> None:
    """An empty grid raises ValueError."""
    weather_df = _make_weather_df([BASE, T1])
    empty_grid = pd.DatetimeIndex([], dtype="datetime64[ns, UTC]", name="settlement_period")

    with pytest.raises(ValueError, match="empty"):
        join_weather_to_grid(weather_df, empty_grid)
