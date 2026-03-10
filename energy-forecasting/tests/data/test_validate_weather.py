"""Tests for src/data/validators/weather.py."""
import pandas as pd
import pytest

from src.data.validators.exceptions import ValidationError
from src.data.validators.weather import validate_weather

_CITIES = ["London", "Manchester", "Edinburgh"]


def _make_df(
    n_timestamps: int = 3,
    start: str = "2024-01-01 00:00",
    temperature: float = 15.0,
    wind_speed: float = 20.0,
    radiation: float = 100.0,
) -> pd.DataFrame:
    """Build a valid weather DataFrame with n_timestamps × 3 cities rows."""
    timestamps = pd.date_range(start, periods=n_timestamps, freq="30min", tz="UTC")
    rows = []
    for ts in timestamps:
        for city in _CITIES:
            rows.append(
                {
                    "timestamp": ts,
                    "city": city,
                    "temperature": temperature,
                    "wind_speed": wind_speed,
                    "radiation": radiation,
                }
            )
    return pd.DataFrame(rows)


class TestValidCases:
    def test_valid_dataframe_passes(self):
        df = _make_df()
        validate_weather(df)  # should not raise

    def test_empty_dataframe_passes(self):
        df = pd.DataFrame(
            columns=["timestamp", "city", "temperature", "wind_speed", "radiation"]
        )
        validate_weather(df)

    def test_boundary_temperature_min(self):
        df = _make_df(temperature=-30.0)
        validate_weather(df)

    def test_boundary_temperature_max(self):
        df = _make_df(temperature=50.0)
        validate_weather(df)

    def test_boundary_wind_speed_zero(self):
        df = _make_df(wind_speed=0.0)
        validate_weather(df)

    def test_boundary_wind_speed_max(self):
        df = _make_df(wind_speed=150.0)
        validate_weather(df)

    def test_boundary_radiation_zero(self):
        df = _make_df(radiation=0.0)
        validate_weather(df)

    def test_null_temperature_is_skipped(self):
        df = _make_df()
        df.at[0, "temperature"] = None
        validate_weather(df)


class TestTemperatureValidation:
    def test_temperature_above_max_raises(self):
        df = _make_df(temperature=999)
        with pytest.raises(ValidationError) as exc_info:
            validate_weather(df)
        assert exc_info.value.field == "temperature"

    def test_temperature_below_min_raises(self):
        df = _make_df(temperature=-31.0)
        with pytest.raises(ValidationError) as exc_info:
            validate_weather(df)
        assert exc_info.value.field == "temperature"

    def test_single_bad_temperature_row_raises(self):
        df = _make_df()
        df.at[0, "temperature"] = 999
        with pytest.raises(ValidationError) as exc_info:
            validate_weather(df)
        assert exc_info.value.field == "temperature"

    def test_error_message_mentions_field(self):
        df = _make_df(temperature=999)
        with pytest.raises(ValidationError) as exc_info:
            validate_weather(df)
        assert "temperature" in str(exc_info.value)


class TestWindSpeedValidation:
    def test_wind_speed_above_max_raises(self):
        df = _make_df(wind_speed=151.0)
        with pytest.raises(ValidationError) as exc_info:
            validate_weather(df)
        assert exc_info.value.field == "wind_speed"

    def test_wind_speed_negative_raises(self):
        df = _make_df(wind_speed=-1.0)
        with pytest.raises(ValidationError) as exc_info:
            validate_weather(df)
        assert exc_info.value.field == "wind_speed"


class TestRadiationValidation:
    def test_radiation_negative_raises(self):
        df = _make_df(radiation=-0.1)
        with pytest.raises(ValidationError) as exc_info:
            validate_weather(df)
        assert exc_info.value.field == "radiation"

    def test_null_radiation_is_skipped(self):
        df = _make_df()
        df.at[0, "radiation"] = None
        validate_weather(df)


class TestCityValidation:
    def test_missing_city_raises(self):
        df = _make_df()
        # Remove all Edinburgh rows
        df = df[df["city"] != "Edinburgh"].reset_index(drop=True)
        with pytest.raises(ValidationError) as exc_info:
            validate_weather(df)
        assert exc_info.value.field == "city"

    def test_missing_city_for_one_timestamp_raises(self):
        df = _make_df(n_timestamps=2)
        # Drop Edinburgh only for the first timestamp
        first_ts = df["timestamp"].iloc[0]
        mask = (df["timestamp"] == first_ts) & (df["city"] == "Edinburgh")
        df = df[~mask].reset_index(drop=True)
        with pytest.raises(ValidationError) as exc_info:
            validate_weather(df)
        assert exc_info.value.field == "city"

    def test_all_cities_present_passes(self):
        df = _make_df(n_timestamps=5)
        validate_weather(df)

    def test_error_message_mentions_missing_city(self):
        df = _make_df()
        df = df[df["city"] != "Manchester"].reset_index(drop=True)
        with pytest.raises(ValidationError) as exc_info:
            validate_weather(df)
        assert "Manchester" in str(exc_info.value)
