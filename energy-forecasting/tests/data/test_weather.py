"""Tests for the weather data collector."""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.collectors.weather import fetch_weather

_HOURLY_DATA = {
    "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
    "temperature_2m": [5.1, 4.8],
    "wind_speed_10m": [12.3, 11.0],
    "shortwave_radiation": [0.0, 0.0],
}

_MOCK_RESPONSE = {"hourly": _HOURLY_DATA}


def _make_mock_response(payload: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.json.return_value = payload
    mock_resp.raise_for_status.return_value = None
    return mock_resp


@pytest.fixture()
def mock_httpx_get():
    mock_resp = _make_mock_response(_MOCK_RESPONSE)
    with patch("src.data.collectors.weather.httpx.get", return_value=mock_resp) as m:
        yield m


def test_returns_dataframe(mock_httpx_get):
    from_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    to_dt = datetime(2024, 1, 2, tzinfo=timezone.utc)
    df = fetch_weather(from_dt, to_dt)
    assert isinstance(df, pd.DataFrame)


def test_correct_columns(mock_httpx_get):
    from_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    to_dt = datetime(2024, 1, 2, tzinfo=timezone.utc)
    df = fetch_weather(from_dt, to_dt)
    assert list(df.columns) == ["city", "timestamp", "temperature", "wind_speed", "radiation"]


def test_three_cities_present(mock_httpx_get):
    from_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    to_dt = datetime(2024, 1, 2, tzinfo=timezone.utc)
    df = fetch_weather(from_dt, to_dt)
    cities = set(df["city"].unique())
    assert cities == {"London", "Manchester", "Edinburgh"}


def test_timestamp_is_datetime(mock_httpx_get):
    from_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    to_dt = datetime(2024, 1, 2, tzinfo=timezone.utc)
    df = fetch_weather(from_dt, to_dt)
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_timestamp_is_utc(mock_httpx_get):
    from_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    to_dt = datetime(2024, 1, 2, tzinfo=timezone.utc)
    df = fetch_weather(from_dt, to_dt)
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_numeric_columns(mock_httpx_get):
    from_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    to_dt = datetime(2024, 1, 2, tzinfo=timezone.utc)
    df = fetch_weather(from_dt, to_dt)
    assert pd.api.types.is_float_dtype(df["temperature"])
    assert pd.api.types.is_float_dtype(df["wind_speed"])
    assert pd.api.types.is_float_dtype(df["radiation"])


def test_row_count(mock_httpx_get):
    # 3 cities × 2 timestamps each = 6 rows
    from_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    to_dt = datetime(2024, 1, 2, tzinfo=timezone.utc)
    df = fetch_weather(from_dt, to_dt)
    assert len(df) == 6


def test_temperature_values(mock_httpx_get):
    from_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    to_dt = datetime(2024, 1, 2, tzinfo=timezone.utc)
    df = fetch_weather(from_dt, to_dt)
    london_rows = df[df["city"] == "London"].reset_index(drop=True)
    assert london_rows["temperature"].iloc[0] == pytest.approx(5.1)


def test_api_called_for_each_city(mock_httpx_get):
    from_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    to_dt = datetime(2024, 1, 2, tzinfo=timezone.utc)
    fetch_weather(from_dt, to_dt)
    assert mock_httpx_get.call_count == 3


def test_empty_response_returns_empty_df():
    mock_resp = _make_mock_response({"hourly": {"time": [], "temperature_2m": [], "wind_speed_10m": [], "shortwave_radiation": []}})
    with patch("src.data.collectors.weather.httpx.get", return_value=mock_resp):
        from_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        to_dt = datetime(2024, 1, 2, tzinfo=timezone.utc)
        df = fetch_weather(from_dt, to_dt)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == ["city", "timestamp", "temperature", "wind_speed", "radiation"]


def test_naive_datetimes_handled():
    mock_resp = _make_mock_response(_MOCK_RESPONSE)
    with patch("src.data.collectors.weather.httpx.get", return_value=mock_resp) as m:
        fetch_weather(datetime(2024, 1, 1), datetime(2024, 1, 2))
        # Just verify it was called (no exception raised)
        assert m.call_count == 3
