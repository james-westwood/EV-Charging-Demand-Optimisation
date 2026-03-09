"""Tests for the carbon intensity collector."""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.collectors.carbon_intensity import fetch_carbon_intensity

_MOCK_RESPONSE = {
    "data": [
        {
            "from": "2024-01-01T00:00Z",
            "to": "2024-01-01T00:30Z",
            "intensity": {"forecast": 212, "actual": 210, "index": "moderate"},
        },
        {
            "from": "2024-01-01T00:30Z",
            "to": "2024-01-01T01:00Z",
            "intensity": {"forecast": 207, "actual": 205, "index": "moderate"},
        },
        {
            "from": "2024-01-01T01:00Z",
            "to": "2024-01-01T01:30Z",
            "intensity": {"forecast": 200, "actual": None, "index": "moderate"},
        },
    ]
}


@pytest.fixture()
def mock_httpx_get():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _MOCK_RESPONSE
    mock_resp.raise_for_status.return_value = None
    with patch("src.data.collectors.carbon_intensity.httpx.get", return_value=mock_resp) as m:
        yield m


def test_returns_dataframe(mock_httpx_get):
    from_dt = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    to_dt = datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc)
    df = fetch_carbon_intensity(from_dt, to_dt)
    assert isinstance(df, pd.DataFrame)


def test_correct_columns(mock_httpx_get):
    from_dt = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    to_dt = datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc)
    df = fetch_carbon_intensity(from_dt, to_dt)
    assert list(df.columns) == ["settlement_period", "intensity_actual", "intensity_forecast"]


def test_row_count(mock_httpx_get):
    from_dt = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    to_dt = datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc)
    df = fetch_carbon_intensity(from_dt, to_dt)
    assert len(df) == 3


def test_settlement_period_is_datetime(mock_httpx_get):
    from_dt = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    to_dt = datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc)
    df = fetch_carbon_intensity(from_dt, to_dt)
    assert pd.api.types.is_datetime64_any_dtype(df["settlement_period"])


def test_settlement_period_is_utc(mock_httpx_get):
    from_dt = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    to_dt = datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc)
    df = fetch_carbon_intensity(from_dt, to_dt)
    assert str(df["settlement_period"].dt.tz) == "UTC"


def test_intensity_values(mock_httpx_get):
    from_dt = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    to_dt = datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc)
    df = fetch_carbon_intensity(from_dt, to_dt)
    assert df["intensity_actual"].iloc[0] == 210
    assert df["intensity_forecast"].iloc[0] == 212


def test_url_contains_formatted_dates(mock_httpx_get):
    from_dt = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    to_dt = datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc)
    fetch_carbon_intensity(from_dt, to_dt)
    call_url = mock_httpx_get.call_args[0][0]
    assert "2024-01-01T00:00Z" in call_url
    assert "2024-01-01T02:00Z" in call_url


def test_empty_response_returns_empty_df():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": []}
    mock_resp.raise_for_status.return_value = None
    with patch("src.data.collectors.carbon_intensity.httpx.get", return_value=mock_resp):
        from_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        to_dt = datetime(2024, 1, 2, tzinfo=timezone.utc)
        df = fetch_carbon_intensity(from_dt, to_dt)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == ["settlement_period", "intensity_actual", "intensity_forecast"]


def test_naive_datetimes_treated_as_utc():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _MOCK_RESPONSE
    mock_resp.raise_for_status.return_value = None
    with patch("src.data.collectors.carbon_intensity.httpx.get", return_value=mock_resp) as m:
        fetch_carbon_intensity(datetime(2024, 1, 1), datetime(2024, 1, 1, 2))
        call_url = m.call_args[0][0]
        assert "2024-01-01T00:00Z" in call_url
