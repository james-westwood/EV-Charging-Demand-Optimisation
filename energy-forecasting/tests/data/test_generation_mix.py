"""Tests for the generation mix collector."""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.collectors.generation_mix import fetch_generation_mix

_FUEL_COLUMNS = ["gas", "coal", "nuclear", "wind", "hydro", "imports", "biomass", "other", "solar"]

_MOCK_RESPONSE = {
    "data": [
        {
            "from": "2024-01-01T00:00Z",
            "to": "2024-01-01T00:30Z",
            "generationmix": [
                {"fuel": "gas", "perc": 30.5},
                {"fuel": "coal", "perc": 1.2},
                {"fuel": "nuclear", "perc": 20.1},
                {"fuel": "wind", "perc": 15.3},
                {"fuel": "hydro", "perc": 2.1},
                {"fuel": "imports", "perc": 5.0},
                {"fuel": "biomass", "perc": 4.5},
                {"fuel": "other", "perc": 0.3},
                {"fuel": "solar", "perc": 21.0},
            ],
        },
        {
            "from": "2024-01-01T00:30Z",
            "to": "2024-01-01T01:00Z",
            "generationmix": [
                {"fuel": "gas", "perc": 28.0},
                {"fuel": "coal", "perc": 1.0},
                {"fuel": "nuclear", "perc": 21.0},
                {"fuel": "wind", "perc": 17.0},
                {"fuel": "hydro", "perc": 2.5},
                {"fuel": "imports", "perc": 4.5},
                {"fuel": "biomass", "perc": 4.0},
                {"fuel": "other", "perc": 0.5},
                {"fuel": "solar", "perc": 21.5},
            ],
        },
    ]
}


@pytest.fixture()
def mock_httpx_get():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _MOCK_RESPONSE
    mock_resp.raise_for_status.return_value = None
    with patch("src.data.collectors.generation_mix.httpx.get", return_value=mock_resp) as m:
        yield m


def test_returns_dataframe(mock_httpx_get):
    df = fetch_generation_mix(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
    )
    assert isinstance(df, pd.DataFrame)


def test_correct_columns(mock_httpx_get):
    df = fetch_generation_mix(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
    )
    assert list(df.columns) == ["settlement_period"] + _FUEL_COLUMNS


def test_row_count(mock_httpx_get):
    df = fetch_generation_mix(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
    )
    assert len(df) == 2


def test_settlement_period_is_datetime(mock_httpx_get):
    df = fetch_generation_mix(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
    )
    assert pd.api.types.is_datetime64_any_dtype(df["settlement_period"])


def test_settlement_period_is_utc(mock_httpx_get):
    df = fetch_generation_mix(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
    )
    assert str(df["settlement_period"].dt.tz) == "UTC"


def test_fuel_columns_are_numeric(mock_httpx_get):
    df = fetch_generation_mix(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
    )
    for col in _FUEL_COLUMNS:
        assert pd.api.types.is_float_dtype(df[col]), f"{col} should be float"


def test_fuel_values(mock_httpx_get):
    df = fetch_generation_mix(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
    )
    assert df["gas"].iloc[0] == pytest.approx(30.5)
    assert df["wind"].iloc[0] == pytest.approx(15.3)


def test_url_contains_formatted_dates(mock_httpx_get):
    fetch_generation_mix(
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
    )
    call_url = mock_httpx_get.call_args[0][0]
    assert "2024-01-01T00:00Z" in call_url
    assert "2024-01-01T01:00Z" in call_url


def test_empty_response_returns_empty_df():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": []}
    mock_resp.raise_for_status.return_value = None
    with patch("src.data.collectors.generation_mix.httpx.get", return_value=mock_resp):
        df = fetch_generation_mix(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == ["settlement_period"] + _FUEL_COLUMNS


def test_naive_datetimes_treated_as_utc():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _MOCK_RESPONSE
    mock_resp.raise_for_status.return_value = None
    with patch("src.data.collectors.generation_mix.httpx.get", return_value=mock_resp) as m:
        fetch_generation_mix(datetime(2024, 1, 1), datetime(2024, 1, 1, 1))
        call_url = m.call_args[0][0]
        assert "2024-01-01T00:00Z" in call_url
