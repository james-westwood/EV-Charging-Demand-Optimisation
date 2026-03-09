"""Tests for test fixture integrity."""
import pandas as pd
import pytest


def test_carbon_intensity_fixture(carbon_intensity_df):
    df = carbon_intensity_df
    assert set(df.columns) == {"settlement_period", "intensity_actual", "intensity_forecast"}
    assert len(df) == 10
    assert pd.api.types.is_datetime64_any_dtype(df["settlement_period"])


def test_generation_mix_fixture(generation_mix_df):
    df = generation_mix_df
    expected_cols = {
        "settlement_period", "gas", "coal", "nuclear", "wind",
        "hydro", "imports", "biomass", "other", "solar",
    }
    assert set(df.columns) == expected_cols
    assert len(df) == 10
    assert pd.api.types.is_datetime64_any_dtype(df["settlement_period"])


def test_weather_fixture(weather_df):
    df = weather_df
    assert set(df.columns) == {"city", "timestamp", "temperature", "wind_speed", "radiation"}
    assert len(df) == 10
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_ev_sessions_fixture(ev_sessions_df):
    df = ev_sessions_df
    assert set(df.columns) == {"session_id", "station_id", "arrival_time", "departure_time", "energy_kwh"}
    assert len(df) == 10
    assert pd.api.types.is_datetime64_any_dtype(df["arrival_time"])
    assert pd.api.types.is_datetime64_any_dtype(df["departure_time"])
    assert (df["departure_time"] > df["arrival_time"]).all()
