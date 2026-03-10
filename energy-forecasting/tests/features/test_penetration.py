"""Tests for src/features/penetration.py."""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.penetration import add_penetration_features


def _make_df(**kwargs: float) -> pd.DataFrame:
    """Create a single-row generation mix DataFrame from keyword arguments."""
    fuel_cols = ["gas", "coal", "nuclear", "wind", "hydro", "imports", "biomass", "other", "solar"]
    row = {col: kwargs.get(col, 0.0) for col in fuel_cols}
    return pd.DataFrame([row])


def test_wind_pct_solar_pct_other_pct_sums_to_100() -> None:
    """Acceptance criteria: wind_pct + solar_pct + other_pct sums to 100 ±0.01.

    other_pct is derived as 100 - wind_pct - solar_pct.
    """
    df = _make_df(gas=400.0, coal=100.0, nuclear=200.0, wind=300.0,
                  hydro=50.0, imports=25.0, biomass=75.0, other=50.0, solar=150.0)
    result = add_penetration_features(df)

    wind_pct = result["wind_pct"].iloc[0]
    solar_pct = result["solar_pct"].iloc[0]
    other_pct = 100.0 - wind_pct - solar_pct

    assert abs(wind_pct + solar_pct + other_pct - 100.0) < 0.01


def test_known_values_wind_pct() -> None:
    """wind = 200, total = 1000 → wind_pct = 20.0."""
    df = _make_df(gas=400.0, coal=200.0, nuclear=100.0, wind=200.0,
                  hydro=0.0, imports=0.0, biomass=0.0, other=100.0, solar=0.0)
    result = add_penetration_features(df)

    assert abs(result["wind_pct"].iloc[0] - 20.0) < 0.01


def test_known_values_solar_pct() -> None:
    """solar = 250, total = 1000 → solar_pct = 25.0."""
    df = _make_df(gas=400.0, coal=100.0, nuclear=100.0, wind=100.0,
                  hydro=0.0, imports=0.0, biomass=0.0, other=50.0, solar=250.0)
    result = add_penetration_features(df)

    assert abs(result["solar_pct"].iloc[0] - 25.0) < 0.01


def test_low_carbon_pct() -> None:
    """Low-carbon sources: nuclear=200, wind=200, hydro=50, solar=50, biomass=0.
    Total = 1000 → low_carbon_pct = 50.0.
    """
    df = _make_df(gas=300.0, coal=200.0, nuclear=200.0, wind=200.0,
                  hydro=50.0, imports=0.0, biomass=0.0, other=0.0, solar=50.0)
    result = add_penetration_features(df)

    assert abs(result["low_carbon_pct"].iloc[0] - 50.0) < 0.01


def test_multiple_rows() -> None:
    """Each row is independently calculated."""
    df = pd.DataFrame([
        {"gas": 800.0, "coal": 0.0, "nuclear": 0.0, "wind": 100.0,
         "hydro": 0.0, "imports": 0.0, "biomass": 0.0, "other": 100.0, "solar": 0.0},
        {"gas": 0.0, "coal": 0.0, "nuclear": 0.0, "wind": 500.0,
         "hydro": 0.0, "imports": 0.0, "biomass": 0.0, "other": 0.0, "solar": 500.0},
    ])
    result = add_penetration_features(df)

    assert abs(result["wind_pct"].iloc[0] - 10.0) < 0.01
    assert abs(result["wind_pct"].iloc[1] - 50.0) < 0.01
    assert abs(result["solar_pct"].iloc[1] - 50.0) < 0.01


def test_output_columns_present() -> None:
    """Result must contain wind_pct, solar_pct, and low_carbon_pct."""
    df = _make_df(gas=500.0, wind=300.0, solar=200.0)
    result = add_penetration_features(df)

    assert "wind_pct" in result.columns
    assert "solar_pct" in result.columns
    assert "low_carbon_pct" in result.columns


def test_original_columns_preserved() -> None:
    """Original columns must still be present in the result."""
    df = _make_df(gas=500.0, wind=300.0, solar=200.0)
    result = add_penetration_features(df)

    for col in df.columns:
        assert col in result.columns


def test_no_fuel_cols_returns_unchanged() -> None:
    """When no fuel columns are present, return unchanged DataFrame."""
    df = pd.DataFrame({"timestamp": ["2025-01-01"]})
    result = add_penetration_features(df)

    assert list(result.columns) == list(df.columns)
