"""Shared pytest fixtures for the EV Charging Demand Optimisation test suite."""
from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def carbon_intensity_df() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES_DIR / "carbon_intensity.parquet", engine="pyarrow")


@pytest.fixture()
def generation_mix_df() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES_DIR / "generation_mix.parquet", engine="pyarrow")


@pytest.fixture()
def weather_df() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES_DIR / "weather.parquet", engine="pyarrow")


@pytest.fixture()
def ev_sessions_df() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES_DIR / "ev_sessions.parquet", engine="pyarrow")
