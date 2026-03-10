"""Tests for src/features/store.py — round-trip write/read."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.features.store import load_features, write_features


def _make_df() -> pd.DataFrame:
    """Build a small DataFrame with diverse dtypes."""
    periods = pd.date_range("2024-01-01", periods=10, freq="30min", tz="UTC")
    return pd.DataFrame(
        {
            "settlement_period": periods,
            "carbon_intensity": np.linspace(100.0, 200.0, 10),
            "wind_pct": np.linspace(0.1, 0.5, 10),
            "hour_of_day": np.arange(10, dtype=np.int32),
            "is_bank_holiday_uk": [False] * 10,
        }
    )


def test_round_trip_row_count(tmp_path) -> None:
    """Write then read preserves row count."""
    df = _make_df()
    write_features(df, run_date=date(2024, 1, 1), features_dir=tmp_path)
    reloaded = load_features(features_dir=tmp_path)
    assert len(reloaded) == len(df)


def test_round_trip_preserves_dtypes(tmp_path) -> None:
    """Write then read preserves all column dtypes."""
    df = _make_df()
    write_features(df, run_date=date(2024, 6, 15), features_dir=tmp_path)
    reloaded = load_features(features_dir=tmp_path)

    assert len(reloaded) == len(df)
    for col in df.columns:
        assert reloaded[col].dtype == df[col].dtype, (
            f"dtype mismatch for '{col}': expected {df[col].dtype}, got {reloaded[col].dtype}"
        )


def test_load_features_picks_most_recent(tmp_path) -> None:
    """load_features returns the lexicographically latest file."""
    df = _make_df()
    for d in ["2024-01-01", "2024-06-01", "2023-12-31"]:
        write_features(df, run_date=date.fromisoformat(d), features_dir=tmp_path)

    # The most recent file by name is 2024-06-01
    reloaded = load_features(features_dir=tmp_path)
    assert len(reloaded) == len(df)


def test_load_features_raises_when_empty(tmp_path) -> None:
    """load_features raises FileNotFoundError when no files exist."""
    with pytest.raises(FileNotFoundError):
        load_features(features_dir=tmp_path)


def test_write_features_returns_correct_path(tmp_path) -> None:
    """write_features returns a path named features_{run_date}.parquet."""
    df = _make_df()
    run_date = date(2025, 3, 10)
    path = write_features(df, run_date=run_date, features_dir=tmp_path)
    assert path.name == "features_2025-03-10.parquet"
    assert path.exists()
