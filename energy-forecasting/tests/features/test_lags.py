"""Tests for src/features/lags.py."""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.lags import add_lag_features


def test_lag_features_acceptance_criteria() -> None:
    """Test lag features against the acceptance criteria.

    Acceptance criteria: Series [1,2,3,...,10] -> lag_1 at index 5 equals 4.
    Note: To get lag_1 at index 5 to be 4, the original series at index 4 must be 4.
    This corresponds to a 0-indexed range(10).
    """
    df = pd.DataFrame({"target": list(range(10))})
    result = add_lag_features(df, ["target"])

    # Column name check
    assert "target_lag_1" in result.columns
    assert "target_lag_2" in result.columns
    assert "target_lag_48" in result.columns
    assert "target_lag_336" in result.columns

    # Value check at index 5
    # Original: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Lag 1:    [NaN, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    # Index 5 of Lag 1 is 4.
    assert result["target_lag_1"].iloc[5] == 4


def test_lag_features_multiple_columns() -> None:
    """Test lag features with multiple target columns."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
        }
    )
    result = add_lag_features(df, ["a", "b"])

    assert "a_lag_1" in result.columns
    assert "b_lag_1" in result.columns
    assert result["a_lag_1"].iloc[1] == 1
    assert result["b_lag_1"].iloc[1] == 10
    assert result["a_lag_2"].iloc[2] == 1
    assert result["b_lag_2"].iloc[2] == 10


def test_lag_features_missing_column() -> None:
    """Test lag features when a requested column is missing."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    # "b" is missing.
    result = add_lag_features(df, ["a", "b"])

    assert "a_lag_1" in result.columns
    assert "b_lag_1" not in result.columns


def test_lag_features_nans() -> None:
    """Test lag features with longer lags to check for NaNs."""
    # Data shorter than 48 to ensure many NaNs.
    df = pd.DataFrame({"target": range(10)})
    result = add_lag_features(df, ["target"])

    assert pd.isna(result["target_lag_48"].iloc[0])
    assert pd.isna(result["target_lag_48"].iloc[9])
    assert pd.isna(result["target_lag_336"].iloc[9])
