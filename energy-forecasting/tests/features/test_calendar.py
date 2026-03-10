"""Tests for src/features/calendar.py."""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.calendar import add_calendar_features


def _make_df_from_index(dates: list[str]) -> pd.DataFrame:
    idx = pd.to_datetime(dates, utc=True)
    return pd.DataFrame({"value": range(len(dates))}, index=idx)


def _make_df_from_column(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"settlement_period": pd.to_datetime(dates, utc=True)})


# --- Acceptance criteria ---

def test_christmas_2025_is_bank_holiday_not_weekend() -> None:
    """2025-12-25 must be a bank holiday and NOT a weekend (it's a Thursday)."""
    df = _make_df_from_index(["2025-12-25"])
    result = add_calendar_features(df)

    assert result["is_bank_holiday_uk"].iloc[0] is True or result["is_bank_holiday_uk"].iloc[0] == True  # noqa: E712
    assert result["is_weekend"].iloc[0] is False or result["is_weekend"].iloc[0] == False  # noqa: E712


# --- Calendar value tests (DatetimeIndex) ---

def test_hour_of_day_range() -> None:
    dates = ["2024-06-01 00:00", "2024-06-01 13:30", "2024-06-01 23:00"]
    df = _make_df_from_index(dates)
    result = add_calendar_features(df)

    assert list(result["hour_of_day"]) == [0, 13, 23]


def test_day_of_week() -> None:
    # 2024-01-01 is Monday (0), 2024-01-06 is Saturday (5), 2024-01-07 is Sunday (6)
    df = _make_df_from_index(["2024-01-01", "2024-01-06", "2024-01-07"])
    result = add_calendar_features(df)

    assert list(result["day_of_week"]) == [0, 5, 6]


def test_month_values() -> None:
    df = _make_df_from_index(["2024-01-15", "2024-06-15", "2024-12-15"])
    result = add_calendar_features(df)

    assert list(result["month"]) == [1, 6, 12]


def test_is_weekend_true_for_saturday_sunday() -> None:
    df = _make_df_from_index(["2024-01-06", "2024-01-07"])  # Sat, Sun
    result = add_calendar_features(df)

    assert result["is_weekend"].all()


def test_is_weekend_false_for_weekdays() -> None:
    # Mon–Fri
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    df = _make_df_from_index(dates)
    result = add_calendar_features(df)

    assert not result["is_weekend"].any()


def test_new_years_day_2024_bank_holiday() -> None:
    df = _make_df_from_index(["2024-01-01"])
    result = add_calendar_features(df)

    assert result["is_bank_holiday_uk"].iloc[0]


def test_ordinary_weekday_not_bank_holiday() -> None:
    df = _make_df_from_index(["2024-03-15"])
    result = add_calendar_features(df)

    assert not result["is_bank_holiday_uk"].iloc[0]


# --- settlement_period column input ---

def test_settlement_period_column_input() -> None:
    df = _make_df_from_column(["2025-12-25"])
    result = add_calendar_features(df)

    assert result["is_bank_holiday_uk"].iloc[0]
    assert not result["is_weekend"].iloc[0]


# --- Error handling ---

def test_missing_datetime_source_raises() -> None:
    df = pd.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(ValueError, match="DatetimeIndex"):
        add_calendar_features(df)


# --- Output columns present ---

def test_output_columns_present() -> None:
    df = _make_df_from_index(["2024-06-01"])
    result = add_calendar_features(df)

    for col in ["hour_of_day", "day_of_week", "month", "is_weekend", "is_bank_holiday_uk"]:
        assert col in result.columns
