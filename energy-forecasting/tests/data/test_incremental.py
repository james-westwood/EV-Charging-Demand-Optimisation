"""Tests for incremental data fetcher."""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.data.collectors.incremental import get_missing_ranges


def test_get_missing_ranges_no_file(tmp_path: Path):
    """If file doesn't exist, return full range."""
    file_path = tmp_path / "missing.parquet"
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 3, 1, tzinfo=timezone.utc)

    ranges = get_missing_ranges(file_path, start, end)
    assert ranges == [(start, end)]


def test_get_missing_ranges_empty_file(tmp_path: Path):
    """If file is empty, return full range."""
    file_path = tmp_path / "empty.parquet"
    df = pd.DataFrame(columns=["settlement_period"])
    df.to_parquet(file_path)

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 3, 1, tzinfo=timezone.utc)

    ranges = get_missing_ranges(file_path, start, end)
    assert ranges == [(start, end)]


def test_get_missing_ranges_full_coverage(tmp_path: Path):
    """If range is fully covered, return empty list."""
    file_path = tmp_path / "full.parquet"
    dates = pd.date_range("2026-01-01", "2026-03-01", freq="30min", tz="UTC")
    df = pd.DataFrame({"settlement_period": dates})
    df.to_parquet(file_path)

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 3, 1, tzinfo=timezone.utc)

    ranges = get_missing_ranges(file_path, start, end)
    assert ranges == []


def test_get_missing_ranges_partial_after(tmp_path: Path):
    """If only start of range is covered, return tail."""
    file_path = tmp_path / "partial.parquet"
    # Existing data for January
    dates = pd.date_range("2026-01-01", "2026-01-31 23:30", freq="30min", tz="UTC")
    df = pd.DataFrame({"settlement_period": dates})
    df.to_parquet(file_path)

    # Request Jan to March
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 3, 31, 23, 30, tzinfo=timezone.utc)

    ranges = get_missing_ranges(file_path, start, end)

    assert len(ranges) == 1
    # Missing range starts from the last available record
    assert ranges[0][0] == dates.max().to_pydatetime()
    assert ranges[0][1] == end


def test_get_missing_ranges_partial_before(tmp_path: Path):
    """If only end of range is covered, return head."""
    file_path = tmp_path / "partial_before.parquet"
    # Existing data for February
    dates = pd.date_range("2026-02-01", "2026-02-28 23:30", freq="30min", tz="UTC")
    df = pd.DataFrame({"settlement_period": dates})
    df.to_parquet(file_path)

    # Request Jan to Feb
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 2, 28, 23, 30, tzinfo=timezone.utc)

    ranges = get_missing_ranges(file_path, start, end)

    assert len(ranges) == 1
    assert ranges[0][0] == start
    assert ranges[0][1] == dates.min().to_pydatetime()


def test_get_missing_ranges_before_and_after(tmp_path: Path):
    """If only middle of range is covered, return both ends."""
    file_path = tmp_path / "middle.parquet"
    # Existing data for February
    dates = pd.date_range("2026-02-01", "2026-02-28 23:30", freq="30min", tz="UTC")
    df = pd.DataFrame({"settlement_period": dates})
    df.to_parquet(file_path)

    # Request Jan to March
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 3, 31, 23, 30, tzinfo=timezone.utc)

    ranges = get_missing_ranges(file_path, start, end)

    assert len(ranges) == 2
    assert ranges[0] == (start, dates.min().to_pydatetime())
    assert ranges[1] == (dates.max().to_pydatetime(), end)


def test_get_missing_ranges_custom_column(tmp_path: Path):
    """Check that custom column names work (e.g. for weather)."""
    file_path = tmp_path / "weather.parquet"
    dates = pd.date_range("2026-01-01", "2026-01-05", freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": dates})
    df.to_parquet(file_path)

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 1, 10, tzinfo=timezone.utc)

    ranges = get_missing_ranges(file_path, start, end, column="timestamp")

    assert len(ranges) == 1
    assert ranges[0][0] == dates.max().to_pydatetime()
    assert ranges[0][1] == end
