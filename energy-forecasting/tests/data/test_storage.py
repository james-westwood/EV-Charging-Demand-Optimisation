import os
import pandas as pd
import pytest
from datetime import date, datetime, timedelta
from pathlib import Path
from src.data.collectors.storage import save_raw_parquet

def test_save_raw_parquet_new_file(tmp_path):
    """Test saving to a new file."""
    source = "test_source"
    today = date(2024, 1, 1)
    
    df = pd.DataFrame({
        "settlement_period": [datetime(2024, 1, 1, 0, 0)],
        "value": [1.0]
    })
    
    file_path = save_raw_parquet(df, source, today, base_path=tmp_path)
    
    assert file_path.exists()
    assert file_path.name == "2024-01-01.parquet"
    
    saved_df = pd.read_parquet(file_path)
    pd.testing.assert_frame_equal(saved_df, df)

def test_save_raw_parquet_append_and_deduplicate(tmp_path):
    """
    Test that writing 10 rows twice results in 20 unique rows 
    (if they have unique settlement_periods).
    And that overlapping rows are deduplicated.
    """
    source = "test_source"
    today = "2024-01-01"
    
    # First 10 rows
    periods1 = [datetime(2024, 1, 1) + timedelta(minutes=30*i) for i in range(10)]
    df1 = pd.DataFrame({
        "settlement_period": periods1,
        "value": range(10)
    })
    
    save_raw_parquet(df1, source, today, base_path=tmp_path)
    
    # Second 10 rows (unique settlement periods)
    periods2 = [datetime(2024, 1, 1) + timedelta(minutes=30*i) for i in range(10, 20)]
    df2 = pd.DataFrame({
        "settlement_period": periods2,
        "value": range(10, 20)
    })
    
    file_path = save_raw_parquet(df2, source, today, base_path=tmp_path)
    
    saved_df = pd.read_parquet(file_path)
    assert len(saved_df) == 20
    assert saved_df["settlement_period"].is_unique
    
    # Third write with overlapping rows
    # 5 new rows, 5 overlapping rows (with updated values)
    periods3 = [datetime(2024, 1, 1) + timedelta(minutes=30*i) for i in range(15, 25)]
    df3 = pd.DataFrame({
        "settlement_period": periods3,
        "value": [99] * 10
    })
    
    file_path = save_raw_parquet(df3, source, today, base_path=tmp_path)
    saved_df = pd.read_parquet(file_path)
    
    # Total unique periods: 0-24 (25 periods)
    assert len(saved_df) == 25
    assert saved_df["settlement_period"].is_unique
    
    # Check that overlapping rows took the 'last' (newest) value
    overlap_row = saved_df[saved_df["settlement_period"] == datetime(2024, 1, 1, 7, 30)] # period 15
    assert overlap_row["value"].iloc[0] == 99

def test_save_raw_parquet_creates_dirs(tmp_path):
    """Test that it creates parent directories."""
    source = "nested/source"
    today = date(2024, 1, 1)
    df = pd.DataFrame({"settlement_period": [datetime(2024, 1, 1)], "value": [1]})
    
    file_path = save_raw_parquet(df, source, today, base_path=tmp_path)
    assert file_path.exists()
    assert (tmp_path / source).is_dir()

def test_save_raw_parquet_no_settlement_period(tmp_path):
    """Test saving when settlement_period is missing (should skip deduplication but still append)."""
    source = "no_period"
    today = "2024-01-01"
    df1 = pd.DataFrame({"value": [1]})
    df2 = pd.DataFrame({"value": [2]})
    
    save_raw_parquet(df1, source, today, base_path=tmp_path)
    file_path = save_raw_parquet(df2, source, today, base_path=tmp_path)
    
    saved_df = pd.read_parquet(file_path)
    assert len(saved_df) == 2
