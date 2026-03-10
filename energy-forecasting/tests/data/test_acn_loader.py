"""Tests for ACN-Data loader."""
import pandas as pd
import pytest
from src.data.collectors.acn_loader import load_acn_data

@pytest.fixture
def sample_acn_csv(tmp_path):
    """Create a sample ACN-Data CSV for testing."""
    csv_content = (
        "sessionID,stationID,connectionTime,disconnectTime,kWhDelivered\n"
        "sess_001,station_1,2024-01-01 08:00:00,2024-01-01 10:00:00,10.5\n"
        "sess_002,station_2,2024-01-01 09:00:00,2024-01-01 12:00:00,25.0\n"
        "sess_003,station_1,2024-01-01 13:00:00,2024-01-01 12:00:00,15.0\n" # Invalid: arrival > departure
    )
    file_path = tmp_path / "acn_data.csv"
    file_path.write_text(csv_content)
    return file_path


def test_load_acn_data(sample_acn_csv):
    """Test loading ACN-Data from CSV."""
    df = load_acn_data(sample_acn_csv)
    
    # Check columns
    expected_cols = ["session_id", "station_id", "arrival_time", "departure_time", "energy_kwh"]
    assert list(df.columns) == expected_cols
    
    # Check row count - one invalid row should be filtered out
    assert len(df) == 2
    
    # Check dtypes
    assert pd.api.types.is_datetime64_any_dtype(df["arrival_time"])
    assert pd.api.types.is_datetime64_any_dtype(df["departure_time"])
    assert pd.api.types.is_numeric_dtype(df["energy_kwh"])
    
    # Check values
    assert df.iloc[0]["session_id"] == "sess_001"
    assert df.iloc[0]["arrival_time"] < df.iloc[0]["departure_time"]
    assert df.iloc[0]["energy_kwh"] == 10.5


def test_load_acn_data_missing_cols(tmp_path):
    """Test loading CSV with missing columns raises ValueError."""
    csv_content = "sessionID,stationID,connectionTime\nsess_001,station_1,2024-01-01 08:00:00"
    file_path = tmp_path / "missing_cols.csv"
    file_path.write_text(csv_content)
    
    with pytest.raises(ValueError, match="CSV missing required columns"):
        load_acn_data(file_path)
