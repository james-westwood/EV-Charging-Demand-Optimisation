"""Loader for Caltech ACN-Data CSV."""
from pathlib import Path
import pandas as pd
from src.logging_config import get_logger

logger = get_logger(__name__)


def load_acn_data(file_path: str | Path) -> pd.DataFrame:
    """Load Caltech ACN-Data CSV from local path.

    Args:
        file_path: Path to the CSV file.

    Returns:
        DataFrame with normalized columns:
        [session_id, station_id, arrival_time, departure_time, energy_kwh]
    """
    logger.info(f"Loading ACN-Data from {file_path}")
    
    # ACN-Data CSV columns typically:
    # _id, connectionTime, disconnectTime, doneChargingTime, kWhDelivered, 
    # sessionID, siteID, spaceID, stationID, timezone, userID
    
    # Map from ACN columns to normalized names
    column_mapping = {
        "sessionID": "session_id",
        "stationID": "station_id",
        "connectionTime": "arrival_time",
        "disconnectTime": "departure_time",
        "kWhDelivered": "energy_kwh",
    }
    
    df = pd.read_csv(file_path)
    
    # Identify which columns exist in the CSV and map them
    cols_to_keep = [col for col in column_mapping.keys() if col in df.columns]
    df = df[cols_to_keep].rename(columns=column_mapping)
    
    # Check if all required normalized columns are present
    required_cols = ["session_id", "station_id", "arrival_time", "departure_time", "energy_kwh"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns in CSV: {missing_cols}")
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    
    # Parse datetimes - ACN usually in UTC or local with timezone column
    # If connectionTime/disconnectTime are strings like 'Wed, 25 Apr 2018 11:08:39 GMT'
    df["arrival_time"] = pd.to_datetime(df["arrival_time"], utc=True)
    df["departure_time"] = pd.to_datetime(df["departure_time"], utc=True)
    
    # Ensure arrival < departure
    invalid_rows = df[df["arrival_time"] >= df["departure_time"]]
    if not invalid_rows.empty:
        logger.warning(f"Found {len(invalid_rows)} rows where arrival_time >= departure_time. Filtering out.")
        df = df[df["arrival_time"] < df["departure_time"]]
    
    # Select final columns in order
    df = df[required_cols]
    
    logger.info(f"Successfully loaded {len(df)} sessions.")
    return df
