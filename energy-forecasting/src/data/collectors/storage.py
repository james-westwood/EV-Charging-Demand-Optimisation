import os
import pandas as pd
from datetime import date as date_type
from pathlib import Path
from src.logging_config import get_logger

logger = get_logger(__name__)

def save_raw_parquet(df: pd.DataFrame, source: str, date: date_type | str, base_path: str | Path = "data/raw") -> Path:
    """
    Save DataFrame to {base_path}/{source}/{YYYY-MM-DD}.parquet.
    If file exists, load, append new rows, deduplicate on settlement_period, write back.
    
    Args:
        df: DataFrame to save.
        source: Source name (e.g., 'carbon_intensity').
        date: Date for the filename (YYYY-MM-DD).
        base_path: Base directory for storage (default: 'data/raw').
        
    Returns:
        Path to the saved file.
    """
    if isinstance(date, date_type):
        date_str = date.isoformat()
    else:
        date_str = str(date)
        
    base_dir = Path(base_path) / source
    file_path = base_dir / f"{date_str}.parquet"
    
    # Create directory if needed
    base_dir.mkdir(parents=True, exist_ok=True)
    
    if file_path.exists():
        logger.info(f"File {file_path} exists. Appending and deduplicating.")
        existing_df = pd.read_parquet(file_path)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        # Deduplicate on settlement_period if it exists
        if "settlement_period" in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=["settlement_period"], keep="last")
        else:
            logger.warning(f"No 'settlement_period' column found in {file_path}. Skipping deduplication.")
        df_to_save = combined_df
    else:
        logger.info(f"Creating new file {file_path}.")
        df_to_save = df
        
    df_to_save.to_parquet(file_path, engine="pyarrow", index=False)
    logger.info(f"Successfully saved {len(df_to_save)} rows to {file_path}.")
    
    return file_path
