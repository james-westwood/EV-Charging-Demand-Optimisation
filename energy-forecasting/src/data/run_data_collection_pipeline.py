"""Fetch historical data in chunked API calls and store in a local DuckDB database."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd

from src.data.collectors.carbon_intensity import fetch_carbon_intensity
from src.data.collectors.generation_mix import fetch_generation_mix
from src.data.collectors.weather import fetch_weather
from src.logging_config import get_logger

logger = get_logger(__name__)

CHUNK_DAYS   = 14
TARGET_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
TARGET_END   = datetime.now(timezone.utc)
DB_PATH      = Path("data/ev_charging.duckdb")


def get_conn() -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS carbon_intensity (
            settlement_period TIMESTAMPTZ PRIMARY KEY,
            intensity_actual  INTEGER,
            intensity_forecast INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS generation_mix (
            settlement_period TIMESTAMPTZ PRIMARY KEY,
            gas     DOUBLE, coal    DOUBLE, nuclear DOUBLE,
            wind    DOUBLE, hydro   DOUBLE, imports DOUBLE,
            biomass DOUBLE, other   DOUBLE, solar   DOUBLE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weather (
            city        VARCHAR,
            timestamp   TIMESTAMPTZ,
            temperature DOUBLE,
            wind_speed  DOUBLE,
            radiation   DOUBLE,
            PRIMARY KEY (city, timestamp)
        )
    """)
    return conn


def upsert(conn: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame):
    """Insert rows that don't already exist (deduplicate on settlement_period)."""
    conn.register("_staging", df)
    conn.execute(f"""
        INSERT OR IGNORE INTO {table}
        SELECT * FROM _staging
    """)
    conn.unregister("_staging")


def date_chunks(start: datetime, end: datetime, chunk_days: int):
    cursor = start
    while cursor < end:
        yield cursor, min(cursor + timedelta(days=chunk_days), end)
        cursor += timedelta(days=chunk_days)


def fetch_all(label: str, fetch_fn: Callable, conn: duckdb.DuckDBPyConnection,
              start: datetime, end: datetime):
    chunks = list(date_chunks(start, end, CHUNK_DAYS))
    total = 0

    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        logger.info("%s chunk %d/%d: %s → %s",
                    label, i, len(chunks), chunk_start.date(), chunk_end.date())
        try:
            df = fetch_fn(chunk_start, chunk_end)
            if not df.empty:
                upsert(conn, label, df)
                total += len(df)
                print(f"  [{i}/{len(chunks)}] {chunk_start.date()} → {chunk_end.date()} "
                      f"— {len(df)} rows")
        except Exception as e:
            logger.error("Failed %s chunk %d: %s", label, i, e)
            print(f"  [{i}/{len(chunks)}] FAILED: {e}")

    logger.info("%s: %d rows written to DuckDB", label, total)
    return total


if __name__ == "__main__":
    print(f"Fetching {TARGET_START.date()} → {TARGET_END.date()} "
          f"in {CHUNK_DAYS}-day chunks\n")

    conn = get_conn()

    print("--- Carbon Intensity ---")
    n_carbon = fetch_all("carbon_intensity", fetch_carbon_intensity,
                         conn, TARGET_START, TARGET_END)

    print("\n--- Generation Mix ---")
    n_gen = fetch_all("generation_mix", fetch_generation_mix,
                      conn, TARGET_START, TARGET_END)

    print("\n--- Weather ---")
    n_weather = fetch_all("weather", fetch_weather,
                          conn, TARGET_START, TARGET_END)
    print("\nDone.")
    print(f"  carbon_intensity: {n_carbon} rows")
    print(f"  generation_mix:   {n_gen} rows")
    print(f"  weather:          {n_weather} rows")
    print(f"  Database: {DB_PATH.resolve()}")

    # Quick sanity check
    print("\nSample carbon_intensity:")
    print(conn.execute("SELECT * FROM carbon_intensity ORDER BY settlement_period LIMIT 3").df())

    conn.close()
