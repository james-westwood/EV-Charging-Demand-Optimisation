"""Script to generate test fixture Parquet files."""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

FIXTURES_DIR = Path(__file__).parent


def create_carbon_intensity():
    base = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    periods = [base + pd.Timedelta(minutes=30 * i) for i in range(10)]
    df = pd.DataFrame({
        "settlement_period": periods,
        "intensity_actual": [210, 205, 198, 192, 188, 183, 178, 175, 172, 168],
        "intensity_forecast": [212, 207, 200, 194, 190, 185, 180, 177, 174, 170],
    })
    df["settlement_period"] = pd.to_datetime(df["settlement_period"], utc=True)
    df.to_parquet(FIXTURES_DIR / "carbon_intensity.parquet", engine="pyarrow", index=False)
    print("Created carbon_intensity.parquet")


def create_generation_mix():
    base = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    periods = [base + pd.Timedelta(minutes=30 * i) for i in range(10)]
    rng = np.random.default_rng(42)
    n = 10
    df = pd.DataFrame({
        "settlement_period": periods,
        "gas": rng.uniform(5000, 8000, n),
        "coal": rng.uniform(0, 500, n),
        "nuclear": rng.uniform(4000, 5000, n),
        "wind": rng.uniform(2000, 6000, n),
        "hydro": rng.uniform(100, 400, n),
        "imports": rng.uniform(500, 1500, n),
        "biomass": rng.uniform(300, 700, n),
        "other": rng.uniform(50, 200, n),
        "solar": rng.uniform(0, 100, n),
    })
    df["settlement_period"] = pd.to_datetime(df["settlement_period"], utc=True)
    df.to_parquet(FIXTURES_DIR / "generation_mix.parquet", engine="pyarrow", index=False)
    print("Created generation_mix.parquet")


def create_weather():
    base = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    # 10 rows: cycle through 3 cities, different timestamps
    cities = ["London", "Manchester", "Edinburgh"]
    rows = []
    for i in range(10):
        city = cities[i % 3]
        ts = base + pd.Timedelta(hours=i // 3)
        rows.append({
            "city": city,
            "timestamp": ts,
            "temperature": 5.0 + i * 0.5,
            "wind_speed": 10.0 + i * 0.3,
            "radiation": max(0.0, 50.0 * (i - 3)),
        })
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.to_parquet(FIXTURES_DIR / "weather.parquet", engine="pyarrow", index=False)
    print("Created weather.parquet")


def create_ev_sessions():
    base = pd.Timestamp("2024-01-01 08:00:00", tz="UTC")
    rows = []
    for i in range(10):
        arrival = base + pd.Timedelta(hours=i * 2)
        departure = arrival + pd.Timedelta(hours=1 + i * 0.5)
        rows.append({
            "session_id": f"sess_{i:03d}",
            "station_id": f"station_{(i % 3) + 1}",
            "arrival_time": arrival,
            "departure_time": departure,
            "energy_kwh": 10.0 + i * 2.5,
        })
    df = pd.DataFrame(rows)
    df["arrival_time"] = pd.to_datetime(df["arrival_time"], utc=True)
    df["departure_time"] = pd.to_datetime(df["departure_time"], utc=True)
    df.to_parquet(FIXTURES_DIR / "ev_sessions.parquet", engine="pyarrow", index=False)
    print("Created ev_sessions.parquet")


if __name__ == "__main__":
    create_carbon_intensity()
    create_generation_mix()
    create_weather()
    create_ev_sessions()
    print("All fixtures created.")
