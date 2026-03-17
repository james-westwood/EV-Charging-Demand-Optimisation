"""Weather data collector using Open-Meteo archive API."""

from datetime import datetime, timezone

import httpx
import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

_CITIES = {
    "London": {"latitude": 51.5, "longitude": -0.1},
    "Manchester": {"latitude": 53.5, "longitude": -2.2},
    "Edinburgh": {"latitude": 55.9, "longitude": -3.2},
}


def fetch_weather(from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
    """Fetch hourly historical weather for London, Manchester, Edinburgh.

    Args:
        from_dt: Start datetime (UTC).
        to_dt: End datetime (UTC).

    Returns:
        Long-format DataFrame with columns [city, timestamp, temperature,
        wind_speed, radiation]. timestamp is timezone-aware UTC.
    """
    start_date = _fmt_date(from_dt)
    end_date = _fmt_date(to_dt)

    all_rows: list[dict] = []

    for city_name, coords in _CITIES.items():
        params = {
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation",
            "timezone": "UTC",
        }

        logger.info("Fetching weather for %s", city_name, extra={"city": city_name})

        response = httpx.get(_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        hourly = payload.get("hourly", {})
        times = hourly.get("time", [])
        temperatures = hourly.get("temperature_2m", [])
        wind_speeds = hourly.get("wind_speed_10m", [])
        radiations = hourly.get("shortwave_radiation", [])

        for i, time_str in enumerate(times):
            all_rows.append(
                {
                    "city": city_name,
                    "timestamp": pd.Timestamp(time_str, tz="UTC"),
                    "temperature": temperatures[i] if i < len(temperatures) else None,
                    "wind_speed": wind_speeds[i] if i < len(wind_speeds) else None,
                    "radiation": radiations[i] if i < len(radiations) else None,
                }
            )

    columns = ["city", "timestamp", "temperature", "wind_speed", "radiation"]
    df = pd.DataFrame(all_rows, columns=columns)  # type: ignore
    if not df.empty:
        df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
        df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce")
        df["radiation"] = pd.to_numeric(df["radiation"], errors="coerce")

    logger.info("Fetched %d weather rows across %d cities", len(df), len(_CITIES))
    return df


def _fmt_date(dt: datetime) -> str:
    """Format datetime as YYYY-MM-DD string for the API."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%d")
