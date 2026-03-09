"""Carbon intensity data collector for api.carbonintensity.org.uk."""
from datetime import datetime, timezone

import httpx
import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://api.carbonintensity.org.uk"


def fetch_carbon_intensity(from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
    """Fetch carbon intensity data for a date range.

    Args:
        from_dt: Start datetime (UTC).
        to_dt: End datetime (UTC).

    Returns:
        DataFrame with columns [settlement_period, intensity_actual, intensity_forecast].
        settlement_period is timezone-aware datetime (UTC).
    """
    from_str = _fmt(from_dt)
    to_str = _fmt(to_dt)
    url = f"{_BASE_URL}/intensity/{from_str}/{to_str}"

    logger.info("Fetching carbon intensity", extra={"from": from_str, "to": to_str})

    response = httpx.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()

    rows = []
    for entry in payload.get("data", []):
        period = pd.Timestamp(entry["from"], tz="UTC")
        intensity = entry.get("intensity", {})
        rows.append(
            {
                "settlement_period": period,
                "intensity_actual": intensity.get("actual"),
                "intensity_forecast": intensity.get("forecast"),
            }
        )

    df = pd.DataFrame(rows, columns=["settlement_period", "intensity_actual", "intensity_forecast"])
    if not df.empty:
        df["intensity_actual"] = df["intensity_actual"].astype("Int64")
        df["intensity_forecast"] = df["intensity_forecast"].astype("Int64")

    logger.info("Fetched %d rows of carbon intensity data", len(df))
    return df


def _fmt(dt: datetime) -> str:
    """Format datetime as ISO 8601 UTC string for the API."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%MZ")
