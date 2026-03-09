"""Generation mix data collector for api.carbonintensity.org.uk."""
from datetime import datetime, timezone

import httpx
import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://api.carbonintensity.org.uk"
_FUEL_COLUMNS = ["gas", "coal", "nuclear", "wind", "hydro", "imports", "biomass", "other", "solar"]


def fetch_generation_mix(from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
    """Fetch half-hourly generation mix for a date range.

    Args:
        from_dt: Start datetime (UTC).
        to_dt: End datetime (UTC).

    Returns:
        DataFrame with columns [settlement_period, gas, coal, nuclear, wind,
        hydro, imports, biomass, other, solar]. All generation values in MW.
    """
    from_str = _fmt(from_dt)
    to_str = _fmt(to_dt)
    url = f"{_BASE_URL}/generation/{from_str}/{to_str}"

    logger.info("Fetching generation mix", extra={"from": from_str, "to": to_str})

    response = httpx.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()

    rows = []
    for entry in payload.get("data", []):
        period = pd.Timestamp(entry["from"], tz="UTC")
        fuel_map = {item["fuel"]: item["perc"] for item in entry.get("generationmix", [])}
        row = {"settlement_period": period}
        for col in _FUEL_COLUMNS:
            row[col] = fuel_map.get(col)
        rows.append(row)

    columns = ["settlement_period"] + _FUEL_COLUMNS
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        for col in _FUEL_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Fetched %d rows of generation mix data", len(df))
    return df


def _fmt(dt: datetime) -> str:
    """Format datetime as ISO 8601 UTC string for the API."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%MZ")
