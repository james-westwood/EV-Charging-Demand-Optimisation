"""Generation penetration features for energy forecasting."""
from __future__ import annotations

import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

_FUEL_COLS = ["gas", "coal", "nuclear", "wind", "hydro", "imports", "biomass", "other", "solar"]
_LOW_CARBON_COLS = ["nuclear", "wind", "hydro", "solar", "biomass"]


def add_penetration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive generation penetration percentages.

    Computes:
    - wind_pct: wind / total_generation * 100
    - solar_pct: solar / total_generation * 100
    - low_carbon_pct: (nuclear + wind + hydro + solar + biomass) / total_generation * 100

    total_generation is the sum of all fuel columns present in _FUEL_COLS.

    Args:
        df: DataFrame containing generation mix columns.

    Returns:
        DataFrame with wind_pct, solar_pct, and low_carbon_pct columns added.
    """
    result = df.copy()

    present_fuel_cols = [c for c in _FUEL_COLS if c in df.columns]
    if not present_fuel_cols:
        logger.warning("No fuel columns found for penetration feature calculation")
        return result

    total = df[present_fuel_cols].sum(axis=1)

    if "wind" in df.columns:
        result["wind_pct"] = df["wind"] / total * 100
        logger.debug("Computed wind_pct")
    else:
        logger.warning("Column 'wind' not found for wind_pct calculation")

    if "solar" in df.columns:
        result["solar_pct"] = df["solar"] / total * 100
        logger.debug("Computed solar_pct")
    else:
        logger.warning("Column 'solar' not found for solar_pct calculation")

    present_low_carbon = [c for c in _LOW_CARBON_COLS if c in df.columns]
    result["low_carbon_pct"] = df[present_low_carbon].sum(axis=1) / total * 100
    logger.debug("Computed low_carbon_pct using columns: %s", present_low_carbon)

    return result
