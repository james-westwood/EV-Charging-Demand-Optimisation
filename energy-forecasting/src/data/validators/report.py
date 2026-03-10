"""Validation report: run all validators row-by-row and collect errors."""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from src.data.validators.carbon_intensity import validate_carbon_intensity
from src.data.validators.ev_sessions import validate_ev_sessions
from src.data.validators.exceptions import ValidationError
from src.data.validators.generation_mix import validate_generation_mix
from src.data.validators.weather import validate_weather
from src.logging_config import get_logger

logger = get_logger(__name__)

# Maps source name to its validator function.
_VALIDATORS: dict[str, Callable[[pd.DataFrame], None]] = {
    "carbon_intensity": validate_carbon_intensity,
    "generation_mix": validate_generation_mix,
    "weather": validate_weather,
    "ev_sessions": validate_ev_sessions,
}


def validate_all(dataframes: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    """Run all validators and collect errors without raising.

    For each source DataFrame, each row is validated individually so that
    multiple bad rows produce multiple error messages.  Cross-row checks
    (duplicate timestamps, 30-min intervals, city coverage) are naturally
    skipped for single-row DataFrames and are therefore not reported here.

    Sources with no errors are omitted from the returned dict.

    Args:
        dataframes: Mapping of source name to DataFrame.  Known source names
            are ``carbon_intensity``, ``generation_mix``, ``weather``, and
            ``ev_sessions``.  Unknown source names are skipped with a warning.

    Returns:
        ``{source_name: [error_message, ...]}`` for every source that has at
        least one validation error.
    """
    report: dict[str, list[str]] = {}

    for source_name, df in dataframes.items():
        validator = _VALIDATORS.get(source_name)
        if validator is None:
            logger.warning(
                "No validator registered for source '%s' — skipping", source_name
            )
            continue

        errors: list[str] = []
        for idx in df.index:
            row_df = df.loc[[idx]]
            try:
                validator(row_df)
            except ValidationError as exc:
                errors.append(str(exc))

        if errors:
            report[source_name] = errors
            logger.info(
                "Source '%s': %d validation error(s) found", source_name, len(errors)
            )
        else:
            logger.info("Source '%s': all rows passed validation", source_name)

    return report
