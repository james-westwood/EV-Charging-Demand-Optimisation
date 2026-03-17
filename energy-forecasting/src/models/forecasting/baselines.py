import numpy as np


def persistence_baseline(series: np.ndarray, h: int = 48) -> np.ndarray:
    """Persistence (naive) baseline: predict that the value h periods ahead equals the current value.

    The simplest possible forecast — "tomorrow at 3pm will look like today at 3pm."
    Used as a lower bound; any useful model should beat this.

    Args:
        series: Array of observed values in chronological order.
        h:      Forecast horizon in settlement periods (30 min each).
                h=48 → 24 hours ahead, h=336 → 1 week ahead.

    Returns:
        Array of length len(series) - h. Element i predicts the value at position i+h.
    """
    return series[:-h]


def seasonal_naive_baseline(series: np.ndarray, h: int, season: int = 336) -> np.ndarray:
    """Seasonal naive baseline: predict that the value h periods ahead equals the value
    from the same point in the previous season.

    Stronger than persistence — accounts for weekly patterns. Default season of 336
    corresponds to one week of 30-minute settlement periods (48 × 7 = 336).

    Args:
        series: Array of observed values in chronological order.
        h:      Forecast horizon in settlement periods.
        season: Season length in periods. Default 336 = 1 week of half-hours.

    Returns:
        Array of length len(series) - season. Element i predicts the value at
        position i+season by looking back to position i+(season-h).
    """
    return series[season - h: len(series) - h]