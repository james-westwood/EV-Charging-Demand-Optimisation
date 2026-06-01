"""Dumb charging baseline for comparison with LP optimizer."""
import numpy as np


def dumb_charging(
    required_kwh: float,
    max_rate_kw: float,
    n_slots: int,
) -> np.ndarray:
    """Generate a naive charging schedule: charge at max power until done.

    This is the "plug in and charge immediately at full power" strategy.
    Used as a baseline to measure how much the LP optimizer improves over naive charging.

    Args:
        required_kwh: Total energy needed (kWh)
        max_rate_kw: Maximum charging power (kW)
        n_slots: Number of 30-min periods available

    Returns:
        Numpy array of charge rates (kW) for each period.
        Each slot contains max_rate_kw until total energy is met, then 0.
        The last charging slot may be partial if the total energy requirement
        doesn't align exactly with a slot boundary.

    Example:
        >>> dumb_charging(required_kwh=10, max_rate_kw=4, n_slots=10)
        array([4., 4., 2., 0., 0., 0., 0., 0., 0., 0.])
        # Explanation: 4 kW × 0.5h = 2 kWh, twice = 4 kWh, then 2 kW × 0.5h = 1 kWh.
        # Total: 4 + 4 + 2 = 10 kWh in first 3 slots, then 0 for the rest.
    """
    schedule = np.zeros(n_slots)
    energy_accumulated = 0.0

    for t in range(n_slots):
        if energy_accumulated >= required_kwh:
            # Already charged enough, rest of slots are zero
            break

        remaining_energy = required_kwh - energy_accumulated
        # Max energy we can charge in this slot: max_rate_kw * 0.5 hours
        max_energy_this_slot = max_rate_kw * 0.5

        if remaining_energy >= max_energy_this_slot:
            # Charge at full power this slot
            schedule[t] = max_rate_kw
            energy_accumulated += max_energy_this_slot
        else:
            # Partial slot: charge only what's needed
            # remaining_energy = charge_rate * 0.5, so charge_rate = remaining_energy / 0.5
            schedule[t] = remaining_energy / 0.5
            energy_accumulated += remaining_energy

    return schedule
