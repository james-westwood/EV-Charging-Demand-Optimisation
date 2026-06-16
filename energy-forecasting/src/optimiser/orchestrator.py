"""Orchestrator for EV charging optimization: LP + baseline + savings."""
from datetime import datetime, timedelta
from typing import Any
from math import ceil

import numpy as np

from src.optimiser.lp import build_lp, solve_lp
from src.optimiser.baseline import dumb_charging
from src.optimiser.savings import carbon_saving_gco2, pct_saving


def optimise_session(
    plugin_time: datetime,
    departure_time: datetime,
    energy_kwh: float,
    max_rate_kw: float,
    carbon_forecast: list[float],
) -> dict[str, Any]:
    """Main orchestrator: given a session and carbon forecast, optimize the schedule.

    ALGORITHM:
    1. Calculate n_slots = time from plugin_time to departure_time in 30-min periods
    2. Validate: carbon_forecast must have at least n_slots entries
    3. Call build_lp() to create the LP problem
    4. Call solve_lp() to get optimal_schedule (numpy array of kW per slot)
    5. Call dumb_charging() to get baseline_schedule
    6. Compute optimal_carbon = Σ(optimal[t] * carbon_forecast[t] * 0.5)
    7. Compute baseline_carbon = Σ(baseline[t] * carbon_forecast[t] * 0.5)
    8. Call carbon_saving_gco2() to get savings_g
    9. Call pct_saving() to get savings_pct
    10. Generate slots = list of ISO datetime strings for each period

    RETURN:
    {
        "schedule": list of optimal kW per slot (convert numpy array to list),
        "dumb_schedule": list of baseline kW per slot,
        "carbon_saving_g": float, gCO2 saved (rounded to 1 decimal),
        "pct_saving": float, percentage (rounded to 1 decimal),
        "optimal_carbon": float, total gCO2 for optimal (rounded to 1 decimal),
        "baseline_carbon": float, total gCO2 for baseline (rounded to 1 decimal),
        "slots": list[str], ISO timestamps for each 30-min slot
    }

    EDGE CASES TO HANDLE:
    - If departure_time <= plugin_time → raise ValueError
    - If carbon_forecast is shorter than n_slots → raise ValueError
    - If carbon_forecast is longer → trim to n_slots
    - Round all floats to 1 decimal for clean JSON output
    - Convert numpy arrays to Python lists before returning

    Args:
        plugin_time: When the car plugs in (UTC datetime)
        departure_time: When the car needs to leave (UTC datetime)
        energy_kwh: How much energy is needed (kWh, e.g. 30)
        max_rate_kw: Maximum charging power (kW, e.g. 11)
        carbon_forecast: Carbon intensity forecast (gCO2/kWh) for each 30-min
                        period from plugin_time onward. Length must be ≥ number
                        of periods until departure.

    Returns:
        Dict with keys: schedule, dumb_schedule, carbon_saving_g, pct_saving,
        optimal_carbon, baseline_carbon, slots

    Raises:
        ValueError: If departure_time <= plugin_time or carbon_forecast is too short
    """
    # TODO: STEP 1 — Validate inputs
    # Validate: Check departure_time > plugin_time
    if departure_time < plugin_time:
        raise ValueError("Departure time must be after plugin time")

    # Calculate time_window = departure_time - plugin_time
    time_window = departure_time - plugin_time

    # Calculate n_slots from the time delta
    n_slots = ceil(time_window.total_seconds() / (30 * 60))

    # Validate len(carbon_forecast) >= n_slots
    if not len(carbon_forecast) >= n_slots:
        raise ValueError("Carbon Forecast must be at least as long as the charging time.")

    # Trim carbon_forecast to exactly n_slots
    carbon_forecast = carbon_forecast[:n_slots]

    # Build and solve LP
    problem, charge_vars = build_lp(carbon_forecast=carbon_forecast,
                                    required_kwh=energy_kwh,
                                    max_rate_kw=max_rate_kw,
                                    n_slots=n_slots)
    optimal_schedule: np.ndarray = solve_lp(problem, charge_vars)

    # Generate baseline
    baseline_schedule: np.ndarray = dumb_charging(required_kwh=energy_kwh,
                                                  max_rate_kw=max_rate_kw,
                                                  n_slots=n_slots) 

    # Calculate carbon costs
    optimal_carbon = sum((o_s * c_f * 0.5) for o_s,c_f in zip(optimal_schedule, carbon_forecast))

    # baseline_carbon = sum of (baseline_schedule[t] * carbon_forecast[t] * 0.5)
    baseline_carbon = sum((o_s * c_f * 0.5) for o_s,c_f in zip(baseline_schedule, carbon_forecast))
    
    # Calculate savings
    savings_g = carbon_saving_gco2(optimal_schedule, baseline_schedule, carbon_forecast)
    savings_pct = pct_saving(optimal_carbon, baseline_carbon)

    # Generate slot timestamps
    # Create a list of ISO timestamp strings
    # Start with plugin_time, add 30 minutes for each slot, format as isoformat()
    iso_slot_stamps = [(plugin_time+(n*timedelta(minutes=30))).isoformat() for n in range(n_slots)]

    # Build and return response dict
    return {
        "schedule": optimal_schedule.tolist(),
        "dumb_schedule": baseline_schedule.tolist(),
        "carbon_saving_g": round(savings_g, 1),
        "pct_saving": round(savings_pct, 1),
        "optimal_carbon": round(optimal_carbon, 1),
        "baseline_carbon": round(baseline_carbon, 1),
        "slots": iso_slot_stamps
    }
