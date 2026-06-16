"""Calculate carbon and cost savings from LP optimization vs baseline."""


def carbon_saving_gco2(
    optimal_schedule: list[float],
    baseline_schedule: list[float],
    carbon_intensity: list[float],
) -> float:
    """Calculate carbon emissions saved by optimal vs baseline schedule.

    Args:
        optimal_schedule: Optimal charging rates (kW) from LP solver, one per slot
        baseline_schedule: Baseline charging rates (kW) from dumb_charging(), one per slot
        carbon_intensity: Carbon intensity (gCO2/kWh) for each slot

    Returns:
        Carbon saved in gCO2 (positive = optimal is better)

    Formula:
        baseline_carbon = Σ(baseline[t] * carbon[t] * 0.5)
        optimal_carbon = Σ(optimal[t] * carbon[t] * 0.5)
        savings = baseline_carbon - optimal_carbon
    """
    baseline_carbon = sum(
        baseline_schedule[t] * carbon_intensity[t] * 0.5
        for t in range(len(carbon_intensity))
    )
    optimal_carbon = sum(
        optimal_schedule[t] * carbon_intensity[t] * 0.5
        for t in range(len(carbon_intensity))
    )
    savings = baseline_carbon - optimal_carbon
    return savings


def pct_saving(optimal_carbon: float, baseline_carbon: float) -> float:
    """Calculate percentage carbon reduction.

    Args:
        optimal_carbon: Total carbon cost of optimal schedule (gCO2)
        baseline_carbon: Total carbon cost of baseline schedule (gCO2)

    Returns:
        Percentage reduction (0–100), or 0 if baseline is zero or negative
    """
    if baseline_carbon <= 0:
        return 0.0
    return ((baseline_carbon - optimal_carbon) / baseline_carbon) * 100
