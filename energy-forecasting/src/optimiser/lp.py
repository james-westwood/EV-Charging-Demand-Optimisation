"""Linear programming solver for EV charging optimization."""
import numpy as np
from pulp import LpProblem, lpSum, LpMinimize, LpVariable, LpStatus

from src.optimiser.exceptions import InfeasibleScheduleError


def build_lp(
    carbon_forecast: list[float],
    required_kwh: float,
    max_rate_kw: float,
    n_slots: int,
) -> tuple[LpProblem, dict]:
    """Build the LP problem for EV charging optimization.

    Args:
        carbon_forecast: Carbon intensity (gCO2/kWh) for each 30-min period
        required_kwh: Total energy needed (kWh)
        max_rate_kw: Maximum charging power (kW)
        n_slots: Number of 30-min periods in the charging window

    Returns:
        Tuple of (LpProblem, charge_variables dict)
    """
    problem = LpProblem(name="minimize_total_carbon", sense=LpMinimize)

    # Decision variables: charge rate (kW) for each period
    charge = {}
    for t in range(n_slots):
        charge[t] = LpVariable(name=f"charge_{t}", lowBound=0, upBound=max_rate_kw)

    # Objective: minimize total carbon emissions
    # Each period is 0.5 hours, so energy = charge[t] * 0.5 (kWh)
    # Carbon cost = carbon_forecast[t] * energy = carbon_forecast[t] * charge[t] * 0.5
    problem += lpSum(
        [carbon_forecast[t] * charge[t] * 0.5 for t in range(n_slots)]
    ), "total_carbon"

    # Constraint: total energy must meet requirement
    problem += (
        lpSum([charge[t] * 0.5 for t in range(n_slots)]) >= required_kwh
    ), "min_energy_requirement"

    return problem, charge


def solve_lp(problem: LpProblem, charge_vars: dict) -> np.ndarray:
    """Solve the LP problem and extract the optimal schedule.

    Args:
        problem: LpProblem instance
        charge_vars: Dict of LpVariable objects indexed by period

    Returns:
        Numpy array of optimal charge rates (kW) for each period

    Raises:
        InfeasibleScheduleError: If the problem is infeasible or unbounded
    """
    status = problem.solve()

    if LpStatus[status] != "Optimal":
        raise InfeasibleScheduleError(
            f"LP solver returned status '{LpStatus[status]}': "
            "no feasible charging schedule exists for these constraints"
        )

    n_slots = len(charge_vars)
    schedule = np.array([charge_vars[t].varValue for t in range(n_slots)])

    return schedule
