"""Exceptions for the EV charging optimiser."""


class InfeasibleScheduleError(Exception):
    """Raised when the LP problem has no feasible solution."""
    pass
