"""Risk-objective entry points for the benchmark pipeline.

This package evaluates scalar objective values from batches of terminal PnL
samples. The current implementation covers the entropic risk used throughout
the predictive alpha decomposition and a smoothed CVaR objective based on the
standard auxiliary-variable formulation.
"""

from .objectives import (
    compute_risk_objective,
    compute_risk_objective_from_config,
    cvar_risk,
    cvar_risk_objective,
    entropic_risk,
)

__all__ = [
    "compute_risk_objective",
    "compute_risk_objective_from_config",
    "cvar_risk",
    "cvar_risk_objective",
    "entropic_risk",
]