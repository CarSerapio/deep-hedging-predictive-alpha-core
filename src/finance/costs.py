"""Transaction-cost utilities for frictional deep-hedging extensions.

The current extension surface implements proportional spot trading costs of the
form ``sum_t c * S_t * |Delta delta_t|`` without terminal liquidation cost.
"""

from __future__ import annotations

import numpy as np

from config import RuntimeConfig


def compute_proportional_transaction_cost(
    path_tensor: np.ndarray,
    hedge_tensor: np.ndarray,
    *,
    proportional_rate: float,
    initial_hedge: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Compute pathwise proportional transaction costs.

    Args:
        path_tensor: Spot paths of shape ``[n_paths, n_steps + 1]``.
        hedge_tensor: Hedge positions of shape ``[n_paths, n_steps]``.
        proportional_rate: Non-negative proportional cost coefficient ``c``.
        initial_hedge: Initial hedge held before the first decision. The first
            trade is charged on ``delta_0 - initial_hedge``.

    Returns:
        Pathwise transaction-cost vector with shape ``[n_paths]``.
    """

    paths, hedges, result_dtype = _validate_path_and_hedge_tensors(path_tensor, hedge_tensor)
    rate = float(proportional_rate)
    if rate < 0.0:
        raise ValueError("proportional_rate must be non-negative")

    initial = _coerce_initial_hedge(initial_hedge, n_paths=paths.shape[0], result_dtype=result_dtype)
    trades = np.empty_like(hedges, dtype=result_dtype)
    trades[:, 0] = hedges[:, 0] - initial
    if hedges.shape[1] > 1:
        trades[:, 1:] = hedges[:, 1:] - hedges[:, :-1]

    step_costs = rate * np.abs(trades) * paths[:, :-1]
    costs = np.sum(step_costs, axis=1, dtype=np.float64)
    return costs.astype(result_dtype, copy=False)


def compute_proportional_transaction_cost_from_config(
    config: RuntimeConfig,
    path_tensor: np.ndarray,
    hedge_tensor: np.ndarray,
    *,
    initial_hedge: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Compute proportional transaction costs using the runtime config."""

    return compute_proportional_transaction_cost(
        path_tensor,
        hedge_tensor,
        proportional_rate=config.costs.proportional_rate,
        initial_hedge=initial_hedge,
    )


def _validate_path_and_hedge_tensors(
    path_tensor: np.ndarray,
    hedge_tensor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.dtype]:
    paths = np.asarray(path_tensor)
    hedges = np.asarray(hedge_tensor)

    if paths.ndim != 2:
        raise ValueError("path_tensor must have shape [n_paths, n_steps + 1]")
    if hedges.ndim != 2:
        raise ValueError("hedge_tensor must have shape [n_paths, n_steps]")
    if paths.shape[0] <= 0 or paths.shape[1] <= 1:
        raise ValueError("path_tensor must contain at least one path and one forward time step")
    if hedges.shape[0] != paths.shape[0] or hedges.shape[1] != paths.shape[1] - 1:
        raise ValueError("hedge_tensor must align with path_tensor as [n_paths, n_steps]")
    if not np.issubdtype(paths.dtype, np.number):
        raise ValueError("path_tensor must be numeric")
    if not np.issubdtype(hedges.dtype, np.number):
        raise ValueError("hedge_tensor must be numeric")

    result_dtype = np.result_type(paths.dtype, hedges.dtype)
    return (
        paths.astype(result_dtype, copy=False),
        hedges.astype(result_dtype, copy=False),
        np.dtype(result_dtype),
    )


def _coerce_initial_hedge(
    initial_hedge: float | np.ndarray,
    *,
    n_paths: int,
    result_dtype: np.dtype,
) -> np.ndarray:
    initial = np.asarray(initial_hedge, dtype=result_dtype)
    if initial.ndim == 0:
        return np.full(n_paths, float(initial), dtype=result_dtype)
    if initial.ndim == 2 and initial.shape[1] == 1:
        initial = initial[:, 0]
    if initial.ndim != 1 or initial.shape[0] != n_paths:
        raise ValueError("initial_hedge must be scalar or have shape [n_paths]")
    return initial.astype(result_dtype, copy=False)