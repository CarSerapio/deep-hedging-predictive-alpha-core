"""Discrete trading-gain, transaction-cost, and terminal PnL computations.

This module implements the self-financing accounting used by the benchmark and
its frictional extensions after a hedge path has been produced. The central
quantities are the discrete trading gain ``sum_t delta_t (S_{t+1} - S_t)``,
the proportional transaction-cost term ``sum_t c * S_t * |Delta delta_t|``,
and the terminal seller P&L ``-Z + (delta · S)_T - C_T(delta)``.

Important assumptions:
- Transaction costs default to zero in the current benchmark configs.
- ``hedge_tensor[:, t]`` is the position chosen at time index ``t`` and held
    over the interval from ``path_tensor[:, t]`` to ``path_tensor[:, t + 1]``.
- Positive terminal PnL is favorable for the liability writer.
"""

from __future__ import annotations

import numpy as np

from config import RuntimeConfig
from finance.costs import compute_proportional_transaction_cost_from_config
from payoffs import compute_payoff


def compute_trading_gain(path_tensor: np.ndarray, hedge_tensor: np.ndarray) -> np.ndarray:
    """Compute the discrete self-financing trading gain pathwise.

    Args:
        path_tensor: Spot paths of shape ``[n_paths, n_steps + 1]``.
        hedge_tensor: Hedge positions of shape ``[n_paths, n_steps]``. Entry
            ``hedge_tensor[:, t]`` is held over the interval from time ``t`` to
            ``t + 1``.

    Returns:
        Trading-gain vector of shape ``[n_paths]``.

    Raises:
        ValueError: If path and hedge tensors are misaligned or non-numeric.
    """

    paths, hedges, result_dtype = _validate_path_and_hedge_tensors(path_tensor, hedge_tensor)

    # This is the discrete-time analogue of the stochastic integral
    # ``(delta · S)_T`` evaluated on the experiment's hedge grid.
    spot_increments = np.diff(paths, axis=1)
    gains = np.sum(hedges * spot_increments, axis=1, dtype=np.float64)
    return gains.astype(result_dtype, copy=False)


def compute_terminal_pnl(
    path_tensor: np.ndarray,
    hedge_tensor: np.ndarray,
    payoff: np.ndarray,
    *,
    transaction_cost: np.ndarray | None = None,
) -> np.ndarray:
    """Compute terminal hedged PnL for the liability writer.

    Args:
        path_tensor: Spot paths of shape ``[n_paths, n_steps + 1]``.
        hedge_tensor: Hedge positions of shape ``[n_paths, n_steps]``.
        payoff: Liability payoff vector ``Z`` of shape ``[n_paths]``.

    Returns:
        Terminal PnL vector of shape ``[n_paths]`` under the sign convention
        ``-Z + (delta · S)_T - C_T(delta)``.

    Raises:
        ValueError: If any supplied tensor has an invalid shape or dtype.
    """

    paths, hedges, result_dtype = _validate_path_and_hedge_tensors(path_tensor, hedge_tensor)
    payoff_array = _validate_payoff(payoff, n_paths=paths.shape[0])

    portfolio_pnl = compute_portfolio_pnl(paths, hedges, transaction_cost=transaction_cost).astype(np.float64, copy=False)
    pnl = -payoff_array + portfolio_pnl
    return pnl.astype(result_dtype, copy=False)


def compute_portfolio_pnl(
    path_tensor: np.ndarray,
    hedge_tensor: np.ndarray,
    *,
    transaction_cost: np.ndarray | None = None,
) -> np.ndarray:
    """Compute no-liability terminal PnL with optional transaction costs."""

    paths, hedges, result_dtype = _validate_path_and_hedge_tensors(path_tensor, hedge_tensor)
    trading_gain = compute_trading_gain(paths, hedges).astype(np.float64, copy=False)
    cost_array = _validate_cost_vector(transaction_cost, n_paths=paths.shape[0])
    pnl = trading_gain - cost_array
    return pnl.astype(result_dtype, copy=False)


def compute_terminal_pnl_from_config(
    config: RuntimeConfig,
    path_tensor: np.ndarray,
    hedge_tensor: np.ndarray,
    *,
    product_type: str = "european_call",
) -> np.ndarray:
    """Compute terminal liability PnL using the configured payoff definition.

    Args:
        config: Validated runtime configuration supplying the strike.
        path_tensor: Spot paths of shape ``[n_paths, n_steps + 1]``.
        hedge_tensor: Hedge positions of shape ``[n_paths, n_steps]``.
        product_type: Liability name forwarded to the payoff dispatcher.

    Returns:
        Terminal PnL vector of shape ``[n_paths]``.
    """

    payoff = compute_payoff(
        path_tensor,
        strike=config.market.strike,
        product_type=product_type,
    )
    transaction_cost = compute_proportional_transaction_cost_from_config(config, path_tensor, hedge_tensor)
    return compute_terminal_pnl(path_tensor, hedge_tensor, payoff, transaction_cost=transaction_cost)


def compute_portfolio_pnl_from_config(
    config: RuntimeConfig,
    path_tensor: np.ndarray,
    hedge_tensor: np.ndarray,
) -> np.ndarray:
    """Compute no-liability terminal PnL using configured transaction costs."""

    transaction_cost = compute_proportional_transaction_cost_from_config(config, path_tensor, hedge_tensor)
    return compute_portfolio_pnl(path_tensor, hedge_tensor, transaction_cost=transaction_cost)


def _validate_path_and_hedge_tensors(
    path_tensor: np.ndarray,
    hedge_tensor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.dtype]:
    """Validate the shared path/hedge chronology used by the PnL layer."""

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

    # The combined dtype preserves the current numeric contract while allowing
    # inputs such as float32 paths and float64 benchmark hedges to interact.
    result_dtype = np.result_type(paths.dtype, hedges.dtype)
    return (
        paths.astype(result_dtype, copy=False),
        hedges.astype(result_dtype, copy=False),
        np.dtype(result_dtype),
    )


def _validate_payoff(payoff: np.ndarray, *, n_paths: int) -> np.ndarray:
    """Validate that the liability payoff matches the simulated path batch."""

    payoff_array = np.asarray(payoff)

    if payoff_array.ndim != 1:
        raise ValueError("payoff must have shape [n_paths]")
    if payoff_array.shape[0] != n_paths:
        raise ValueError("payoff must align with the number of simulated paths")
    if not np.issubdtype(payoff_array.dtype, np.number):
        raise ValueError("payoff must be numeric")

    return payoff_array.astype(np.float64, copy=False)


def _validate_cost_vector(transaction_cost: np.ndarray | None, *, n_paths: int) -> np.ndarray:
    if transaction_cost is None:
        return np.zeros(n_paths, dtype=np.float64)

    cost_array = np.asarray(transaction_cost)
    if cost_array.ndim != 1:
        raise ValueError("transaction_cost must have shape [n_paths]")
    if cost_array.shape[0] != n_paths:
        raise ValueError("transaction_cost must align with the number of simulated paths")
    if not np.issubdtype(cost_array.dtype, np.number):
        raise ValueError("transaction_cost must be numeric")
    return cost_array.astype(np.float64, copy=False)