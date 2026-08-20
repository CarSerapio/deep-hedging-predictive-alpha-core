"""European call payoff engine for the baseline liability experiment.

This module maps simulated underlying price paths to the terminal liability
payoff ``Z = max(S_T - K, 0)`` used in the benchmark. The implementation is
vectorized across paths and assumes a single-asset European call with one cash
settlement at maturity.
"""

from __future__ import annotations

import numpy as np

from config import RuntimeConfig


def compute_payoff(
    path_tensor: np.ndarray,
    *,
    strike: float,
    product_type: str = "european_call",
) -> np.ndarray:
    """Dispatch to the supported terminal payoff for the benchmark liability.

    Args:
        path_tensor: Simulated spot paths with shape ``[n_paths, n_steps + 1]``.
        strike: Option strike in price units.
        product_type: Name of the liability payoff. Only ``"european_call"`` is
            implemented in the current benchmark.

    Returns:
        Payoff vector of shape ``[n_paths]``.

    Raises:
        ValueError: If the requested ``product_type`` is unsupported or if the
            delegated payoff validation fails.
    """

    normalized_type = product_type.lower()
    if normalized_type != "european_call":
        raise ValueError(f"Unsupported product_type: {product_type}")

    return european_call_payoff(path_tensor, strike=strike)


def european_call_payoff(path_tensor: np.ndarray, *, strike: float) -> np.ndarray:
    """Compute the terminal European call payoff pathwise.

    This implements the liability
    ``Z = max(S_T - K, 0)``
    used in the current benchmark baseline.

    Args:
        path_tensor: Spot paths of shape ``[n_paths, n_steps + 1]``.
        strike: Call strike ``K`` in price units.

    Returns:
        Payoff vector of shape ``[n_paths]`` with the same dtype family as the
        input path tensor.

    Raises:
        ValueError: If the path tensor is not two-dimensional, is empty, is not
            numeric, or if ``strike`` is non-positive.
    """

    _validate_inputs(path_tensor=path_tensor, strike=strike)

    # Only the terminal spot enters the European payoff, so intermediate hedge
    # times affect the liability indirectly through the learned or benchmark
    # trading strategy rather than through path-dependent exercise features.
    terminal_spot = np.asarray(path_tensor)[:, -1] # Extract the terminal spot prices from the path tensor, which are located in the last column of the 2D array. This represents the underlying asset prices at maturity for each simulated path.
    payoff = np.maximum(terminal_spot - strike, 0.0) # Compute the European call payoff for each path by taking the maximum of the difference between the terminal spot price and the strike price, or zero. This implements the payoff function Z = max(S_T - K, 0) for a European call option.
    return payoff.astype(np.asarray(path_tensor).dtype, copy=False) # Return the computed payoff vector with the same dtype as the input path tensor, ensuring that the output is a copy of the data rather than a view of the original array. This maintains consistency in data types across the computation pipeline.


def european_call_payoff_from_config(config: RuntimeConfig, path_tensor: np.ndarray) -> np.ndarray:
    """Compute the benchmark call payoff using the configured market strike.

    Args:
        config: Validated runtime configuration containing ``market.strike``.
        path_tensor: Spot paths of shape ``[n_paths, n_steps + 1]``.

    Returns:
        Payoff vector of shape ``[n_paths]``.
    """

    return european_call_payoff(path_tensor, strike=config.market.strike)


def _validate_inputs(*, path_tensor: np.ndarray, strike: float) -> None:
    """Validate the numerical assumptions behind the baseline payoff."""

    array = np.asarray(path_tensor) # Convert the input path tensor to a NumPy array to ensure consistent handling of the data type and shape for validation purposes. This allows for checking the properties of the array, such as its dimensions and numeric type, before proceeding with the payoff computation.
    if array.ndim != 2:
        raise ValueError("path_tensor must have shape [n_paths, n_steps + 1]")
    if array.shape[0] <= 0 or array.shape[1] <= 1:
        raise ValueError("path_tensor must contain at least one path and one forward time step")
    if not np.issubdtype(array.dtype, np.number): # Check if the data type of the array is a subtype of numeric types (e.g., integer, float). This ensures that the path tensor contains valid numerical values suitable for mathematical operations in the payoff computation.
        raise ValueError("path_tensor must be numeric")
    if strike <= 0:
        raise ValueError("strike must be positive")
