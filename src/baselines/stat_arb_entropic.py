"""Analytic entropic statistical-arbitrage benchmark.

This module implements the closed-form standalone trading position used by the
analytic benchmark decomposition,

``delta_t^SA = mu / (theta sigma^2 S_t)``.

The benchmark is pathwise but frictionless. It uses the spot observed at each
hedge time and returns the position held over the following interval.
"""

from __future__ import annotations

import numpy as np

from config import RuntimeConfig


def entropic_stat_arb_position(
    spot: float | np.ndarray,
    *,
    mu: float,
    theta: float,
    sigma: float,
) -> np.ndarray:
    """Compute the entropic statistical-arbitrage position.

    Args:
        spot: Current underlying spot ``S_t``. May be scalar or array-valued.
        mu: Annualized physical-measure drift ``mu``.
        theta: Positive entropic risk-aversion parameter ``theta``.
        sigma: Positive annualized volatility ``sigma``.

    Returns:
        Array broadcast over ``spot`` containing the benchmark position in
        shares of the underlying.

    Raises:
        ValueError: If ``spot`` contains non-positive values or if ``theta`` or
            ``sigma`` is not positive.
    """

    spot_array = np.asarray(spot, dtype=np.float64)
    if np.any(spot_array <= 0.0):
        raise ValueError("spot values must be positive")
    if theta <= 0.0:
        raise ValueError("theta must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    denominator = theta * sigma * sigma * spot_array
    return mu / denominator


def entropic_stat_arb_benchmark(
    path_tensor: np.ndarray,
    *,
    mu: float,
    theta: float,
    sigma: float,
) -> np.ndarray:
    """Compute the pathwise entropic statistical-arbitrage benchmark tensor.

    Args:
        path_tensor: Spot paths of shape ``[n_paths, n_steps + 1]``.
        mu: Annualized physical-measure drift ``mu``.
        theta: Positive entropic risk-aversion parameter ``theta``.
        sigma: Positive annualized volatility ``sigma``.

    Returns:
        Array of shape ``[n_paths, n_steps]``. Entry ``[:, t]`` is the
        standalone trading position chosen at hedge time ``t`` and held over
        the interval ``[t, t + 1]``.

    Raises:
        ValueError: If the path tensor does not match the benchmark shape
            convention or if the direct parameter validation fails.
    """

    paths = np.asarray(path_tensor)
    if paths.ndim != 2:
        raise ValueError("path_tensor must have shape [n_paths, n_steps + 1]")
    if paths.shape[0] <= 0 or paths.shape[1] <= 1:
        raise ValueError("path_tensor must contain at least one path and one forward time step")

    # As with the Black-Scholes benchmark, there is no new decision at the
    # terminal node; the final path column only closes the last holding period.
    hedge_spots = paths[:, :-1].astype(np.float64, copy=False)
    positions = entropic_stat_arb_position(
        hedge_spots,
        mu=mu,
        theta=theta,
        sigma=sigma,
    )
    return positions.astype(paths.dtype, copy=False)


def entropic_stat_arb_benchmark_from_config(
    config: RuntimeConfig,
    path_tensor: np.ndarray,
) -> np.ndarray:
    """Compute the entropic statistical-arbitrage tensor from config values.

    Args:
        config: Validated runtime configuration supplying ``market.mu``,
            ``market.sigma``, and ``risk.theta``.
        path_tensor: Spot paths of shape ``[n_paths, n_steps + 1]``.

    Returns:
        Array of shape ``[n_paths, n_steps]`` containing the entropic
        statistical-arbitrage benchmark positions.

    Raises:
        ValueError: If the config is not entropic or its entropic parameter is
            missing.
    """

    if config.risk.kind.lower() != "entropic":
        raise ValueError("entropic statistical-arbitrage benchmark requires risk.kind='entropic'")
    if config.risk.theta is None:
        raise ValueError("entropic statistical-arbitrage benchmark requires risk.theta")

    return entropic_stat_arb_benchmark(
        path_tensor,
        mu=config.market.mu,
        theta=config.risk.theta,
        sigma=config.market.sigma,
    )