"""Exact GBM path simulation for the baseline one-asset hedging experiment.

This module implements the single-asset geometric Brownian motion environment
used to generate Monte Carlo trajectories for benchmarking hedges and, later,
training deep hedging policies. The simulator works under the physical measure
to separate liability hedging from predictive-alpha effects in real-world
dynamics.

Important assumptions:
- The underlying follows lognormal dynamics with constant drift ``mu`` and
    constant volatility ``sigma`` over the full horizon.
- The output chronology is discrete and includes the initial spot in column 0,
    followed by one simulated price per hedge interval.
- The implementation uses the exact lognormal transition, not an Euler
    approximation, so only the time grid is discretized.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np

from config import RuntimeConfig


@dataclass(frozen=True)
class SimulatedMarketData:
    """Simulated spot paths plus optional predictive-signal observations."""

    path_tensor: np.ndarray
    predictive_signal_tensor: np.ndarray | None = None


def simulate_gbm_paths(
    *,
    s0: float,                              # Initial stock price
    mu: float,                              # Expected return (drift)
    sigma: float,                           # Annual volatility
    maturity: float,                        # Time to maturity in years
    n_steps: int,                           # Number of time steps
    n_paths: int,                           # Number of paths to simulate (Monte Carlo samples)
    seed: int | None = None,                # Optional RNG seed for reproducibility
    dtype: np.dtype | type[np.floating] = np.float32,   # Output precision
) -> np.ndarray:
    """Simulate exact geometric Brownian motion price paths.

    Implements the path generator used by the current simulated-data benchmark.
    The dynamics correspond to
    ``dS_t / S_t = mu dt + sigma dW_t``
    under the physical measure, with an exact lognormal transition over each
    discrete trading interval.

    Args:
        s0: Initial asset price ``S_0`` in currency units.
        mu: Annualized physical-measure drift.
        sigma: Annualized volatility.
        maturity: Time to maturity in years.
        n_steps: Number of discrete trading intervals on ``[0, maturity]``.
        n_paths: Number of independently simulated trajectories.
        seed: Optional NumPy RNG seed for reproducible Monte Carlo samples.
        dtype: Floating-point dtype used for the returned path tensor.

    Returns:
        Array of shape ``[n_paths, n_steps + 1]``. Column 0 stores the initial
        spot and each later column stores the next hedge-time price.

    Raises:
        ValueError: If a model parameter or path count violates the simulation
            assumptions.
        OverflowError: If the simulated log-prices cannot be represented safely
            in the requested output dtype.
    """

    _validate_inputs(s0=s0, sigma=sigma, maturity=maturity, n_steps=n_steps, n_paths=n_paths)

    dt = maturity / n_steps 
    rng = np.random.default_rng(seed) # Generate a random number generator instance using the provided seed for reproducibility.
    # The volatility correction converts arithmetic drift into the drift of
    # log-prices so exponentiating cumulative log-returns reproduces exact GBM.
    drift = (mu - 0.5 * sigma * sigma) * dt
    diffusion = sigma * np.sqrt(dt)

    # Generate standard normal random shocks for each path and time step, which will be used to simulate the stochastic component of the GBM paths. 
    # The shape of the shocks array is (n_paths, n_steps), where each row corresponds to a simulated path and each column corresponds to a time step.
    shocks = rng.standard_normal(size=(n_paths, n_steps), dtype=np.float64) 
    log_returns = drift + diffusion * shocks

    log_s0 = log(s0)
    log_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64) # Create an empty array to store the log-prices of the simulated paths, with the first column reserved for the initial log-price and subsequent columns for the cumulative log-returns.
    log_paths[:, 0] = log_s0 # Set the first column of the log_paths array to the initial log-price, which serves as the starting point for simulating the GBM paths.
    # Compute the cumulative sum of the log-returns along each path and add the initial log-price to obtain the log-prices at each time step. 
    # Store these values in the subsequent columns of the log_paths array.
    log_paths[:, 1:] = log_s0 + np.cumsum(log_returns, axis=1) 

    # The overflow check is done in log-space because that is where the exact
    # GBM transition is accumulated numerically.
    _raise_on_overflow(log_paths, dtype=np.dtype(dtype))

    paths = np.exp(log_paths, dtype=np.float64)
    return paths.astype(dtype, copy=False) # Return the simulated GBM paths as an array of the specified dtype, ensuring that the output is a copy of the data rather than a view of the original log_paths array.


def simulate_gbm_paths_with_predictive_signal(
    *,
    s0: float,
    mu: float,
    sigma: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    signal_phi: float,
    signal_innovation_scale: float,
    signal_drift_scale: float,
    signal_initial_value: float = 0.0,
    seed: int | None = None,
    dtype: np.dtype | type[np.floating] = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate GBM spot paths plus a predictive AR(1) drift signal.

    The signal value at hedge time ``t`` is observed before the return over the
    interval ``[t, t+1]`` and perturbs the physical drift by
    ``signal_drift_scale * signal_t``.
    """

    _validate_inputs(s0=s0, sigma=sigma, maturity=maturity, n_steps=n_steps, n_paths=n_paths)

    dt = maturity / n_steps
    rng = np.random.default_rng(seed)
    drift_scale = float(signal_drift_scale)
    signal_phi_value = float(signal_phi)
    innovation_scale = float(signal_innovation_scale)
    current_signal = np.full(n_paths, float(signal_initial_value), dtype=np.float64)
    signal_tensor = np.empty((n_paths, n_steps), dtype=np.float64)
    log_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    log_paths[:, 0] = log(s0)
    diffusion = sigma * np.sqrt(dt)

    for step in range(n_steps):
        signal_tensor[:, step] = current_signal
        drift = (mu + drift_scale * current_signal - 0.5 * sigma * sigma) * dt
        shocks = rng.standard_normal(size=n_paths, dtype=np.float64)
        log_paths[:, step + 1] = log_paths[:, step] + drift + diffusion * shocks
        innovations = rng.standard_normal(size=n_paths, dtype=np.float64)
        current_signal = signal_phi_value * current_signal + innovation_scale * innovations

    _raise_on_overflow(log_paths, dtype=np.dtype(dtype))
    paths = np.exp(log_paths, dtype=np.float64).astype(dtype, copy=False)
    return paths, signal_tensor.astype(dtype, copy=False)


def simulate_gbm_paths_from_config(
    config: RuntimeConfig,
    *,
    split: str = "train",
    n_paths: int | None = None,
    seed: int | None = None,
    dtype: np.dtype | type[np.floating] = np.float32,
) -> np.ndarray:
    """Simulate GBM paths from a validated runtime configuration.

    Args:
        config: Typed experiment configuration containing market and path
            settings.
        split: Requested data split name. Used only to select a default path
            count when ``n_paths`` is omitted.
        n_paths: Optional override for the number of simulated paths.
        seed: Optional RNG seed override. When omitted, ``config.paths.seed`` is
            used.
        dtype: Floating-point dtype for the returned path tensor.

    Returns:
        Array of shape ``[n_paths, n_steps + 1]`` aligned with the benchmark
        hedge chronology encoded in ``config.market``.

    Raises:
        ValueError: If ``split`` is unknown or if any delegated simulation
            assumptions are violated.
    """

    split_name = split.lower()
    if n_paths is None:
        n_paths = _resolve_split_path_count(config, split_name)
    if seed is None:
        seed = config.paths.seed

    return simulate_gbm_paths(
        s0=config.market.s0,
        mu=config.market.mu,
        sigma=config.market.sigma,
        maturity=config.market.maturity,
        n_steps=config.market.n_steps,
        n_paths=n_paths,
        seed=seed,
        dtype=dtype,
    )


def simulate_market_data_from_config(
    config: RuntimeConfig,
    *,
    split: str = "train",
    n_paths: int | None = None,
    seed: int | None = None,
    dtype: np.dtype | type[np.floating] = np.float32,
) -> SimulatedMarketData:
    """Simulate the configured market inputs, including optional signal paths."""

    if not config.signal.enabled:
        return SimulatedMarketData(
            path_tensor=simulate_gbm_paths_from_config(
                config,
                split=split,
                n_paths=n_paths,
                seed=seed,
                dtype=dtype,
            ),
            predictive_signal_tensor=None,
        )

    split_name = split.lower()
    resolved_n_paths = _resolve_split_path_count(config, split_name) if n_paths is None else int(n_paths)
    resolved_seed = config.paths.seed if seed is None else int(seed)
    path_tensor, predictive_signal_tensor = simulate_gbm_paths_with_predictive_signal(
        s0=config.market.s0,
        mu=config.market.mu,
        sigma=config.market.sigma,
        maturity=config.market.maturity,
        n_steps=config.market.n_steps,
        n_paths=resolved_n_paths,
        signal_phi=config.signal.ar1_phi,
        signal_innovation_scale=config.signal.innovation_scale,
        signal_drift_scale=config.signal.drift_scale,
        signal_initial_value=config.signal.initial_value,
        seed=resolved_seed,
        dtype=dtype,
    )
    return SimulatedMarketData(
        path_tensor=path_tensor,
        predictive_signal_tensor=predictive_signal_tensor,
    )


def _resolve_split_path_count(config: RuntimeConfig, split: str) -> int:
    """Map a split label to the configured Monte Carlo sample size."""

    if split == "train":
        return config.paths.train_paths
    if split in {"validation", "val"}:
        return config.paths.val_paths
    if split == "test":
        return config.paths.test_paths
    raise ValueError(f"Unknown split: {split}")


def _validate_inputs(*, s0: float, sigma: float, maturity: float, n_steps: int, n_paths: int) -> None:
    """Validate the scalar parameters of the GBM benchmark environment."""

    if s0 <= 0:
        raise ValueError("s0 must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if maturity <= 0:
        raise ValueError("maturity must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")


def _raise_on_overflow(log_paths: np.ndarray, *, dtype: np.dtype) -> None:
    """Guard against exponentiating values outside the safe dtype range."""

    finfo = np.finfo(dtype)
    max_log = np.log(finfo.max)
    min_log = np.log(finfo.tiny)

    if np.max(log_paths) >= max_log or np.min(log_paths) <= min_log:
        raise OverflowError(
            "Simulated log-prices exceed the safe numeric range for the requested dtype"
        )