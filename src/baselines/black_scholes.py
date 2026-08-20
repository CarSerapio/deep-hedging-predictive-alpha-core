"""Black-Scholes benchmark pricing and delta computations.

This module belongs to the benchmark-hedging stage of the research pipeline.
It computes the classical European-call price and delta on the same discrete
hedge dates used by the Monte Carlo experiment, so later deep-hedging outputs
can be compared against a well-understood frictionless reference strategy.

Important assumptions:
- The option is a single-asset European call.
- The benchmark formulas use constant volatility and continuous compounding.
- The pathwise wrapper consumes simulated physical-measure paths but applies the
    usual Black-Scholes closed forms only as a hedge benchmark, not as a claim
    that the simulation itself is risk-neutral.
"""

from __future__ import annotations

from math import erf, sqrt

import numpy as np

from config import RuntimeConfig


def build_time_to_maturity_grid(
    maturity: float,
    n_steps: int,
    *,
    dtype: np.dtype | type[np.floating] = np.float32,
) -> np.ndarray:
    """Return the time-to-maturity values associated with hedge decisions.

    Args:
        maturity: Total option maturity in years.
        n_steps: Number of discrete hedge intervals.
        dtype: Floating-point dtype for the returned vector.

    Returns:
        Array of shape ``[n_steps]``. Entry ``tau[t]`` is the residual maturity
        seen by a hedge chosen at time index ``t`` and held over the next
        interval.

    Raises:
        ValueError: If ``maturity`` or ``n_steps`` is non-positive.
    """

    if maturity <= 0:
        raise ValueError("maturity must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    dt = maturity / n_steps
    time_grid = maturity - np.arange(n_steps, dtype=np.float64) * dt
    return time_grid.astype(dtype, copy=False)


def black_scholes_call_price(
    spot: float | np.ndarray,
    *,
    strike: float,
    time_to_maturity: float | np.ndarray,
    sigma: float | np.ndarray,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> np.ndarray:
    """Compute the Black-Scholes price of a European call.

    Args:
        spot: Scalar or array of underlying spot prices.
        strike: Call strike in price units.
        time_to_maturity: Residual maturity in years. Can broadcast against
            ``spot``.
        sigma: Annualized volatility. Can be scalar or broadcastable array.
        rate: Continuously compounded risk-free rate.
        dividend_yield: Continuously compounded dividend yield.

    Returns:
        Array broadcast over the supplied inputs containing call prices in the
        same currency units as ``spot`` and ``strike``.

    Raises:
        ValueError: If ``strike`` is non-positive, if any spot or volatility is
            non-positive, or if any maturity is negative.
    """

    spot_array, tau_array, sigma_array = _coerce_inputs(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        sigma=sigma,
    )

    intrinsic_value = np.maximum(spot_array - strike, 0.0)
    mature_mask = tau_array <= 0.0
    safe_tau = np.where(mature_mask, 1.0, tau_array)

    d1 = _d1(
        spot=spot_array,
        strike=strike,
        time_to_maturity=safe_tau,
        sigma=sigma_array,
        rate=rate,
        dividend_yield=dividend_yield,
    )
    d2 = d1 - sigma_array * np.sqrt(safe_tau)

    # Mature options are valued by intrinsic value so the broadcasted d1/d2
    # algebra never divides by zero at expiry.
    price = (
        spot_array * np.exp(-dividend_yield * safe_tau) * _normal_cdf(d1)
        - strike * np.exp(-rate * safe_tau) * _normal_cdf(d2)
    )
    return np.where(mature_mask, intrinsic_value, price)


def black_scholes_call_delta(
    spot: float | np.ndarray,
    *,
    strike: float,
    time_to_maturity: float | np.ndarray,
    sigma: float | np.ndarray,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> np.ndarray:
    """Compute the Black-Scholes delta of a European call.

    The delta is the benchmark hedge position in shares of the underlying held
    over the next trading interval.

    Args:
        spot: Scalar or array of underlying spot prices.
        strike: Call strike in price units.
        time_to_maturity: Residual maturity in years. Can broadcast against
            ``spot``.
        sigma: Annualized volatility. Can be scalar or broadcastable array.
        rate: Continuously compounded risk-free rate.
        dividend_yield: Continuously compounded dividend yield.

    Returns:
        Array broadcast over the supplied inputs containing hedge ratios between
        0 and 1 for a non-dividend-free call benchmark.

    Raises:
        ValueError: If the shared input validation fails.
    """

    spot_array, tau_array, sigma_array = _coerce_inputs(
        spot=spot,
        strike=strike,
        time_to_maturity=time_to_maturity,
        sigma=sigma,
    )

    mature_mask = tau_array <= 0.0 
    safe_tau = np.where(mature_mask, 1.0, tau_array)
    d1 = _d1(
        spot=spot_array,
        strike=strike,
        time_to_maturity=safe_tau,
        sigma=sigma_array,
        rate=rate,
        dividend_yield=dividend_yield,
    )
    delta = np.exp(-dividend_yield * safe_tau) * _normal_cdf(d1)

    # At maturity the call payoff kink is handled explicitly: 0 OTM, 1 ITM, and
    # 0.5 exactly at the strike as the standard symmetric convention.
    mature_delta = np.where(spot_array > strike, 1.0, np.where(spot_array < strike, 0.0, 0.5))
    return np.where(mature_mask, mature_delta, delta)


def black_scholes_price_and_delta(
    path_tensor: np.ndarray,
    *,
    strike: float,
    sigma: float | np.ndarray,
    maturity: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Black-Scholes prices and deltas along simulated paths.

    Args:
        path_tensor: Spot paths of shape ``[n_paths, n_steps + 1]``.
        strike: Call strike in price units.
        sigma: Annualized volatility used in the benchmark formula.
        maturity: Total maturity in years.
        rate: Continuously compounded risk-free rate.
        dividend_yield: Continuously compounded dividend yield.

    Returns:
        Tuple ``(prices, deltas)`` where each array has shape
        ``[n_paths, n_steps]``. Column ``t`` corresponds to the hedge chosen at
        time ``t`` using ``path_tensor[:, t]`` and residual maturity
        ``tau[t]``.

    Raises:
        ValueError: If the path tensor does not follow the benchmark shape
            convention.
    """

    paths = np.asarray(path_tensor)
    if paths.ndim != 2:
        raise ValueError("path_tensor must have shape [n_paths, n_steps + 1]")
    if paths.shape[0] <= 0 or paths.shape[1] <= 1:
        raise ValueError("path_tensor must contain at least one path and one forward time step")

    n_steps = paths.shape[1] - 1
    tau = build_time_to_maturity_grid(maturity, n_steps, dtype=np.float64)
    # The final path column is excluded because the hedge chosen at index t is
    # held over [t, t + 1]; there is no new hedge decision at maturity.
    hedge_spots = paths[:, :-1].astype(np.float64, copy=False)

    prices = black_scholes_call_price(
        hedge_spots,
        strike=strike,
        time_to_maturity=tau[np.newaxis, :], # Broadcast the time-to-maturity array across the batch dimension of the hedge spots to ensure that each path's hedge decision uses the correct same residual maturity values for the Black-Scholes pricing formula. This allows for efficient computation of prices for all paths and time steps simultaneously.
        sigma=sigma,
        rate=rate,
        dividend_yield=dividend_yield,
    )
    deltas = black_scholes_call_delta(
        hedge_spots,
        strike=strike,
        time_to_maturity=tau[np.newaxis, :],
        sigma=sigma,
        rate=rate,
        dividend_yield=dividend_yield,
    )

    return prices.astype(paths.dtype, copy=False), deltas.astype(paths.dtype, copy=False)


def black_scholes_price_and_delta_from_config(
    config: RuntimeConfig,
    path_tensor: np.ndarray,
    *,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Black-Scholes price and delta tensors from ``RuntimeConfig``.

    Args:
        config: Validated runtime configuration supplying strike, volatility,
            and maturity.
        path_tensor: Spot paths of shape ``[n_paths, n_steps + 1]``.
        rate: Continuously compounded risk-free rate.
        dividend_yield: Continuously compounded dividend yield.

    Returns:
        Tuple ``(prices, deltas)`` with shape ``[n_paths, n_steps]`` for both
        arrays.
    """

    return black_scholes_price_and_delta(
        path_tensor,
        strike=config.market.strike,
        sigma=config.market.sigma,
        maturity=config.market.maturity,
        rate=rate,
        dividend_yield=dividend_yield,
    )


def _coerce_inputs(
    *,
    spot: float | np.ndarray,
    strike: float,
    time_to_maturity: float | np.ndarray,
    sigma: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coerce scalar or array inputs into a common floating-point representation."""

    if strike <= 0:
        raise ValueError("strike must be positive")

    spot_array = np.asarray(spot, dtype=np.float64)
    tau_array = np.asarray(time_to_maturity, dtype=np.float64)
    sigma_array = np.asarray(sigma, dtype=np.float64)

    if np.any(spot_array <= 0):
        raise ValueError("spot values must be positive")
    if np.any(sigma_array <= 0):
        raise ValueError("sigma values must be positive")
    if np.any(tau_array < 0):
        raise ValueError("time_to_maturity values must be non-negative")

    return spot_array, tau_array, sigma_array


def _d1(
    *,
    spot: np.ndarray,
    strike: float,
    time_to_maturity: np.ndarray,
    sigma: np.ndarray,
    rate: float,
    dividend_yield: float,
) -> np.ndarray:
    """Return the Black-Scholes ``d1`` term used in price and delta formulas."""

    numerator = np.log(spot / strike) + (rate - dividend_yield + 0.5 * sigma * sigma) * time_to_maturity
    denominator = sigma * np.sqrt(time_to_maturity)
    return numerator / denominator


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    """Approximate the standard normal CDF using ``math.erf`` without SciPy."""

    erf_vectorized = np.vectorize(erf, otypes=[np.float64])
    return 0.5 * (1.0 + erf_vectorized(x / sqrt(2.0)))