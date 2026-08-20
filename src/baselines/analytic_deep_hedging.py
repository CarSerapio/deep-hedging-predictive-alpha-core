"""Analytic deep-hedging benchmark implied by Theorem 2.

This module belongs to the benchmark-hedging stage of the research pipeline.
It implements the pathwise analytic benchmark

``delta_t^DH,analytic = Delta_t + delta_t^SA``

using the existing Black-Scholes hedge and entropic statistical-arbitrage
benchmark on the shared discrete hedge grid.
"""

from __future__ import annotations

import numpy as np

from baselines.black_scholes import black_scholes_price_and_delta
from baselines.stat_arb_entropic import entropic_stat_arb_benchmark
from config import RuntimeConfig


def analytic_deep_hedging_benchmark(
    path_tensor: np.ndarray,
    *,
    strike: float,
    sigma: float,
    maturity: float,
    mu: float,
    theta: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> np.ndarray:
    """Compute the analytic deep-hedging benchmark tensor pathwise.

    Args:
        path_tensor: Spot paths of shape ``[n_paths, n_steps + 1]``.
        strike: Call strike used by the Black-Scholes benchmark.
        sigma: Annualized volatility.
        maturity: Total maturity in years.
        mu: Annualized physical-measure drift.
        theta: Positive entropic risk-aversion parameter.
        rate: Continuously compounded risk-free rate.
        dividend_yield: Continuously compounded dividend yield.

    Returns:
        Array of shape ``[n_paths, n_steps]`` containing the analytic benchmark
        hedge held over each trading interval.

    Raises:
        ValueError: If any delegated benchmark validation fails.
    """

    _, black_scholes_delta = black_scholes_price_and_delta(
        path_tensor,
        strike=strike,
        sigma=sigma,
        maturity=maturity,
        rate=rate,
        dividend_yield=dividend_yield,
    )
    stat_arb_delta = entropic_stat_arb_benchmark(
        path_tensor,
        mu=mu,
        theta=theta,
        sigma=sigma,
    )
    return black_scholes_delta + stat_arb_delta


def analytic_deep_hedging_benchmark_from_config(
    config: RuntimeConfig,
    path_tensor: np.ndarray,
    *,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> np.ndarray:
    """Compute the analytic deep-hedging benchmark from ``RuntimeConfig``.

    Args:
        config: Validated runtime configuration supplying market parameters and
            entropic risk aversion.
        path_tensor: Spot paths of shape ``[n_paths, n_steps + 1]``.
        rate: Continuously compounded risk-free rate.
        dividend_yield: Continuously compounded dividend yield.

    Returns:
        Array of shape ``[n_paths, n_steps]`` containing the analytic
        deep-hedging benchmark.

    Raises:
        ValueError: If the config does not provide an entropic risk parameter.
    """

    if config.risk.kind.lower() != "entropic":
        raise ValueError("analytic deep-hedging benchmark requires risk.kind='entropic'")
    if config.risk.theta is None:
        raise ValueError("analytic deep-hedging benchmark requires risk.theta")

    return analytic_deep_hedging_benchmark(
        path_tensor,
        strike=config.market.strike,
        sigma=config.market.sigma,
        maturity=config.market.maturity,
        mu=config.market.mu,
        theta=config.risk.theta,
        rate=rate,
        dividend_yield=dividend_yield,
    )