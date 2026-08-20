"""Analytic benchmark hedges used to interpret later learned strategies.

The current baseline package exposes the Black-Scholes hedge and the entropic
statistical-arbitrage benchmark on the same discrete hedge grid used by the
simulated experiments.
"""

from .analytic_deep_hedging import (
    analytic_deep_hedging_benchmark,
    analytic_deep_hedging_benchmark_from_config,
)
from .black_scholes import (
    black_scholes_call_delta,
    black_scholes_call_price,
    black_scholes_price_and_delta,
    black_scholes_price_and_delta_from_config,
    build_time_to_maturity_grid,
)
from .stat_arb_entropic import (
    entropic_stat_arb_benchmark,
    entropic_stat_arb_benchmark_from_config,
    entropic_stat_arb_position,
)

__all__ = [
    "analytic_deep_hedging_benchmark",
    "analytic_deep_hedging_benchmark_from_config",
    "black_scholes_call_delta",
    "black_scholes_call_price",
    "black_scholes_price_and_delta",
    "black_scholes_price_and_delta_from_config",
    "build_time_to_maturity_grid",
    "entropic_stat_arb_benchmark",
    "entropic_stat_arb_benchmark_from_config",
    "entropic_stat_arb_position",
]