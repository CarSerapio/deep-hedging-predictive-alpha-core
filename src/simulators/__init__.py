"""Market-simulation entry points for the benchmark pipeline.

This package currently exposes exact geometric Brownian motion simulation under
the physical measure. The exported functions provide the simulated price paths
that later modules transform into payoffs, benchmark hedges, and terminal PnL.
"""

from .gbm import SimulatedMarketData, simulate_gbm_paths, simulate_gbm_paths_from_config, simulate_market_data_from_config

__all__ = ["SimulatedMarketData", "simulate_gbm_paths", "simulate_gbm_paths_from_config", "simulate_market_data_from_config"]