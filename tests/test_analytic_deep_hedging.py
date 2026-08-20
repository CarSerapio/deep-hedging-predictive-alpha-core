"""Validation tests for the analytic deep-hedging benchmark.

These tests verify the central analytic decomposition identity on the discrete
hedge grid: the analytic deep-hedging tensor must equal the pathwise sum of
the Black-Scholes benchmark and the entropic statistical-arbitrage benchmark.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baselines.analytic_deep_hedging import (  # noqa: E402
    analytic_deep_hedging_benchmark,
    analytic_deep_hedging_benchmark_from_config,
)
from baselines.black_scholes import black_scholes_price_and_delta_from_config  # noqa: E402
from baselines.stat_arb_entropic import entropic_stat_arb_benchmark_from_config  # noqa: E402
from config import load_config  # noqa: E402
from simulators.gbm import simulate_gbm_paths_from_config  # noqa: E402


class AnalyticDeepHedgingBenchmarkTests(unittest.TestCase):
    """Checks the pathwise analytic decomposition benchmark."""

    def test_pathwise_tensor_equals_sum_of_component_benchmarks(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "benchmark_entropic.yaml")
        paths = simulate_gbm_paths_from_config(config, n_paths=32, seed=19)

        benchmark = analytic_deep_hedging_benchmark_from_config(config, paths)
        _, black_scholes_delta = black_scholes_price_and_delta_from_config(config, paths)
        stat_arb_delta = entropic_stat_arb_benchmark_from_config(config, paths)

        self.assertTrue(np.array_equal(benchmark, black_scholes_delta + stat_arb_delta))

    def test_direct_function_matches_config_wrapper(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "benchmark_entropic.yaml")
        paths = np.array(
            [
                [100.0, 101.0, 98.0],
                [100.0, 99.0, 102.0],
            ],
            dtype=np.float32,
        )

        direct = analytic_deep_hedging_benchmark(
            paths,
            strike=config.market.strike,
            sigma=config.market.sigma,
            maturity=config.market.maturity,
            mu=config.market.mu,
            theta=config.risk.theta,
        )
        wrapped = analytic_deep_hedging_benchmark_from_config(config, paths)

        self.assertTrue(np.array_equal(direct, wrapped))
        self.assertEqual(wrapped.shape, (2, 2))
        self.assertEqual(wrapped.dtype, np.float32)

    def test_non_entropic_config_is_rejected(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "benchmark_cvar.yaml")
        paths = np.array([[100.0, 101.0]], dtype=np.float32)

        with self.assertRaises(ValueError):
            analytic_deep_hedging_benchmark_from_config(config, paths)


if __name__ == "__main__":
    unittest.main()