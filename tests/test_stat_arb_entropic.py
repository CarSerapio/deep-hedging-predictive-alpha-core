"""Validation tests for the entropic statistical-arbitrage benchmark.

These tests verify the closed-form benchmark. They focus on the parameter
sensitivities emphasized by the methodology and on the pathwise tensor contract
shared with the other hedge benchmarks.
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

from baselines.stat_arb_entropic import (  # noqa: E402
    entropic_stat_arb_benchmark,
    entropic_stat_arb_benchmark_from_config,
    entropic_stat_arb_position,
)
from config import load_config  # noqa: E402


class EntropicStatArbBenchmarkTests(unittest.TestCase):
    """Checks the analytic standalone trading benchmark from Appendix B."""

    def test_position_is_proportional_to_drift(self) -> None:
        spot = np.array([80.0, 100.0, 120.0], dtype=np.float64)

        base = entropic_stat_arb_position(spot, mu=0.05, theta=1.0, sigma=0.2)
        doubled = entropic_stat_arb_position(spot, mu=0.10, theta=1.0, sigma=0.2)

        self.assertTrue(np.allclose(doubled, 2.0 * base))

    def test_position_is_inversely_proportional_to_spot(self) -> None:
        spot = np.array([80.0, 100.0, 160.0], dtype=np.float64)
        position = entropic_stat_arb_position(spot, mu=0.05, theta=1.0, sigma=0.2)

        self.assertTrue(np.allclose(position * spot, np.full_like(spot, position[0] * spot[0])))
        self.assertTrue(np.all(np.diff(position) < 0.0))

    def test_position_has_inverse_quadratic_dependence_on_sigma(self) -> None:
        spot = np.array([100.0], dtype=np.float64)

        low_sigma = entropic_stat_arb_position(spot, mu=0.05, theta=1.0, sigma=0.2)
        high_sigma = entropic_stat_arb_position(spot, mu=0.05, theta=1.0, sigma=0.4)

        self.assertAlmostEqual(float(high_sigma[0]), float(low_sigma[0]) / 4.0, places=12)

    def test_config_wrapper_returns_pathwise_tensor(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "benchmark_entropic.yaml")
        paths = np.array(
            [
                [100.0, 101.0, 99.0],
                [100.0, 102.0, 103.0],
            ],
            dtype=np.float32,
        )

        positions = entropic_stat_arb_benchmark_from_config(config, paths)

        self.assertEqual(positions.shape, (2, 2))
        self.assertEqual(positions.dtype, np.float32)
        expected = entropic_stat_arb_benchmark(
            paths,
            mu=config.market.mu,
            theta=config.risk.theta,
            sigma=config.market.sigma,
        )
        self.assertTrue(np.allclose(positions, expected))

    def test_non_entropic_config_is_rejected(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "benchmark_cvar.yaml")
        paths = np.array([[100.0, 101.0]], dtype=np.float32)

        with self.assertRaises(ValueError):
            entropic_stat_arb_benchmark_from_config(config, paths)


if __name__ == "__main__":
    unittest.main()