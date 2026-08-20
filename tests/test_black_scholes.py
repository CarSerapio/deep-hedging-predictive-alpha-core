"""Validation tests for the Black-Scholes benchmark hedge layer.

The benchmark tests check the discrete maturity-grid convention, closed-form
sanity conditions for price and delta, and the pathwise shape contract used by
later PnL and training code.
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

from baselines.black_scholes import (  # noqa: E402
    black_scholes_call_delta,
    black_scholes_call_price,
    black_scholes_price_and_delta_from_config,
    build_time_to_maturity_grid,
)
from config import load_config  # noqa: E402
from simulators.gbm import simulate_gbm_paths_from_config  # noqa: E402


class BlackScholesTests(unittest.TestCase):
    """Checks that the analytic benchmark hedge is aligned with the benchmark grid."""

    def test_time_to_maturity_grid_matches_benchmark_convention(self) -> None:
        tau = build_time_to_maturity_grid(0.08, 20)

        self.assertEqual(tau.shape, (20,))
        self.assertAlmostEqual(float(tau[0]), 0.08, places=7)
        self.assertAlmostEqual(float(tau[-1]), 0.004, places=7)

    def test_atm_delta_is_near_one_half_under_zero_rates(self) -> None:
        delta = black_scholes_call_delta(
            100.0,
            strike=100.0,
            time_to_maturity=0.08,
            sigma=0.20,
        )

        self.assertGreater(float(delta), 0.50)
        self.assertLess(float(delta), 0.53)

    def test_price_and_delta_are_monotone_in_spot(self) -> None:
        spots = np.array([80.0, 100.0, 120.0], dtype=np.float64)
        prices = black_scholes_call_price(spots, strike=100.0, time_to_maturity=0.08, sigma=0.20)
        deltas = black_scholes_call_delta(spots, strike=100.0, time_to_maturity=0.08, sigma=0.20)

        self.assertTrue(np.all(np.diff(prices) > 0.0))
        self.assertTrue(np.all(np.diff(deltas) > 0.0))

    def test_terminal_delta_and_intrinsic_value_are_handled_cleanly(self) -> None:
        spots = np.array([90.0, 100.0, 110.0], dtype=np.float64)
        prices = black_scholes_call_price(spots, strike=100.0, time_to_maturity=0.0, sigma=0.20)
        deltas = black_scholes_call_delta(spots, strike=100.0, time_to_maturity=0.0, sigma=0.20)

        self.assertTrue(np.array_equal(prices, np.array([0.0, 0.0, 10.0], dtype=np.float64)))
        self.assertTrue(np.array_equal(deltas, np.array([0.0, 0.5, 1.0], dtype=np.float64)))

    def test_config_wrapper_returns_pathwise_price_and_delta_tensors(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
        paths = simulate_gbm_paths_from_config(config, n_paths=64, seed=5)

        prices, deltas = black_scholes_price_and_delta_from_config(config, paths)

        self.assertEqual(prices.shape, (64, config.market.n_steps))
        self.assertEqual(deltas.shape, (64, config.market.n_steps))
        self.assertEqual(prices.dtype, np.float32)
        self.assertEqual(deltas.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(prices)))
        self.assertTrue(np.all(np.isfinite(deltas)))
        self.assertTrue(np.all((0.0 <= deltas) & (deltas <= 1.0)))


if __name__ == "__main__":
    unittest.main()