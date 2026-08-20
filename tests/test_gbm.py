"""Validation tests for the GBM market simulator.

The simulator tests verify the path tensor contract, positivity of prices under
GBM, deterministic replay under a fixed seed, and agreement between empirical
log-return moments and the exact model moments used by the benchmark.
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

from config import load_config  # noqa: E402
from simulators.gbm import simulate_gbm_paths, simulate_gbm_paths_from_config  # noqa: E402


class GBMSimulatorTests(unittest.TestCase):
    """Checks that the simulated market environment matches its stated model."""

    def test_shape_and_positive_prices(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
        paths = simulate_gbm_paths_from_config(config, split="validation", n_paths=128, seed=11)

        self.assertEqual(paths.shape, (128, config.market.n_steps + 1))
        self.assertEqual(paths.dtype, np.float32)
        self.assertTrue(np.all(paths > 0.0))
        self.assertTrue(np.allclose(paths[:, 0], config.market.s0)) # Check that the initial prices in the simulated paths match the specified initial price in the configuration.

    def test_seed_reproducibility(self) -> None:
        first = simulate_gbm_paths(
            s0=100.0,
            mu=0.05,
            sigma=0.20,
            maturity=0.08,
            n_steps=20,
            n_paths=64,
            seed=123,
        )
        second = simulate_gbm_paths(
            s0=100.0,
            mu=0.05,
            sigma=0.20,
            maturity=0.08,
            n_steps=20,
            n_paths=64,
            seed=123,
        )

        self.assertTrue(np.array_equal(first, second))

    def test_log_return_moments_match_theory(self) -> None:
        mu = 0.05
        sigma = 0.20
        maturity = 0.08
        n_steps = 20
        dt = maturity / n_steps

        paths = simulate_gbm_paths(
            s0=100.0,
            mu=mu,
            sigma=sigma,
            maturity=maturity,
            n_steps=n_steps,
            n_paths=50000,
            seed=7,
            dtype=np.float64,
        )
        log_returns = np.log(paths[:, 1:] / paths[:, :-1]).reshape(-1)

        expected_mean = (mu - 0.5 * sigma * sigma) * dt
        expected_var = sigma * sigma * dt

        self.assertAlmostEqual(float(log_returns.mean()), expected_mean, delta=5e-5)
        self.assertAlmostEqual(float(log_returns.var()), expected_var, delta=5e-5)


if __name__ == "__main__":
    unittest.main()