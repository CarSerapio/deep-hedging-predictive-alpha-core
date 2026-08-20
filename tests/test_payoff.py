"""Validation tests for the European call payoff engine.

These tests confirm the liability definition used in the benchmark pipeline,
including pathwise payoff arithmetic, config-driven strike lookup, and basic
input validation.
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
from payoffs.european_call import compute_payoff, european_call_payoff, european_call_payoff_from_config  # noqa: E402


class EuropeanCallPayoffTests(unittest.TestCase):
    """Checks that the payoff layer implements the intended terminal liability."""

    def test_toy_paths_produce_expected_payoffs(self) -> None:
        paths = np.array(
            [
                [100.0, 98.0, 97.0],
                [100.0, 102.0, 110.0],
                [100.0, 99.0, 100.0],
            ],
            dtype=np.float32,
        )

        payoff = european_call_payoff(paths, strike=100.0)

        expected = np.array([0.0, 10.0, 0.0], dtype=np.float32)
        self.assertTrue(np.array_equal(payoff, expected))
        self.assertEqual(payoff.dtype, np.float32)

    def test_from_config_uses_market_strike(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
        paths = np.array(
            [
                [100.0, 100.5, 101.0],
                [100.0, 99.5, 95.0],
            ],
            dtype=np.float32,
        )

        payoff = european_call_payoff_from_config(config, paths)

        expected = np.array([1.0, 0.0], dtype=np.float32)
        self.assertTrue(np.array_equal(payoff, expected))

    def test_compute_payoff_rejects_unknown_product_type(self) -> None:
        paths = np.array([[100.0, 101.0]], dtype=np.float32)

        with self.assertRaises(ValueError):
            compute_payoff(paths, strike=100.0, product_type="put")

    def test_payoff_requires_two_dimensional_path_tensor(self) -> None:
        with self.assertRaises(ValueError):
            european_call_payoff(np.array([100.0, 105.0], dtype=np.float32), strike=100.0)


if __name__ == "__main__":
    unittest.main()