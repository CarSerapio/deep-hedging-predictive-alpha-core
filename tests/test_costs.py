"""Validation tests for the proportional transaction-cost engine.

These tests cover pathwise proportional spot trading costs computed from
absolute trade sizes.
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

from config import build_runtime_config_from_dict, load_config_dict  # noqa: E402
from finance.costs import (  # noqa: E402
    compute_proportional_transaction_cost,
    compute_proportional_transaction_cost_from_config,
)


class TransactionCostTests(unittest.TestCase):
    """Checks proportional transaction-cost accounting on toy hedge paths."""

    def test_zero_turnover_has_zero_cost(self) -> None:
        paths = np.array(
            [
                [100.0, 101.0, 102.0],
                [100.0, 99.0, 98.0],
            ],
            dtype=np.float32,
        )
        hedges = np.zeros((2, 2), dtype=np.float32)

        costs = compute_proportional_transaction_cost(paths, hedges, proportional_rate=0.01)

        self.assertTrue(np.array_equal(costs, np.zeros(2, dtype=np.float32)))

    def test_opening_and_rebalancing_costs_match_hand_computation(self) -> None:
        paths = np.array([[100.0, 105.0, 110.0]], dtype=np.float32)
        hedges = np.array([[1.0, 0.5]], dtype=np.float32)

        costs = compute_proportional_transaction_cost(paths, hedges, proportional_rate=0.01)

        expected = np.array([1.525], dtype=np.float32)
        self.assertTrue(np.allclose(costs, expected, atol=1e-6))

    def test_config_wrapper_uses_cost_rate(self) -> None:
        payload = load_config_dict(PROJECT_ROOT / "configs" / "base.yaml")
        payload["costs"]["proportional_rate"] = 0.002
        config = build_runtime_config_from_dict(payload, source_path="cost-wrapper")

        paths = np.array([[100.0, 110.0]], dtype=np.float32)
        hedges = np.array([[0.75]], dtype=np.float32)

        costs = compute_proportional_transaction_cost_from_config(config, paths, hedges)

        self.assertTrue(np.allclose(costs, np.array([0.15], dtype=np.float32), atol=1e-6))


if __name__ == "__main__":
    unittest.main()