"""Focused tests for the holdout-alpha helper logic."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import build_runtime_config_from_dict, load_config_dict  # noqa: E402
from evaluation.holdout_alpha import evaluate_holdout_alpha, evaluate_strategy_performance  # noqa: E402


class HoldoutAlphaTests(unittest.TestCase):
    """Check the pure-NumPy holdout evaluation logic."""

    def test_candidate_excess_vs_controls_matches_direct_pnl_difference(self) -> None:
        config = _build_test_config(proportional_rate=0.0)
        path_tensor = np.asarray(
            [
                [1.0, 1.1, 1.2],
                [1.0, 0.9, 1.0],
            ],
            dtype=np.float64,
        )
        candidate_hedge = np.asarray(
            [
                [0.5, 0.5],
                [0.5, 0.5],
            ],
            dtype=np.float64,
        )

        metrics = evaluate_holdout_alpha(config, path_tensor, candidate_hedge)

        self.assertAlmostEqual(metrics.candidate.to_summary_dict()["pnl_mean"], 0.05)
        self.assertAlmostEqual(metrics.to_summary_dict()["benchmark_adjusted"]["vs_passive"]["mean_excess_pnl"], 0.05)
        self.assertAlmostEqual(metrics.to_summary_dict()["benchmark_adjusted"]["vs_long_only"]["mean_excess_pnl"], -0.05)

    def test_transaction_cost_summary_reflects_first_trade_charge(self) -> None:
        config = _build_test_config(proportional_rate=0.01)
        path_tensor = np.asarray(
            [
                [1.0, 1.1, 1.2],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        long_only_hedge = np.ones((2, 2), dtype=np.float64)

        performance = evaluate_strategy_performance(config, path_tensor, long_only_hedge, label="long_only")

        self.assertAlmostEqual(performance.to_summary_dict()["mean_transaction_cost"], 0.01)

    def test_invalid_hedge_shape_is_rejected(self) -> None:
        config = _build_test_config(proportional_rate=0.0)
        path_tensor = np.asarray([[1.0, 1.1, 1.2]], dtype=np.float64)
        bad_hedge = np.asarray([0.5, 0.5], dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "hedge_tensor"):
            evaluate_holdout_alpha(config, path_tensor, bad_hedge)


def _build_test_config(*, proportional_rate: float):
    payload = copy.deepcopy(
        load_config_dict(PROJECT_ROOT / "configs" / "entropic_no_liability_unit_spot_cost_0p0025.yaml")
    )
    payload["costs"]["proportional_rate"] = proportional_rate
    payload["paths"]["train_paths"] = 4
    payload["paths"]["val_paths"] = 4
    payload["paths"]["test_paths"] = 4
    return build_runtime_config_from_dict(payload, source_path="holdout-alpha-test-config")


if __name__ == "__main__":
    unittest.main()