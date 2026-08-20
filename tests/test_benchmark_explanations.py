"""Focused tests for the benchmark-explanation helpers."""

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
from evaluation.benchmark_explanations import (  # noqa: E402
    build_volatility_scaled_long_only_hedge,
    deduplicate_factor_vectors,
    evaluate_benchmark_explanations,
    fit_linear_exposure_regression,
)


class BenchmarkExplanationTests(unittest.TestCase):
    """Check the pure-NumPy benchmark-explanation logic."""

    def test_volatility_scaled_long_only_uses_inverse_sigma_ratio(self) -> None:
        hedge = build_volatility_scaled_long_only_hedge(
            n_paths=2,
            n_steps=3,
            reference_sigma=0.2,
            current_sigma=0.4,
        )

        np.testing.assert_allclose(hedge, 0.5 * np.ones((2, 3), dtype=np.float64))

    def test_linear_exposure_regression_recovers_known_coefficients(self) -> None:
        factor = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        target = 1.5 + 2.0 * factor

        summary = fit_linear_exposure_regression(target, {"factor": factor})

        self.assertAlmostEqual(summary["intercept"], 1.5)
        self.assertAlmostEqual(summary["coefficients"]["factor"], 2.0)
        self.assertAlmostEqual(summary["r2"], 1.0)

    def test_signal_rule_limitation_and_alias_detection_are_reported(self) -> None:
        config = _build_test_config(mu=0.05, sigma=0.2)
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

        metrics = evaluate_benchmark_explanations(
            config,
            path_tensor,
            candidate_hedge,
            reference_sigma=0.2,
        )
        summary = metrics.to_summary_dict()

        self.assertFalse(summary["limitations"]["simple_signal_rule_available"])
        self.assertTrue(summary["limitations"]["buy_and_hold_equals_constant_long_only"])

        distinct, aliases = deduplicate_factor_vectors(
            {
                "buy_and_hold": np.asarray([1.0, 2.0], dtype=np.float64),
                "constant_long_only": np.asarray([3.0, 5.0], dtype=np.float64),
            }
        )
        self.assertEqual(list(distinct.keys()), ["buy_and_hold"])
        self.assertEqual(aliases, {"constant_long_only": "buy_and_hold"})

    def test_signal_rule_is_marked_available_for_predictive_feature_state(self) -> None:
        config = _build_predictive_test_config(mu=0.05, sigma=0.2)
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

        metrics = evaluate_benchmark_explanations(
            config,
            path_tensor,
            candidate_hedge,
            reference_sigma=0.2,
        )

        self.assertTrue(metrics.to_summary_dict()["limitations"]["simple_signal_rule_available"])


def _build_test_config(*, mu: float, sigma: float):
    payload = copy.deepcopy(
        load_config_dict(PROJECT_ROOT / "configs" / "entropic_no_liability_unit_spot_cost_0p0025.yaml")
    )
    payload["market"]["mu"] = mu
    payload["market"]["sigma"] = sigma
    payload["paths"]["train_paths"] = 4
    payload["paths"]["val_paths"] = 4
    payload["paths"]["test_paths"] = 4
    return build_runtime_config_from_dict(payload, source_path="benchmark-explanations-test-config")


def _build_predictive_test_config(*, mu: float, sigma: float):
    payload = copy.deepcopy(
        load_config_dict(PROJECT_ROOT / "configs" / "predictive_signal_no_liability_unit_spot_cost_0p0025.yaml")
    )
    payload["market"]["mu"] = mu
    payload["market"]["sigma"] = sigma
    payload["paths"]["train_paths"] = 4
    payload["paths"]["val_paths"] = 4
    payload["paths"]["test_paths"] = 4
    return build_runtime_config_from_dict(payload, source_path="predictive-benchmark-explanations-test-config")


if __name__ == "__main__":
    unittest.main()