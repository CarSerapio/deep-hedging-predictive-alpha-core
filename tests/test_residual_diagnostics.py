"""Focused tests for the residual diagnostics helpers."""

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
from evaluation import evaluate_residual_diagnostics  # noqa: E402
from simulators.gbm import simulate_gbm_paths_from_config  # noqa: E402


class ResidualDiagnosticsTests(unittest.TestCase):
    """Check the helper logic used by the residual diagnostics report."""

    def test_zero_residual_produces_zero_summary_metrics(self) -> None:
        config = _build_runtime_config(n_steps=4)
        paths = simulate_gbm_paths_from_config(config, n_paths=32, seed=17)
        benchmark_delta = np.full((32, 4), 0.25, dtype=np.float64)

        metrics = evaluate_residual_diagnostics(
            config,
            paths,
            benchmark_delta,
            benchmark_delta,
        )
        summary = metrics.to_summary_dict()

        self.assertAlmostEqual(summary["mean_residual"], 0.0, places=12)
        self.assertAlmostEqual(summary["mean_abs_residual"], 0.0, places=12)
        self.assertAlmostEqual(summary["rmse_residual"], 0.0, places=12)
        self.assertAlmostEqual(summary["max_abs_residual"], 0.0, places=12)
        np.testing.assert_allclose(summary["mean_abs_residual_profile"], 0.0, atol=1e-12)

    def test_moneyness_buckets_capture_structured_residual_levels(self) -> None:
        config = _build_runtime_config(n_steps=2)
        paths = np.array(
            [
                [0.90, 0.90, 0.90],
                [1.00, 1.00, 1.00],
                [1.10, 1.10, 1.10],
            ],
            dtype=np.float64,
        )
        hedge_component = np.array(
            [
                [0.10, 0.10],
                [0.20, 0.20],
                [0.30, 0.30],
            ],
            dtype=np.float64,
        )
        benchmark_delta = np.zeros_like(hedge_component)

        metrics = evaluate_residual_diagnostics(
            config,
            paths,
            hedge_component,
            benchmark_delta,
        )
        summary = metrics.to_summary_dict()

        self.assertAlmostEqual(summary["moneyness_buckets"]["lt_0p95"]["mean_abs_residual"], 0.10, places=12)
        self.assertAlmostEqual(summary["moneyness_buckets"]["between_0p95_and_1p05"]["mean_abs_residual"], 0.20, places=12)
        self.assertAlmostEqual(summary["moneyness_buckets"]["gt_1p05"]["mean_abs_residual"], 0.30, places=12)

    def test_path_shape_mismatch_is_rejected(self) -> None:
        config = _build_runtime_config(n_steps=3)
        paths = np.ones((5, 3), dtype=np.float64)
        hedge_component = np.zeros((5, 3), dtype=np.float64)
        benchmark_delta = np.zeros((5, 3), dtype=np.float64)

        with self.assertRaises(ValueError):
            evaluate_residual_diagnostics(
                config,
                paths,
                hedge_component,
                benchmark_delta,
            )


def _build_runtime_config(*, n_steps: int):
    payload = copy.deepcopy(load_config_dict(PROJECT_ROOT / "configs" / "entropic_with_liability_unit_spot_frictionless.yaml"))
    payload["market"]["n_steps"] = n_steps
    payload["market"]["dt"] = payload["market"]["maturity"] / n_steps
    payload["paths"]["test_paths"] = 32
    payload["paths"]["seed"] = 11
    return build_runtime_config_from_dict(payload, source_path=f"residual-diagnostics-test-{n_steps}")


if __name__ == "__main__":
    unittest.main()