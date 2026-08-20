"""Validation tests for the benchmark decomposition metrics."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baselines.analytic_deep_hedging import analytic_deep_hedging_benchmark_from_config  # noqa: E402
from baselines.black_scholes import black_scholes_price_and_delta_from_config  # noqa: E402
from config import load_config  # noqa: E402
from evaluation import compute_pathwise_normalized_gap, evaluate_hedge_tensor_decomposition  # noqa: E402
from simulators.gbm import simulate_gbm_paths_from_config  # noqa: E402


class BenchmarkDecompositionTests(unittest.TestCase):
    """Checks the raw and adjusted benchmark-gap identities."""

    def test_analytic_entropic_benchmark_has_zero_adjusted_gap(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "benchmark_entropic_unit_spot.yaml")
        paths = simulate_gbm_paths_from_config(config, n_paths=48, seed=23)

        analytic_hedge = analytic_deep_hedging_benchmark_from_config(config, paths)
        metrics = evaluate_hedge_tensor_decomposition(config, paths, analytic_hedge)

        self.assertIsNotNone(metrics.adjusted_gap)
        np.testing.assert_allclose(metrics.adjusted_gap, 0.0, atol=1e-8)
        self.assertGreater(float(np.mean(metrics.raw_gap)), 0.0)

    def test_black_scholes_hedge_has_zero_raw_gap(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "benchmark_entropic_unit_spot.yaml")
        paths = simulate_gbm_paths_from_config(config, n_paths=48, seed=29)

        _, black_scholes_delta = black_scholes_price_and_delta_from_config(config, paths)
        metrics = evaluate_hedge_tensor_decomposition(config, paths, black_scholes_delta)

        np.testing.assert_allclose(metrics.raw_gap, 0.0, atol=1e-8)
        self.assertIsNotNone(metrics.adjusted_gap)
        self.assertGreater(float(np.mean(metrics.adjusted_gap)), 0.0)

    def test_cvar_metrics_omit_entropic_adjustment(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "benchmark_cvar_unit_spot.yaml")
        paths = simulate_gbm_paths_from_config(config, n_paths=32, seed=31)
        _, black_scholes_delta = black_scholes_price_and_delta_from_config(config, paths)

        metrics = evaluate_hedge_tensor_decomposition(config, paths, black_scholes_delta)

        np.testing.assert_allclose(metrics.raw_gap, 0.0, atol=1e-8)
        self.assertIsNone(metrics.adjusted_gap)

    def test_shape_mismatch_is_rejected(self) -> None:
        hedge_tensor = np.zeros((4, 3), dtype=np.float32)
        benchmark_tensor = np.zeros((4, 4), dtype=np.float32)

        with self.assertRaises(ValueError):
            compute_pathwise_normalized_gap(hedge_tensor, benchmark_tensor)


if __name__ == "__main__":
    unittest.main()