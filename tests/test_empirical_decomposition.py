"""Focused tests for the empirical decomposition helpers."""

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

from baselines.black_scholes import black_scholes_price_and_delta_from_config  # noqa: E402
from config import build_runtime_config_from_dict, load_config_dict  # noqa: E402
from evaluation import evaluate_empirical_decomposition  # noqa: E402
from simulators.gbm import simulate_gbm_paths_from_config  # noqa: E402


class EmpiricalDecompositionTests(unittest.TestCase):
    """Check that the decomposition computes eta^Z on matched policy pairs."""

    def test_black_scholes_delta_is_recovered_when_with_liability_adds_only_delta(self) -> None:
        no_liability_config, with_liability_config = _build_matched_entropic_pair()
        paths = simulate_gbm_paths_from_config(no_liability_config, n_paths=48, seed=23)
        _, black_scholes_delta = black_scholes_price_and_delta_from_config(with_liability_config, paths)

        no_liability_hedge = np.zeros_like(black_scholes_delta)
        with_liability_hedge = np.asarray(black_scholes_delta, dtype=np.float64)

        metrics = evaluate_empirical_decomposition(
            no_liability_config,
            with_liability_config,
            paths,
            no_liability_hedge,
            with_liability_hedge,
        )

        np.testing.assert_allclose(metrics.hedge_component, black_scholes_delta, atol=1e-8)
        np.testing.assert_allclose(metrics.benchmark_gap, 0.0, atol=1e-8)
        self.assertAlmostEqual(metrics.reconstructed_with_liability_max_abs_error, 0.0, places=12)

    def test_shape_mismatch_is_rejected(self) -> None:
        no_liability_config, with_liability_config = _build_matched_entropic_pair()
        paths = simulate_gbm_paths_from_config(no_liability_config, n_paths=16, seed=31)
        _, black_scholes_delta = black_scholes_price_and_delta_from_config(with_liability_config, paths)

        no_liability_hedge = np.zeros_like(black_scholes_delta)
        with self.assertRaises(ValueError):
            evaluate_empirical_decomposition(
                no_liability_config,
                with_liability_config,
                paths,
                no_liability_hedge[:, :-1],
                black_scholes_delta,
            )

    def test_mismatched_cost_regime_is_rejected(self) -> None:
        no_liability_config, with_liability_config = _build_matched_entropic_pair()
        mismatched_payload = load_config_dict(PROJECT_ROOT / "configs" / "entropic_with_liability_unit_spot.yaml")
        mismatched_payload["costs"]["proportional_rate"] = 0.02
        mismatched_with_liability_config = build_runtime_config_from_dict(mismatched_payload, source_path="with-liability-mismatch")
        paths = simulate_gbm_paths_from_config(no_liability_config, n_paths=16, seed=37)
        _, black_scholes_delta = black_scholes_price_and_delta_from_config(with_liability_config, paths)

        with self.assertRaises(ValueError):
            evaluate_empirical_decomposition(
                no_liability_config,
                mismatched_with_liability_config,
                paths,
                np.zeros_like(black_scholes_delta),
                black_scholes_delta,
            )


def _build_matched_entropic_pair():
    no_liability_payload = load_config_dict(PROJECT_ROOT / "configs" / "entropic_no_liability_unit_spot.yaml")
    with_liability_payload = load_config_dict(PROJECT_ROOT / "configs" / "entropic_with_liability_unit_spot.yaml")
    no_liability_payload = copy.deepcopy(no_liability_payload)
    with_liability_payload = copy.deepcopy(with_liability_payload)
    no_liability_payload["paths"]["test_paths"] = 48
    with_liability_payload["paths"]["test_paths"] = 48
    no_liability_payload["paths"]["seed"] = 19
    with_liability_payload["paths"]["seed"] = 19
    no_liability_payload["risk"]["theta"] = 1.0
    with_liability_payload["risk"]["theta"] = 1.0
    return (
        build_runtime_config_from_dict(no_liability_payload, source_path="decomposition-no-liability"),
        build_runtime_config_from_dict(with_liability_payload, source_path="decomposition-with-liability"),
    )


if __name__ == "__main__":
    unittest.main()