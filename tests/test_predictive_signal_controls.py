"""Focused tests for predictive-track signal-destruction controls."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import load_config, load_config_dict  # noqa: E402
from evaluation.anti_spurious_controls import build_shuffled_signal_tensor, summarize_hedge_response  # noqa: E402
from policies import build_mlp_policy_from_config, is_tensorflow_available, rollout_policy_from_config  # noqa: E402
from simulators import simulate_market_data_from_config  # noqa: E402
from workflows.predictive_signal_controls import build_predictive_signal_controls_report  # noqa: E402


class PredictiveSignalControlsTests(unittest.TestCase):
    """Check the helper and report logic used by predictive signal controls."""

    def test_build_shuffled_signal_tensor_preserves_per_step_values(self) -> None:
        signal = np.asarray(
            [
                [1.0, 4.0],
                [2.0, 5.0],
                [3.0, 6.0],
            ],
            dtype=np.float64,
        )

        shuffled = build_shuffled_signal_tensor(signal, seed=17)

        self.assertEqual(shuffled.shape, signal.shape)
        np.testing.assert_allclose(np.sort(shuffled[:, 0]), np.sort(signal[:, 0]))
        np.testing.assert_allclose(np.sort(shuffled[:, 1]), np.sort(signal[:, 1]))

    def test_summarize_hedge_response_reports_shift_size(self) -> None:
        baseline = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        comparison = np.asarray([[1.0, 1.0], [1.0, 3.0]], dtype=np.float64)

        summary = summarize_hedge_response(baseline, comparison)

        self.assertAlmostEqual(summary["mean_abs_hedge_shift"], 1.0)
        self.assertAlmostEqual(summary["max_abs_hedge_shift"], 2.0)

    def test_predictive_control_report_replays_saved_policy_and_detects_signal_response(self) -> None:
        if not is_tensorflow_available():
            self.skipTest("TensorFlow is unavailable in the current Python environment.")

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            holdout_evaluation_dir = root / "holdout_evaluation"
            holdout_evaluation_dir.mkdir(parents=True, exist_ok=True)
            policy_run_dir = root / "policy_run"
            policy_run_dir.mkdir(parents=True, exist_ok=True)
            policy_config_path = root / "policy_config.yaml"
            holdout_config_path = root / "holdout_config.yaml"
            holdout_evaluation_summary_path = holdout_evaluation_dir / "summary.json"

            policy_payload = load_config_dict(
                PROJECT_ROOT / "configs" / "predictive_signal_no_liability_unit_spot_cost_0p0025.yaml"
            )
            policy_payload["paths"]["train_paths"] = 8
            policy_payload["paths"]["val_paths"] = 8
            policy_payload["paths"]["test_paths"] = 8
            policy_payload["paths"]["seed"] = 901
            policy_config_path.write_text(json.dumps(policy_payload, indent=2), encoding="utf-8")

            holdout_payload = json.loads(json.dumps(policy_payload))
            holdout_payload["experiment"]["name"] = "predictive_holdout_test"
            holdout_payload["experiment"]["regime_label"] = "predictive_holdout_test"
            holdout_payload["market"]["mu"] = 0.08
            holdout_payload["paths"]["seed"] = 902
            holdout_config_path.write_text(json.dumps(holdout_payload, indent=2), encoding="utf-8")

            policy_config = load_config(policy_config_path)
            holdout_config = load_config(holdout_config_path)
            policy = build_mlp_policy_from_config(policy_config, seed=policy_config.paths.seed)
            policy.save_weights(policy_run_dir / "policy.weights.h5")

            market_data = simulate_market_data_from_config(
                holdout_config,
                split="test",
                seed=holdout_config.paths.seed + 2,
            )
            assert market_data.predictive_signal_tensor is not None
            saved_candidate_hedge = np.asarray(
                rollout_policy_from_config(
                    holdout_config,
                    market_data.path_tensor,
                    policy,
                    predictive_signal_tensor=market_data.predictive_signal_tensor,
                    training=False,
                ),
                dtype=np.float64,
            )
            np.savez(holdout_evaluation_dir / "predictive_holdout_test_arrays.npz", candidate_hedge=saved_candidate_hedge)

            holdout_evaluation_summary_path.write_text(
                json.dumps(
                    {
                        "verification_passed": True,
                        "policy_config": str(policy_config_path.resolve()),
                        "policy_run_dir": str(policy_run_dir.resolve()),
                        "holdout_regimes": [
                            {
                                "regime_label": "predictive_holdout_test",
                                "config_path": str(holdout_config_path.resolve()),
                                "changed_dimensions": ["drift", "seed"],
                            }
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            report = build_predictive_signal_controls_report(
                holdout_evaluation_summary_path=holdout_evaluation_summary_path,
                policy_run_dir=policy_run_dir,
            )

            self.assertTrue(report["verification"]["baseline_replay_matches_saved_holdout_hedges"])
            self.assertEqual(report["aggregate"]["n_holdout_regimes"], 1)
            self.assertIn("zero_predictive_signal", report["holdout_regimes"][0])


if __name__ == "__main__":
    unittest.main()