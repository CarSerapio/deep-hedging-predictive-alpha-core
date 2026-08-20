"""Focused tests for the predictive-signal extension track."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import build_runtime_config_from_dict, load_config, load_config_dict  # noqa: E402
from policies import build_mlp_policy_from_config, is_tensorflow_available, rollout_policy_from_config  # noqa: E402
from simulators import simulate_market_data_from_config  # noqa: E402
from training import train_entropic_deep_hedger  # noqa: E402

if is_tensorflow_available():
    import tensorflow as tf  # noqa: E402


class PredictiveSignalExtensionTests(unittest.TestCase):
    """Check that the predictive-signal extension stays isolated and usable."""

    def test_predictive_signal_config_loads_with_four_feature_state(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "predictive_signal_no_liability_unit_spot_cost_0p0025.yaml")

        self.assertTrue(config.signal.enabled)
        self.assertEqual(
            config.model.feature_names,
            ["spot", "time_to_maturity", "previous_hedge", "predictive_signal"],
        )

    def test_signal_enabled_simulator_returns_signal_tensor(self) -> None:
        config = _small_predictive_signal_config(signal_initial_value=1.0, signal_phi=0.0, innovation_scale=0.0)

        market_data = simulate_market_data_from_config(config, split="test", seed=17)

        self.assertEqual(market_data.path_tensor.shape, (config.paths.test_paths, config.market.n_steps + 1))
        self.assertIsNotNone(market_data.predictive_signal_tensor)
        assert market_data.predictive_signal_tensor is not None
        self.assertEqual(market_data.predictive_signal_tensor.shape, (config.paths.test_paths, config.market.n_steps))
        self.assertTrue(np.allclose(market_data.predictive_signal_tensor[:, 0], 1.0))

    def test_rollout_can_use_predictive_signal_column(self) -> None:
        if not is_tensorflow_available():
            self.skipTest("TensorFlow is unavailable in the current Python environment.")

        class SignalOnlyPolicy(tf.keras.Model):
            def call(self, features, training: bool = False):  # type: ignore[override]
                del training
                return features[:, 3:4]

        config = _small_predictive_signal_config()
        paths = np.asarray(
            [
                [1.0, 1.1, 1.2],
                [1.0, 0.9, 1.0],
            ],
            dtype=np.float32,
        )
        predictive_signal = np.asarray(
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ],
            dtype=np.float32,
        )

        hedges = rollout_policy_from_config(
            config,
            paths,
            SignalOnlyPolicy(),
            predictive_signal_tensor=predictive_signal,
        ).numpy()

        np.testing.assert_allclose(hedges, predictive_signal, atol=1e-7)

    def test_entropic_training_replays_saved_predictive_signal_hedges(self) -> None:
        if not is_tensorflow_available():
            self.skipTest("TensorFlow is unavailable in the current Python environment.")

        config = _small_predictive_signal_config()
        with TemporaryDirectory() as tmpdir:
            result = train_entropic_deep_hedger(config, tmpdir, deterministic=True)

            restored_policy = build_mlp_policy_from_config(config, seed=config.paths.seed)
            restored_policy.load_weights(result.checkpoint_path)
            test_market = simulate_market_data_from_config(config, split="test", seed=config.paths.seed + 2)
            restored_hedges = rollout_policy_from_config(
                config,
                result.test_paths,
                restored_policy,
                predictive_signal_tensor=test_market.predictive_signal_tensor,
            ).numpy()
            np.testing.assert_allclose(restored_hedges, result.test_hedges, atol=1e-5)


def _small_predictive_signal_config(*, signal_initial_value: float = 0.25, signal_phi: float = 0.5, innovation_scale: float = 0.1):
    payload = load_config_dict(PROJECT_ROOT / "configs" / "predictive_signal_no_liability_unit_spot_cost_0p0025.yaml")
    payload["market"]["mu"] = 0.02
    payload["market"]["sigma"] = 0.15
    payload["paths"]["train_paths"] = 96
    payload["paths"]["val_paths"] = 48
    payload["paths"]["test_paths"] = 48
    payload["paths"]["seed"] = 37
    payload["training"]["batch_size"] = 24
    payload["training"]["epochs"] = 2
    payload["training"]["learning_rate"] = 0.005
    payload["training"]["patience"] = 1
    payload["signal"]["initial_value"] = signal_initial_value
    payload["signal"]["ar1_phi"] = signal_phi
    payload["signal"]["innovation_scale"] = innovation_scale
    return build_runtime_config_from_dict(payload, source_path="predictive-signal-test")


if __name__ == "__main__":
    unittest.main()