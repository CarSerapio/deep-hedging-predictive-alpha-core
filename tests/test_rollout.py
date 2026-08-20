"""Validation tests for the sequential hedge rollout engine.

These tests verify the rollout execution contract: features are built from
current hedge-time information, the previous hedge is fed back correctly, and a
zero-initialized policy produces deterministic zero rollouts.
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
from policies import (  # noqa: E402
    build_hedge_features,
    build_mlp_policy_from_config,
    is_tensorflow_available,
    rollout_policy,
    rollout_policy_from_config,
)
from simulators.gbm import simulate_gbm_paths_from_config  # noqa: E402

if is_tensorflow_available():
    import tensorflow as tf  # noqa: E402


class RolloutTests(unittest.TestCase):
    """Checks the current sequential hedge rollout on benchmark paths."""

    def test_build_hedge_features_concatenates_spot_tau_and_previous_hedge(self) -> None:
        if not is_tensorflow_available():
            self.skipTest("TensorFlow is unavailable in the current Python environment.")

        features = build_hedge_features(
            [100.0, 101.0],
            time_to_maturity=0.08,
            previous_hedge=[0.0, 0.25],
        )

        expected = tf.constant(
            [
                [100.0, 0.08, 0.0],
                [101.0, 0.08, 0.25],
            ],
            dtype=tf.float32,
        )
        self.assertTrue(bool(tf.reduce_all(tf.abs(features - expected) < 1e-7)))

    def test_zero_initialized_policy_rollout_is_zero_and_deterministic(self) -> None:
        if not is_tensorflow_available():
            self.skipTest("TensorFlow is unavailable in the current Python environment.")

        config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
        policy = build_mlp_policy_from_config(config, seed=11)
        for variable in policy.trainable_variables:
            variable.assign(tf.zeros_like(variable))

        paths = simulate_gbm_paths_from_config(config, n_paths=8, seed=5)
        first = rollout_policy_from_config(config, paths, policy)
        second = rollout_policy_from_config(config, paths, policy)

        self.assertEqual(tuple(first.shape), (8, config.market.n_steps))
        self.assertTrue(bool(tf.reduce_all(first == 0.0)))
        self.assertTrue(bool(tf.reduce_all(first == second)))

    def test_rollout_uses_current_spot_and_previous_hedge_without_off_by_one(self) -> None:
        if not is_tensorflow_available():
            self.skipTest("TensorFlow is unavailable in the current Python environment.")

        class DeterministicPolicy(tf.keras.Model):
            def call(self, features, training: bool = False):  # type: ignore[override]
                del training
                spot = features[:, 0:1]
                tau = features[:, 1:2]
                previous = features[:, 2:3]
                return 0.001 * spot + 0.5 * tau + previous

        paths = np.array(
            [
                [100.0, 101.0, 102.0],
                [200.0, 210.0, 220.0],
            ],
            dtype=np.float32,
        )

        hedges = rollout_policy(paths, DeterministicPolicy(), maturity=0.08)

        expected = tf.constant(
            [
                [0.14, 0.261],
                [0.24, 0.47],
            ],
            dtype=tf.float32,
        )
        self.assertTrue(bool(tf.reduce_all(tf.abs(hedges - expected) < 1e-6)))


if __name__ == "__main__":
    unittest.main()