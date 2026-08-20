"""Validation tests for the MLP hedge policy.

The tests verify the TensorFlow import guard in unsupported environments and,
when TensorFlow is available, the shape and gradient conditions required by the
implementation procedure.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import load_config  # noqa: E402
from policies.mlp_policy import build_mlp_policy, build_mlp_policy_from_config, is_tensorflow_available  # noqa: E402

if is_tensorflow_available():
    import tensorflow as tf  # noqa: E402


class MLPPolicyTests(unittest.TestCase):
    """Checks the feed-forward hedge policy contract."""

    def test_build_raises_clear_error_without_tensorflow(self) -> None:
        if is_tensorflow_available():
            self.skipTest("TensorFlow is available; import-guard behavior is not active.")

        with self.assertRaisesRegex(ImportError, "TensorFlow is required for the policy network"):
            build_mlp_policy()

    def test_policy_output_shape_matches_batch_and_output_dim(self) -> None:
        if not is_tensorflow_available():
            self.skipTest("TensorFlow is unavailable in the current Python environment.")

        config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
        model = build_mlp_policy_from_config(config, seed=13)
        features = tf.constant(
            [
                [100.0, 0.08, 0.0],
                [101.5, 0.076, 0.2],
                [99.0, 0.072, -0.1],
            ],
            dtype=tf.float32,
        )

        outputs = model(features, training=False)

        self.assertEqual(tuple(outputs.shape), (3, 1))
        self.assertEqual(model.input_dim, 3)
        self.assertEqual(model.hidden_layers, 4)
        self.assertEqual(model.hidden_width, 32)

    def test_gradients_propagate_through_all_layers(self) -> None:
        if not is_tensorflow_available():
            self.skipTest("TensorFlow is unavailable in the current Python environment.")

        config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
        model = build_mlp_policy_from_config(config, seed=7)
        features = tf.constant(
            [
                [100.0, 0.08, 0.0],
                [102.0, 0.076, 0.25],
                [98.0, 0.072, -0.15],
                [101.0, 0.068, 0.1],
            ],
            dtype=tf.float32,
        )

        with tf.GradientTape() as tape:
            outputs = model(features, training=True)
            loss = tf.reduce_mean(tf.square(outputs))
        gradients = tape.gradient(loss, model.trainable_variables)

        self.assertEqual(len(gradients), len(model.trainable_variables))
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(bool(tf.reduce_all(tf.math.is_finite(gradient))) for gradient in gradients if gradient is not None))

    def test_layer_construction_matches_deephedging_dense_layer_defaults(self) -> None:
        if not is_tensorflow_available():
            self.skipTest("TensorFlow is unavailable in the current Python environment.")

        config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
        model = build_mlp_policy_from_config(config, seed=5)

        dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]

        self.assertEqual(len(model.hidden_stack), 4)
        self.assertEqual(len(dense_layers), 5)
        for layer in model.hidden_stack:
            self.assertEqual(layer.units, 32)
            self.assertTrue(layer.use_bias)
            self.assertEqual(layer.activation.__name__, "relu")
            self.assertEqual(layer.kernel_initializer.__class__.__name__, "GlorotUniform")
            self.assertEqual(layer.bias_initializer.__class__.__name__, "Zeros")
            self.assertEqual(layer.kernel_initializer.get_config(), tf.keras.initializers.GlorotUniform().get_config())
            self.assertEqual(layer.bias_initializer.get_config(), tf.keras.initializers.Zeros().get_config())

        self.assertEqual(model.output_layer.units, 1)
        self.assertTrue(model.output_layer.use_bias)
        self.assertEqual(model.output_layer.activation.__name__, "linear")
        self.assertEqual(model.output_layer.kernel_initializer.__class__.__name__, "GlorotUniform")
        self.assertEqual(model.output_layer.bias_initializer.__class__.__name__, "Zeros")
        self.assertEqual(model.output_layer.kernel_initializer.get_config(), tf.keras.initializers.GlorotUniform().get_config())
        self.assertEqual(model.output_layer.bias_initializer.get_config(), tf.keras.initializers.Zeros().get_config())


if __name__ == "__main__":
    unittest.main()