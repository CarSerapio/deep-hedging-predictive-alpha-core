"""Feed-forward TensorFlow policy network for deep hedging.

This module belongs to the model-definition stage of the research pipeline. It
implements the paper-minimal feed-forward policy used in the Horikawa-style
benchmark: four hidden layers with 32 ReLU units that map hedge-time features
``[S_t, tau_t, delta_{t-1}]`` to the next hedge position.

The implementation keeps TensorFlow as an optional dependency at import time so
the broader NumPy-based benchmark modules remain usable in environments that do
not yet provide a TensorFlow-supported Python version.
"""

from __future__ import annotations

from typing import Any

from config import RuntimeConfig

try:
    import tensorflow as tf  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in current environment
    tf = None  # type: ignore[assignment]
    _TENSORFLOW_IMPORT_ERROR = exc
else:
    _TENSORFLOW_IMPORT_ERROR = None


def is_tensorflow_available() -> bool:
    """Return whether TensorFlow is importable in the active Python environment."""

    return tf is not None


def _require_tensorflow() -> Any:
    """Return the TensorFlow module or raise a clear environment error."""

    if tf is None:
        raise ImportError(
            "TensorFlow is required for the policy network. "
            "The current environment cannot import tensorflow; use a TensorFlow-supported "
            "Python version such as 3.10-3.12."
        ) from _TENSORFLOW_IMPORT_ERROR
    return tf


if tf is not None:

    class MLPPolicy(tf.keras.Model):
        """Feed-forward hedge policy mapping hedge-time features to next position.

        Args:
            input_dim: Number of hedge-time input features.
            hidden_layers: Number of hidden Dense layers.
            hidden_width: Width of each hidden Dense layer.
            activation: Hidden-layer activation function.
            output_dim: Output dimension. The baseline hedge policy uses one
                output corresponding to the next underlying position.
            seed: Optional random seed forwarded to Glorot initializers.
            name: Keras model name.
            dtype: TensorFlow compute dtype.
        """

        def __init__(
            self,
            *,
            input_dim: int,
            hidden_layers: int = 4,
            hidden_width: int = 32,
            activation: str = "relu",
            output_dim: int = 1,
            seed: int | None = None,
            name: str = "mlp_policy",
            dtype: str = "float32",
        ) -> None:
            super().__init__(name=name, dtype=dtype)

            if input_dim <= 0:
                raise ValueError("input_dim must be positive")
            if hidden_layers <= 0:
                raise ValueError("hidden_layers must be positive")
            if hidden_width <= 0:
                raise ValueError("hidden_width must be positive")
            if output_dim <= 0:
                raise ValueError("output_dim must be positive")

            self.input_dim = input_dim
            self.hidden_layers = hidden_layers
            self.hidden_width = hidden_width
            self.activation = activation
            self.output_dim = output_dim

            self.hidden_stack = [
                tf.keras.layers.Dense(
                    hidden_width,
                    activation=activation,
                    name=f"hidden_{layer_index + 1}",
                    dtype=dtype,
                )
                for layer_index in range(hidden_layers)
            ]
            self.output_layer = tf.keras.layers.Dense(
                output_dim,
                activation="linear",
                name="output",
                dtype=dtype,
            )

        def call(self, features: Any, training: bool = False) -> Any:
            """Map hedge-time features to the next hedge position.

            Args:
                features: Tensor-like object with shape ``[batch, input_dim]``.
                    In the baseline benchmark the columns are ``[S_t, tau_t,
                    delta_{t-1}]``.
                training: Standard Keras training flag.

            Returns:
                Tensor with shape ``[batch, output_dim]``.

            Raises:
                ValueError: If ``features`` is not rank-2 or has the wrong
                    feature dimension.
            """

            tensor = tf.convert_to_tensor(features, dtype=self.compute_dtype)
            if tensor.shape.rank != 2:
                raise ValueError("features must have shape [batch, input_dim]")
            if tensor.shape[-1] is not None and tensor.shape[-1] != self.input_dim:
                raise ValueError(f"features must have width {self.input_dim}")
            if tensor.shape[-1] is None:
                tf.debugging.assert_equal(
                    tf.shape(tensor)[-1],
                    self.input_dim,
                    message="features must match the configured input width",
                )

            hidden = tensor
            for layer in self.hidden_stack:
                hidden = layer(hidden, training=training)
            return self.output_layer(hidden, training=training)

else:

    class MLPPolicy:  # type: ignore[no-redef]
        """Placeholder class used when TensorFlow is unavailable."""


def build_mlp_policy(
    *,
    input_dim: int = 3,
    hidden_layers: int = 4,
    hidden_width: int = 32,
    activation: str = "relu",
    output_dim: int = 1,
    seed: int | None = None,
    name: str = "mlp_policy",
) -> MLPPolicy:
    """Build and initialize the baseline feed-forward policy network.

    Args:
        input_dim: Number of hedge-time features.
        hidden_layers: Number of hidden Dense layers.
        hidden_width: Width of each hidden layer.
        activation: Hidden-layer activation.
        output_dim: Policy output width.
        seed: Optional seed retained for API compatibility. Weight
            initialization now follows the default Keras Dense construction
            used by ``deephedging.DenseLayer`` and relies on the global TensorFlow
            seed for reproducibility.
        name: Keras model name.

    Returns:
        Initialized ``MLPPolicy`` instance.

    Raises:
        ImportError: If TensorFlow is unavailable.
        ValueError: If a structural parameter is invalid.
    """

    tf_module = _require_tensorflow()
    model = MLPPolicy(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        hidden_width=hidden_width,
        activation=activation,
        output_dim=output_dim,
        seed=seed,
        name=name,
        dtype="float32",
    )
    # Explicit initialization ensures the returned module already has weights,
    # which keeps later rollout and optimizer setup deterministic under a seed.
    model(tf_module.zeros((1, input_dim), dtype=tf_module.float32), training=False) # [[0.0, 0.0, 0.0]] is the feature vector for the first hedge time in the benchmark, so this is a valid input for weight initialization.
    return model # return neural network model instance itself, not the output of the model (optimal hedge)


def build_mlp_policy_from_config(
    config: RuntimeConfig,
    *,
    output_dim: int = 1,
    seed: int | None = None,
    name: str = "mlp_policy",
) -> MLPPolicy:
    """Build the baseline policy network from ``RuntimeConfig``.

    Args:
        config: Validated runtime configuration containing model hyperparameters
            and the hedge-time feature list.
        output_dim: Policy output width.
        seed: Optional weight-initialization seed.
        name: Keras model name.

    Returns:
        Initialized ``MLPPolicy`` instance.

    Raises:
        ImportError: If TensorFlow is unavailable.
    """

    return build_mlp_policy(
        input_dim=len(config.model.feature_names),
        hidden_layers=config.model.hidden_layers,
        hidden_width=config.model.hidden_width,
        activation=config.model.activation,
        output_dim=output_dim,
        seed=seed,
        name=name,
    )