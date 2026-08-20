"""Sequential rollout of hedge policies on simulated price paths.

This module belongs to the strategy-execution stage of the research pipeline.
It unrolls a hedge policy through time on a batch of price paths by repeatedly
building the baseline state vector ``[S_t, tau_t, delta_{t-1}]``, predicting
the next hedge position, and feeding that position back as the previous hedge
at the next time step.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from baselines.black_scholes import build_time_to_maturity_grid
from config import RuntimeConfig

try:
    import tensorflow as tf  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
    tf = None  # type: ignore[assignment]
    _TENSORFLOW_IMPORT_ERROR = exc
else:
    _TENSORFLOW_IMPORT_ERROR = None


def _require_tensorflow() -> Any:
    """Return TensorFlow or raise a clear error for unsupported environments."""

    if tf is None:
        raise ImportError(
            "TensorFlow is required for policy rollout. "
            "Use a TensorFlow-supported Python version such as 3.10-3.12."
        ) from _TENSORFLOW_IMPORT_ERROR
    return tf


def build_hedge_features(
    spot: Any,
    *,
    time_to_maturity: float | Any,
    previous_hedge: Any,
    predictive_signal: Any | None = None,
) -> Any:
    """Build the baseline hedge-time state vector.

    Args:
        spot: Current spot values with shape ``[n_paths]`` or ``[n_paths, 1]``.
        time_to_maturity: Residual maturity ``tau_t`` in years. May be scalar,
            vector, or column vector broadcastable across paths.
        previous_hedge: Previous hedge ``delta_{t-1}`` with shape ``[n_paths]``
            or ``[n_paths, 1]``.

    Returns:
        Tensor with shape ``[n_paths, 3]`` for the benchmark state or
        ``[n_paths, 4]`` when a predictive signal is supplied.

    Raises:
        ImportError: If TensorFlow is unavailable.
        ValueError: If the provided tensors are not path-aligned.
    """

    tf_module = _require_tensorflow()
    spot_column = _as_column_tensor(spot, name="spot", tf_module=tf_module)
    previous_column = _as_column_tensor(previous_hedge, name="previous_hedge", tf_module=tf_module)

    if spot_column.shape[0] is not None and previous_column.shape[0] is not None:
        if spot_column.shape[0] != previous_column.shape[0]:
            raise ValueError("spot and previous_hedge must have the same number of paths")
    else:
        tf_module.debugging.assert_equal(
            tf_module.shape(spot_column)[0],
            tf_module.shape(previous_column)[0],
            message="spot and previous_hedge must have the same number of paths",
        )

    n_paths = tf_module.shape(spot_column)[0]
    tau_tensor = tf_module.convert_to_tensor(time_to_maturity, dtype=spot_column.dtype)
    if tau_tensor.shape.rank == 0:
        tau_column = tf_module.fill((n_paths, 1), tau_tensor)
    else:
        tau_column = _as_column_tensor(tau_tensor, name="time_to_maturity", tf_module=tf_module)
        if tau_column.shape[0] is not None and spot_column.shape[0] is not None:
            if tau_column.shape[0] != spot_column.shape[0]:
                raise ValueError("time_to_maturity must align with the number of paths")
        else:
            tf_module.debugging.assert_equal(
                tf_module.shape(tau_column)[0],
                n_paths,
                message="time_to_maturity must align with the number of paths",
            )

    columns = [spot_column, tau_column, previous_column]
    if predictive_signal is not None:
        signal_column = _as_column_tensor(predictive_signal, name="predictive_signal", tf_module=tf_module)
        if signal_column.shape[0] is not None and spot_column.shape[0] is not None:
            if signal_column.shape[0] != spot_column.shape[0]:
                raise ValueError("predictive_signal must align with the number of paths")
        else:
            tf_module.debugging.assert_equal(
                tf_module.shape(signal_column)[0],
                n_paths,
                message="predictive_signal must align with the number of paths",
            )
        columns.append(signal_column)

    return tf_module.concat(columns, axis=1)


def rollout_policy(
    path_tensor: Any,
    policy: Any,
    *,
    maturity: float,
    feature_names: Sequence[str] | None = None,
    predictive_signal_tensor: Any | None = None,
    initial_hedge: float | None = 0.0,
    training: bool = False,
) -> Any:
    """Sequentially unroll a hedge policy through time on simulated paths.

    Args:
        path_tensor: Spot paths with shape ``[n_paths, n_steps + 1]``.
        policy: TensorFlow policy mapping the requested hedge-time feature
            vector to the next hedge position of shape ``[n_paths, 1]``.
        maturity: Total maturity in years.
        feature_names: Ordered feature names expected by the policy. Supported
            states are the benchmark hedge state and the same state extended by
            ``predictive_signal``.
        predictive_signal_tensor: Optional predictive-signal observations with
            shape ``[n_paths, n_steps]``.
        initial_hedge: Scalar initial hedge used before the first decision.
        training: Standard model training flag forwarded to the policy.

    Returns:
        Tensor of shape ``[n_paths, n_steps]`` containing the hedge chosen at
        each hedge time and held over the following interval.

    Raises:
        ImportError: If TensorFlow is unavailable.
        ValueError: If the path tensor is malformed or if the policy output does
            not have shape ``[n_paths, 1]``.
    """

    tf_module = _require_tensorflow()
    paths = tf_module.convert_to_tensor(path_tensor, dtype=tf_module.float32)
    if paths.shape.rank != 2:
        raise ValueError("path_tensor must have shape [n_paths, n_steps + 1]")
    if paths.shape[1] is None:
        raise ValueError("path_tensor must have a statically known time dimension")
    if paths.shape[1] <= 1:
        raise ValueError("path_tensor must contain at least one path and one forward time step")

    n_steps = paths.shape[1] - 1
    requested_features = list(feature_names) if feature_names is not None else ["spot", "time_to_maturity", "previous_hedge"]
    supported_feature_sets = {
        ("spot", "time_to_maturity", "previous_hedge"),
        ("spot", "time_to_maturity", "previous_hedge", "predictive_signal"),
    }
    if tuple(requested_features) not in supported_feature_sets:
        raise ValueError("Unsupported feature_names for rollout_policy")

    needs_predictive_signal = "predictive_signal" in requested_features
    signal_tensor = None
    if needs_predictive_signal:
        if predictive_signal_tensor is None:
            raise ValueError("predictive_signal_tensor is required when feature_names include 'predictive_signal'")
        signal_tensor = tf_module.convert_to_tensor(predictive_signal_tensor, dtype=paths.dtype)
        if signal_tensor.shape.rank != 2:
            raise ValueError("predictive_signal_tensor must have shape [n_paths, n_steps]")
        if signal_tensor.shape[1] is not None and signal_tensor.shape[1] != n_steps:
            raise ValueError("predictive_signal_tensor must align with the hedge-time grid")
        if signal_tensor.shape[0] is not None and paths.shape[0] is not None and signal_tensor.shape[0] != paths.shape[0]:
            raise ValueError("predictive_signal_tensor must align with the number of paths")

    tau_grid = build_time_to_maturity_grid(maturity, n_steps, dtype=np.float32)
    previous_hedge = _resolve_previous_hedge(
        n_paths=tf_module.shape(paths)[0],
        initial_hedge=initial_hedge,
        dtype=paths.dtype,
        tf_module=tf_module,
    )

    hedges = []
    for step in range(n_steps):
        features = build_hedge_features(
            paths[:, step],
            time_to_maturity=float(tau_grid[step]),
            previous_hedge=previous_hedge,
            predictive_signal=None if signal_tensor is None else signal_tensor[:, step],
        )
        hedge = tf_module.convert_to_tensor(policy(features, training=training), dtype=paths.dtype)
        if hedge.shape.rank != 2:
            raise ValueError("policy output must have shape [n_paths, 1]")
        if hedge.shape[-1] is not None and hedge.shape[-1] != 1:
            raise ValueError("policy output must have width 1 for the one-asset benchmark")
        if hedge.shape[-1] is None:
            tf_module.debugging.assert_equal(
                tf_module.shape(hedge)[-1],
                1,
                message="policy output must have width 1 for the one-asset benchmark",
            )

        hedges.append(hedge[:, 0])
        previous_hedge = hedge

    return tf_module.stack(hedges, axis=1)


def rollout_policy_from_config(
    config: RuntimeConfig,
    path_tensor: Any,
    policy: Any,
    *,
    predictive_signal_tensor: Any | None = None,
    initial_hedge: float | None = 0.0,
    training: bool = False,
) -> Any:
    """Roll out a hedge policy using the maturity encoded in ``RuntimeConfig``.

    Args:
        config: Validated runtime configuration supplying ``market.maturity``.
        path_tensor: Spot paths with shape ``[n_paths, n_steps + 1]``.
        policy: TensorFlow policy network.
        predictive_signal_tensor: Optional predictive-signal observations with
            shape ``[n_paths, n_steps]``.
        initial_hedge: Scalar initial hedge used before the first decision.
        training: Standard model training flag forwarded to the policy.

    Returns:
        Tensor of shape ``[n_paths, n_steps]`` containing the sequential hedge
        rollout.
    """

    return rollout_policy(
        path_tensor,
        policy,
        maturity=config.market.maturity,
        feature_names=config.model.feature_names,
        predictive_signal_tensor=predictive_signal_tensor,
        initial_hedge=initial_hedge,
        training=training,
    )


def _resolve_previous_hedge(
    *,
    n_paths: Any,
    initial_hedge: float | None,
    dtype: Any,
    tf_module: Any,
) -> Any:
    """Return the hedge state seen by the policy before the first decision."""

    initial_value = 0.0 if initial_hedge is None else float(initial_hedge)
    return tf_module.fill(
        (n_paths, 1),
        tf_module.cast(initial_value, dtype=dtype),
    )


def _as_column_tensor(value: Any, *, name: str, tf_module: Any) -> Any:
    """Convert a vector-like object into a rank-2 column tensor."""

    tensor = tf_module.convert_to_tensor(value, dtype=tf_module.float32)
    if tensor.shape.rank == 1:
        return tensor[:, tf_module.newaxis]
    if tensor.shape.rank == 2:
        if tensor.shape[-1] is not None and tensor.shape[-1] != 1:
            raise ValueError(f"{name} must have shape [n_paths] or [n_paths, 1]")
        if tensor.shape[-1] is None:
            tf_module.debugging.assert_equal(
                tf_module.shape(tensor)[-1],
                1,
                message=f"{name} must have shape [n_paths] or [n_paths, 1]",
            )
        return tensor
    raise ValueError(f"{name} must have shape [n_paths] or [n_paths, 1]")