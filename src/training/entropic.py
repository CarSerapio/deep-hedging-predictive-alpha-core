"""Entropic training loop for deep hedging.

This module trains the policy and rollout stack under the paper-minimal
entropic-risk objective described in the benchmark procedure.
It intentionally keeps the benchmark path simple: direct hedge positions,
zero initial hedge, and no auxiliary OCE intercept layers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from baselines.black_scholes import black_scholes_price_and_delta_from_config
from config import RuntimeConfig
from evaluation import evaluate_hedge_tensor_decomposition
from finance.pnl import compute_portfolio_pnl_from_config, compute_terminal_pnl_from_config
from policies import build_mlp_policy_from_config, rollout_policy_from_config
from risk import entropic_risk
from simulators import simulate_market_data_from_config
from simulators.gbm import simulate_gbm_paths_from_config
from utils.seeding import set_global_seed

try:
    import tensorflow as tf  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
    tf = None  # type: ignore[assignment]
    _TENSORFLOW_IMPORT_ERROR = exc
else:
    _TENSORFLOW_IMPORT_ERROR = None


@dataclass(frozen=True)
class EntropicTrainingResult:
    """Artifacts and summary metrics from one entropic training run."""

    artifact_dir: str
    checkpoint_path: str
    history_path: str
    test_hedges_path: str
    theta: float
    best_epoch: int
    epochs_ran: int
    stopped_early: bool
    initial_train_loss: float
    initial_val_loss: float
    initial_val_adjusted_gap_mean: float
    train_losses: list[float]
    val_losses: list[float]
    best_val_loss: float
    val_adjusted_gap_means: list[float]
    best_val_adjusted_gap_mean: float
    checkpoint_metric: str
    best_checkpoint_score: float
    checkpoint_val_loss: float
    checkpoint_val_adjusted_gap_mean: float
    test_risk: float
    mean_abs_hedge: float
    mean_abs_delta_gap: float
    test_paths: np.ndarray
    test_hedges: np.ndarray
    test_pnl: np.ndarray


def train_entropic_deep_hedger(
    config: RuntimeConfig,
    output_dir: str | Path,
    *,
    theta: float | None = None,
    seed: int | None = None,
    deterministic: bool = False,
    checkpoint_metric: str = "val_loss",
) -> EntropicTrainingResult:
    """Train one deep hedger under entropic risk and persist artifacts."""

    tf_module = _require_tensorflow()
    runtime_config, theta_value = _resolve_entropic_config(config, theta=theta)
    checkpoint_metric_name = _resolve_checkpoint_metric(checkpoint_metric)
    base_seed = runtime_config.paths.seed if seed is None else int(seed)
    set_global_seed(base_seed, deterministic=deterministic)

    train_market = simulate_market_data_from_config(runtime_config, split="train", seed=base_seed)
    val_market = simulate_market_data_from_config(runtime_config, split="validation", seed=base_seed + 1)
    test_market = simulate_market_data_from_config(runtime_config, split="test", seed=base_seed + 2)
    policy = build_mlp_policy_from_config(runtime_config, seed=base_seed)
    optimizer = _build_optimizer(runtime_config, tf_module=tf_module)

    artifact_root = Path(output_dir).expanduser().resolve()
    run_dir = artifact_root / f"theta_{_format_theta_label(theta_value)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "policy.weights.h5"
    history_path = run_dir / "history.json"
    test_hedges_path = run_dir / "test_hedges.npy"

    initial_train_loss = _evaluate_entropic_dataset_risk(
        runtime_config,
        train_market.path_tensor,
        policy,
        predictive_signal_tensor=train_market.predictive_signal_tensor,
        theta=theta_value,
        batch_size=runtime_config.training.batch_size,
    )
    initial_val_loss, initial_val_adjusted_gap_mean = _evaluate_entropic_validation_metrics(
        runtime_config,
        val_market.path_tensor,
        policy,
        predictive_signal_tensor=val_market.predictive_signal_tensor,
        theta=theta_value,
        batch_size=runtime_config.training.batch_size,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_adjusted_gap_means: list[float] = []
    best_val_loss = float("inf")
    best_val_adjusted_gap_mean = float("inf")
    best_checkpoint_score = float("inf")
    checkpoint_val_loss = float("inf")
    checkpoint_val_adjusted_gap_mean = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False

    for epoch_index in range(runtime_config.training.epochs):
        dataset = _build_path_dataset(
            train_market.path_tensor,
            predictive_signal_tensor=train_market.predictive_signal_tensor,
            batch_size=runtime_config.training.batch_size,
            shuffle=True,
            seed=base_seed + epoch_index,
            tf_module=tf_module,
        )
        for batch in dataset:
            if train_market.predictive_signal_tensor is None:
                batch_paths = batch
                batch_predictive_signal = None
            else:
                batch_paths, batch_predictive_signal = batch
            _run_train_step(
                runtime_config,
                batch_paths,
                batch_predictive_signal,
                policy,
                optimizer,
                theta=theta_value,
                tf_module=tf_module,
            )

        train_loss = _evaluate_entropic_dataset_risk(
            runtime_config,
            train_market.path_tensor,
            policy,
            predictive_signal_tensor=train_market.predictive_signal_tensor,
            theta=theta_value,
            batch_size=runtime_config.training.batch_size,
        )
        val_loss, val_adjusted_gap_mean = _evaluate_entropic_validation_metrics(
            runtime_config,
            val_market.path_tensor,
            policy,
            predictive_signal_tensor=val_market.predictive_signal_tensor,
            theta=theta_value,
            batch_size=runtime_config.training.batch_size,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_adjusted_gap_means.append(val_adjusted_gap_mean)
        best_val_loss = min(best_val_loss, val_loss)
        best_val_adjusted_gap_mean = min(best_val_adjusted_gap_mean, val_adjusted_gap_mean)

        checkpoint_score = val_loss if checkpoint_metric_name == "val_loss" else val_adjusted_gap_mean

        if checkpoint_score < best_checkpoint_score - 1e-8:
            best_checkpoint_score = checkpoint_score
            best_epoch = epoch_index + 1
            checkpoint_val_loss = val_loss
            checkpoint_val_adjusted_gap_mean = val_adjusted_gap_mean
            epochs_without_improvement = 0
            policy.save_weights(checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > runtime_config.training.patience:
                stopped_early = True
                break

    if best_epoch == 0:
        raise RuntimeError("Entropic training did not produce a valid checkpoint.")

    policy.load_weights(checkpoint_path)
    test_hedges = _rollout_dataset(
        runtime_config,
        test_market.path_tensor,
        policy,
        predictive_signal_tensor=test_market.predictive_signal_tensor,
        batch_size=runtime_config.training.batch_size,
    )
    test_pnl = _compute_terminal_pnl_array(runtime_config, test_market.path_tensor, test_hedges)
    _, black_scholes_delta = black_scholes_price_and_delta_from_config(runtime_config, test_market.path_tensor)
    test_risk = float(entropic_risk(test_pnl, theta=theta_value))
    mean_abs_hedge = float(np.mean(np.abs(test_hedges), dtype=np.float64))
    mean_abs_delta_gap = float(np.mean(np.abs(test_hedges - black_scholes_delta), dtype=np.float64))

    np.save(test_hedges_path, test_hedges)
    history_payload = {
        "config_hash": runtime_config.config_hash,
        "experiment_name": runtime_config.experiment.name,
        "with_liability": runtime_config.experiment.with_liability,
        "cost_proportional_rate": runtime_config.costs.proportional_rate,
        "theta": theta_value,
        "best_epoch": best_epoch,
        "epochs_ran": len(train_losses),
        "stopped_early": stopped_early,
        "checkpoint_metric": checkpoint_metric_name,
        "best_checkpoint_score": best_checkpoint_score,
        "initial_train_loss": initial_train_loss,
        "initial_val_loss": initial_val_loss,
        "initial_val_adjusted_gap_mean": initial_val_adjusted_gap_mean,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "val_adjusted_gap_means": val_adjusted_gap_means,
        "best_val_adjusted_gap_mean": best_val_adjusted_gap_mean,
        "checkpoint_val_loss": checkpoint_val_loss,
        "checkpoint_val_adjusted_gap_mean": checkpoint_val_adjusted_gap_mean,
        "test_risk": test_risk,
        "mean_abs_hedge": mean_abs_hedge,
        "mean_abs_delta_gap": mean_abs_delta_gap,
    }
    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")

    return EntropicTrainingResult(
        artifact_dir=str(run_dir),
        checkpoint_path=str(checkpoint_path),
        history_path=str(history_path),
        test_hedges_path=str(test_hedges_path),
        theta=theta_value,
        best_epoch=best_epoch,
        epochs_ran=len(train_losses),
        stopped_early=stopped_early,
        initial_train_loss=initial_train_loss,
        initial_val_loss=initial_val_loss,
        initial_val_adjusted_gap_mean=initial_val_adjusted_gap_mean,
        train_losses=train_losses,
        val_losses=val_losses,
        best_val_loss=best_val_loss,
        val_adjusted_gap_means=val_adjusted_gap_means,
        best_val_adjusted_gap_mean=best_val_adjusted_gap_mean,
        checkpoint_metric=checkpoint_metric_name,
        best_checkpoint_score=best_checkpoint_score,
        checkpoint_val_loss=checkpoint_val_loss,
        checkpoint_val_adjusted_gap_mean=checkpoint_val_adjusted_gap_mean,
        test_risk=test_risk,
        mean_abs_hedge=mean_abs_hedge,
        mean_abs_delta_gap=mean_abs_delta_gap,
        test_paths=test_market.path_tensor,
        test_hedges=test_hedges,
        test_pnl=test_pnl,
    )


def train_entropic_variants(
    config: RuntimeConfig,
    output_dir: str | Path,
    *,
    theta_values: Sequence[float] = (1.0, 100.0),
    seed: int | None = None,
    deterministic: bool = False,
    checkpoint_metric: str = "val_loss",
) -> list[EntropicTrainingResult]:
    """Train one or more entropic variants on a shared configuration."""

    if len(theta_values) == 0:
        raise ValueError("theta_values must contain at least one entropic parameter")

    results = []
    for theta_value in theta_values:
        results.append(
            train_entropic_deep_hedger(
                config,
                output_dir,
                theta=float(theta_value),
                seed=seed,
                deterministic=deterministic,
                checkpoint_metric=checkpoint_metric,
            )
        )
    return results


def _resolve_entropic_config(
    config: RuntimeConfig,
    *,
    theta: float | None,
) -> tuple[RuntimeConfig, float]:
    """Return a runtime config whose risk block matches entropic training."""

    if config.risk.kind.lower() != "entropic":
        raise ValueError("Entropic training requires an entropic-risk configuration")

    theta_value = config.risk.theta if theta is None else float(theta)
    if theta_value is None or theta_value <= 0:
        raise ValueError("Entropic training requires a positive entropic theta")

    return replace(config, risk=replace(config.risk, kind="entropic", theta=theta_value, alpha=None)), theta_value


def _require_tensorflow() -> Any:
    """Return TensorFlow or raise a clear environment error."""

    if tf is None:
        raise ImportError(
            "TensorFlow is required for entropic training. "
            "Use a TensorFlow-supported Python version such as 3.10-3.12."
        ) from _TENSORFLOW_IMPORT_ERROR
    return tf


def _resolve_checkpoint_metric(checkpoint_metric: str) -> str:
    """Normalize and validate the checkpoint-selection metric."""

    normalized = checkpoint_metric.strip().lower()
    if normalized == "entropic_risk":
        normalized = "val_loss"
    if normalized not in {"val_loss", "adjusted_gap"}:
        raise ValueError("checkpoint_metric must be one of: val_loss, adjusted_gap")
    return normalized


def _simulate_training_splits(
    config: RuntimeConfig,
    *,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the train, validation, and test path batches for one run."""

    train_paths = simulate_gbm_paths_from_config(config, split="train", seed=base_seed)
    val_paths = simulate_gbm_paths_from_config(config, split="validation", seed=base_seed + 1)
    test_paths = simulate_gbm_paths_from_config(config, split="test", seed=base_seed + 2)
    return train_paths, val_paths, test_paths


def _build_optimizer(config: RuntimeConfig, *, tf_module: Any) -> Any:
    """Build the configured TensorFlow optimizer."""

    optimizer_name = config.training.optimizer.lower()
    if optimizer_name != "adam":
        raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")

    return tf_module.keras.optimizers.Adam(learning_rate=config.training.learning_rate)


def _build_path_dataset(
    path_tensor: np.ndarray,
    *,
    predictive_signal_tensor: np.ndarray | None = None,
    batch_size: int,
    shuffle: bool,
    seed: int,
    tf_module: Any,
) -> Any:
    """Build a TensorFlow dataset of path batches."""

    if predictive_signal_tensor is None:
        dataset = tf_module.data.Dataset.from_tensor_slices(path_tensor.astype(np.float32, copy=False))
    else:
        signals = np.asarray(predictive_signal_tensor, dtype=np.float32)
        if signals.shape[0] != path_tensor.shape[0]:
            raise ValueError("predictive_signal_tensor must align with path_tensor")
        dataset = tf_module.data.Dataset.from_tensor_slices(
            (path_tensor.astype(np.float32, copy=False), signals)
        )
    if shuffle:
        buffer_size = min(path_tensor.shape[0], max(batch_size * 4, batch_size))
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True)
    return dataset.batch(batch_size, drop_remainder=False).prefetch(tf_module.data.AUTOTUNE)


def _run_train_step(
    config: RuntimeConfig,
    batch_paths: Any,
    batch_predictive_signal: Any | None,
    policy: Any,
    optimizer: Any,
    *,
    theta: float,
    tf_module: Any,
) -> float:
    """Execute one SGD update on a batch of simulated paths."""

    with tf_module.GradientTape() as tape:
        hedge_tensor = rollout_policy_from_config(
            config,
            batch_paths,
            policy,
            predictive_signal_tensor=batch_predictive_signal,
            training=True,
        )
        pnl_sample = _compute_terminal_pnl_tensor(config, batch_paths, hedge_tensor, tf_module=tf_module)
        loss = _entropic_risk_tensor(pnl_sample, theta=theta, tf_module=tf_module)

    trainable_variables = list(policy.trainable_variables)

    gradients = tape.gradient(loss, trainable_variables)
    gradient_pairs = [(gradient, variable) for gradient, variable in zip(gradients, trainable_variables) if gradient is not None]
    if len(gradient_pairs) == 0:
        raise RuntimeError("Entropic training produced no gradients for the policy network")

    clipped_gradients, _ = tf_module.clip_by_global_norm(
        [gradient for gradient, _ in gradient_pairs],
        config.training.gradient_clip,
    )
    optimizer.apply_gradients(zip(clipped_gradients, [variable for _, variable in gradient_pairs]))
    return float(loss.numpy())


def _evaluate_entropic_dataset_risk(
    config: RuntimeConfig,
    path_tensor: np.ndarray,
    policy: Any,
    *,
    predictive_signal_tensor: np.ndarray | None = None,
    theta: float,
    batch_size: int,
) -> float:
    """Evaluate entropic risk on a full dataset."""

    pnl_sample = _collect_dataset_pnl(
        config,
        path_tensor,
        policy,
        predictive_signal_tensor=predictive_signal_tensor,
        batch_size=batch_size,
    )
    return float(entropic_risk(pnl_sample, theta=theta))


def _evaluate_entropic_validation_metrics(
    config: RuntimeConfig,
    path_tensor: np.ndarray,
    policy: Any,
    *,
    predictive_signal_tensor: np.ndarray | None = None,
    theta: float,
    batch_size: int,
) -> tuple[float, float]:
    """Evaluate validation entropic risk and the mean adjusted gap together."""

    hedge_tensor = _rollout_dataset(
        config,
        path_tensor,
        policy,
        predictive_signal_tensor=predictive_signal_tensor,
        batch_size=batch_size,
    )
    pnl_sample = _compute_terminal_pnl_array(config, path_tensor, hedge_tensor)
    val_loss = float(entropic_risk(pnl_sample, theta=theta))
    decomposition_metrics = evaluate_hedge_tensor_decomposition(config, path_tensor, hedge_tensor)
    if decomposition_metrics.adjusted_gap is None:
        raise RuntimeError("Entropic validation expected an adjusted-gap diagnostic")
    adjusted_gap_mean = float(np.mean(decomposition_metrics.adjusted_gap, dtype=np.float64))
    return val_loss, adjusted_gap_mean


def _collect_dataset_pnl(
    config: RuntimeConfig,
    path_tensor: np.ndarray,
    policy: Any,
    *,
    predictive_signal_tensor: np.ndarray | None = None,
    batch_size: int,
) -> np.ndarray:
    """Roll out a trained policy on a dataset and collect its pathwise PnL."""

    hedge_tensor = _rollout_dataset(
        config,
        path_tensor,
        policy,
        predictive_signal_tensor=predictive_signal_tensor,
        batch_size=batch_size,
    )
    return _compute_terminal_pnl_array(config, path_tensor, hedge_tensor)


def _rollout_dataset(
    config: RuntimeConfig,
    path_tensor: np.ndarray,
    policy: Any,
    *,
    predictive_signal_tensor: np.ndarray | None = None,
    batch_size: int,
) -> np.ndarray:
    """Roll out a policy over a full path tensor in mini-batches."""

    hedges = []
    n_paths = path_tensor.shape[0]
    for start in range(0, n_paths, batch_size):
        stop = min(start + batch_size, n_paths)
        batch_hedges = rollout_policy_from_config(
            config,
            path_tensor[start:stop],
            policy,
            predictive_signal_tensor=None if predictive_signal_tensor is None else predictive_signal_tensor[start:stop],
            training=False,
        )
        hedges.append(np.asarray(batch_hedges.numpy(), dtype=np.float32))
    return np.concatenate(hedges, axis=0)


def _compute_terminal_pnl_array(
    config: RuntimeConfig,
    path_tensor: np.ndarray,
    hedge_tensor: np.ndarray,
) -> np.ndarray:
    """Compute terminal PnL under the current liability flag."""

    if config.experiment.with_liability:
        return compute_terminal_pnl_from_config(config, path_tensor, hedge_tensor)
    return compute_portfolio_pnl_from_config(config, path_tensor, hedge_tensor)


def _compute_terminal_pnl_tensor(
    config: RuntimeConfig,
    path_tensor: Any,
    hedge_tensor: Any,
    *,
    tf_module: Any,
) -> Any:
    """TensorFlow version of the terminal PnL objective."""

    paths = tf_module.convert_to_tensor(path_tensor, dtype=tf_module.float32)
    hedges = tf_module.convert_to_tensor(hedge_tensor, dtype=paths.dtype)
    gains = tf_module.reduce_sum(hedges * (paths[:, 1:] - paths[:, :-1]), axis=1)
    trades = tf_module.concat([hedges[:, :1], hedges[:, 1:] - hedges[:, :-1]], axis=1)
    cost_rate = tf_module.cast(config.costs.proportional_rate, dtype=paths.dtype)
    costs = cost_rate * tf_module.reduce_sum(tf_module.math.abs(trades) * paths[:, :-1], axis=1)
    if not config.experiment.with_liability:
        return gains - costs

    strike = tf_module.cast(config.market.strike, dtype=paths.dtype)
    payoff = tf_module.nn.relu(paths[:, -1] - strike)
    return gains - payoff - costs


def _entropic_risk_tensor(pnl_sample: Any, *, theta: float, tf_module: Any) -> Any:
    """Differentiable TensorFlow entropic risk used during optimization."""

    pnl_tensor = tf_module.convert_to_tensor(pnl_sample, dtype=tf_module.float32)
    theta_tensor = tf_module.cast(theta, dtype=pnl_tensor.dtype)
    sample_count = tf_module.cast(tf_module.shape(pnl_tensor)[0], dtype=pnl_tensor.dtype)
    return (tf_module.reduce_logsumexp(-theta_tensor * pnl_tensor) - tf_module.math.log(sample_count)) / theta_tensor


def _format_theta_label(theta: float) -> str:
    """Format theta so it stays readable inside artifact directory names."""

    return format(theta, "g").replace("-", "neg_").replace(".", "p")