"""CVaR training loop for deep hedging.

This module trains the policy and rollout stack under a smoothed CVaR
objective. The implementation uses the standard auxiliary-variable
formulation with a trainable scalar ``eta`` and produces the same artifact family as entropic training:

- a trained checkpoint for the best validation epoch,
- serialized training-curve data,
- and held-out hedge tensors for the test split.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np

from baselines.black_scholes import black_scholes_price_and_delta_from_config
from config import RuntimeConfig
from risk import cvar_risk
from simulators import simulate_market_data_from_config
from training.entropic import (
    _build_optimizer,
    _build_path_dataset,
    _collect_dataset_pnl,
    _compute_terminal_pnl_tensor,
    _require_tensorflow,
    _rollout_dataset,
    _simulate_training_splits,
)
from policies import build_mlp_policy_from_config
from utils.seeding import set_global_seed


@dataclass(frozen=True)
class CVaRTrainingResult:
    """Artifacts and summary metrics from one CVaR training run."""

    artifact_dir: str
    checkpoint_path: str
    history_path: str
    test_hedges_path: str
    alpha: float
    best_eta: float
    best_epoch: int
    epochs_ran: int
    stopped_early: bool
    initial_train_loss: float
    initial_val_loss: float
    train_losses: list[float]
    val_losses: list[float]
    eta_history: list[float]
    best_val_loss: float
    test_risk: float
    mean_abs_hedge: float
    mean_abs_delta_gap: float
    test_paths: np.ndarray
    test_hedges: np.ndarray
    test_pnl: np.ndarray


def train_cvar_deep_hedger(
    config: RuntimeConfig,
    output_dir: str | Path,
    *,
    alpha: float | None = None,
    seed: int | None = None,
    deterministic: bool = False,
    smoothing: float = 1e-3,
    warm_start_epochs: int = 0,
    delta_anchor_weight: float = 0.0,
) -> CVaRTrainingResult:
    """Train one deep hedger under CVaR and persist artifacts."""

    tf_module = _require_tensorflow()
    runtime_config, alpha_value = _resolve_cvar_config(config, alpha=alpha)
    if runtime_config.signal.enabled:
        raise ValueError("The predictive-signal extension is currently wired through the entropic training track only.")
    base_seed = runtime_config.paths.seed if seed is None else int(seed)
    set_global_seed(base_seed, deterministic=deterministic)

    train_paths, val_paths, test_paths = _simulate_training_splits(runtime_config, base_seed=base_seed)
    policy = build_mlp_policy_from_config(runtime_config, seed=base_seed)
    warm_start_history = _warm_start_policy_to_black_scholes(
        runtime_config,
        train_paths,
        val_paths,
        policy,
        epochs=warm_start_epochs,
        batch_size=runtime_config.training.batch_size,
        seed=base_seed,
        tf_module=tf_module,
    )
    optimizer = _build_optimizer(runtime_config, tf_module=tf_module)
    eta = tf_module.Variable(0.0, trainable=True, dtype=tf_module.float32, name="cvar_eta")

    artifact_root = Path(output_dir).expanduser().resolve()
    run_dir = artifact_root / f"alpha_{_format_alpha_label(alpha_value)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "policy.weights.h5"
    history_path = run_dir / "history.json"
    test_hedges_path = run_dir / "test_hedges.npy"

    initial_train_loss = _evaluate_cvar_dataset_risk(
        runtime_config,
        train_paths,
        policy,
        alpha=alpha_value,
        batch_size=runtime_config.training.batch_size,
        smoothing=smoothing,
    )
    initial_val_loss = _evaluate_cvar_dataset_risk(
        runtime_config,
        val_paths,
        policy,
        alpha=alpha_value,
        batch_size=runtime_config.training.batch_size,
        smoothing=smoothing,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    eta_history: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_eta = 0.0
    epochs_without_improvement = 0
    stopped_early = False

    for epoch_index in range(runtime_config.training.epochs):
        dataset = _build_path_dataset(
            train_paths,
            batch_size=runtime_config.training.batch_size,
            shuffle=True,
            seed=base_seed + epoch_index,
            tf_module=tf_module,
        )
        for batch_paths in dataset:
            _run_cvar_train_step(
                runtime_config,
                batch_paths,
                policy,
                eta,
                optimizer,
                alpha=alpha_value,
                smoothing=smoothing,
                delta_anchor_weight=delta_anchor_weight,
                tf_module=tf_module,
            )

        train_loss = _evaluate_cvar_dataset_risk(
            runtime_config,
            train_paths,
            policy,
            alpha=alpha_value,
            batch_size=runtime_config.training.batch_size,
            smoothing=smoothing,
        )
        val_loss = _evaluate_cvar_dataset_risk(
            runtime_config,
            val_paths,
            policy,
            alpha=alpha_value,
            batch_size=runtime_config.training.batch_size,
            smoothing=smoothing,
        )
        current_eta = float(eta.numpy())
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        eta_history.append(current_eta)

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_epoch = epoch_index + 1
            best_eta = current_eta
            epochs_without_improvement = 0
            policy.save_weights(checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > runtime_config.training.patience:
                stopped_early = True
                break

    if best_epoch == 0:
        raise RuntimeError("CVaR training did not produce a valid checkpoint.")

    policy.load_weights(checkpoint_path)
    test_hedges = _rollout_dataset(
        runtime_config,
        test_paths,
        policy,
        batch_size=runtime_config.training.batch_size,
    )
    test_pnl = _collect_dataset_pnl(
        runtime_config,
        test_paths,
        policy,
        batch_size=runtime_config.training.batch_size,
    )
    _, black_scholes_delta = black_scholes_price_and_delta_from_config(runtime_config, test_paths)
    test_risk = float(cvar_risk(test_pnl, alpha=alpha_value, smoothing=smoothing))
    mean_abs_hedge = float(np.mean(np.abs(test_hedges), dtype=np.float64))
    mean_abs_delta_gap = float(np.mean(np.abs(test_hedges - black_scholes_delta), dtype=np.float64))

    np.save(test_hedges_path, test_hedges)
    history_payload = {
        "config_hash": runtime_config.config_hash,
        "alpha": alpha_value,
        "warm_start_epochs": warm_start_epochs,
        "delta_anchor_weight": delta_anchor_weight,
        "warm_start_epochs_ran": len(warm_start_history["train_losses"]),
        "warm_start_train_losses": warm_start_history["train_losses"],
        "warm_start_val_losses": warm_start_history["val_losses"],
        "best_eta": best_eta,
        "best_epoch": best_epoch,
        "epochs_ran": len(train_losses),
        "stopped_early": stopped_early,
        "initial_train_loss": initial_train_loss,
        "initial_val_loss": initial_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "eta_history": eta_history,
        "best_val_loss": best_val_loss,
        "test_risk": test_risk,
        "mean_abs_hedge": mean_abs_hedge,
        "mean_abs_delta_gap": mean_abs_delta_gap,
    }
    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")

    return CVaRTrainingResult(
        artifact_dir=str(run_dir),
        checkpoint_path=str(checkpoint_path),
        history_path=str(history_path),
        test_hedges_path=str(test_hedges_path),
        alpha=alpha_value,
        best_eta=best_eta,
        best_epoch=best_epoch,
        epochs_ran=len(train_losses),
        stopped_early=stopped_early,
        initial_train_loss=initial_train_loss,
        initial_val_loss=initial_val_loss,
        train_losses=train_losses,
        val_losses=val_losses,
        eta_history=eta_history,
        best_val_loss=best_val_loss,
        test_risk=test_risk,
        mean_abs_hedge=mean_abs_hedge,
        mean_abs_delta_gap=mean_abs_delta_gap,
        test_paths=test_paths,
        test_hedges=test_hedges,
        test_pnl=test_pnl,
    )


def train_cvar_variants(
    config: RuntimeConfig,
    output_dir: str | Path,
    *,
    alpha_values: Sequence[float] = (0.5,),
    seed: int | None = None,
    deterministic: bool = False,
    smoothing: float = 1e-3,
    warm_start_epochs: int = 0,
    delta_anchor_weight: float = 0.0,
) -> list[CVaRTrainingResult]:
    """Train one or more CVaR variants on a shared configuration."""

    if len(alpha_values) == 0:
        raise ValueError("alpha_values must contain at least one CVaR parameter")

    results = []
    for alpha_value in alpha_values:
        results.append(
            train_cvar_deep_hedger(
                config,
                output_dir,
                alpha=float(alpha_value),
                seed=seed,
                deterministic=deterministic,
                smoothing=smoothing,
                warm_start_epochs=warm_start_epochs,
                delta_anchor_weight=delta_anchor_weight,
            )
        )
    return results


def _resolve_cvar_config(
    config: RuntimeConfig,
    *,
    alpha: float | None,
) -> tuple[RuntimeConfig, float]:
    """Return a runtime config whose risk block matches CVaR training."""

    if config.risk.kind.lower() != "cvar":
        raise ValueError("CVaR training requires a CVaR-risk configuration")

    alpha_value = config.risk.alpha if alpha is None else float(alpha)
    if alpha_value is None or not (0.0 < alpha_value <= 1.0):
        raise ValueError("CVaR training requires alpha in (0, 1]")

    return replace(config, risk=replace(config.risk, kind="cvar", theta=None, alpha=alpha_value)), alpha_value


def _run_cvar_train_step(
    config: RuntimeConfig,
    batch_paths,
    policy,
    eta,
    optimizer,
    *,
    alpha: float,
    smoothing: float,
    delta_anchor_weight: float,
    tf_module,
) -> float:
    """Execute one SGD update on a batch of simulated paths under CVaR."""

    with tf_module.GradientTape() as tape:
        hedge_tensor = _rollout_dataset_for_training(config, batch_paths, policy)
        pnl_sample = _compute_terminal_pnl_tensor(config, batch_paths, hedge_tensor, tf_module=tf_module)
        loss = _cvar_risk_tensor(
            pnl_sample,
            eta=eta,
            alpha=alpha,
            smoothing=smoothing,
            tf_module=tf_module,
        )
        if delta_anchor_weight > 0.0:
            _, target_delta = black_scholes_price_and_delta_from_config(config, np.asarray(batch_paths))
            target_tensor = tf_module.convert_to_tensor(target_delta, dtype=tf_module.float32)
            loss = loss + tf_module.cast(delta_anchor_weight, dtype=loss.dtype) * tf_module.reduce_mean(
                tf_module.square(hedge_tensor - target_tensor)
            )

    trainable_variables = list(policy.trainable_variables) + [eta]
    gradients = tape.gradient(loss, trainable_variables)
    gradient_pairs = [(gradient, variable) for gradient, variable in zip(gradients, trainable_variables) if gradient is not None]
    if len(gradient_pairs) == 0:
        raise RuntimeError("CVaR training produced no gradients for the policy network")

    clipped_gradients, _ = tf_module.clip_by_global_norm(
        [gradient for gradient, _ in gradient_pairs],
        config.training.gradient_clip,
    )
    optimizer.apply_gradients(zip(clipped_gradients, [variable for _, variable in gradient_pairs]))
    return float(loss.numpy())


def _warm_start_policy_to_black_scholes(
    config: RuntimeConfig,
    train_paths: np.ndarray,
    val_paths: np.ndarray,
    policy,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    tf_module,
) -> dict[str, list[float]]:
    """Warm-start the policy by imitating Black-Scholes deltas."""

    if epochs <= 0:
        return {"train_losses": [], "val_losses": []}

    optimizer = _build_optimizer(config, tf_module=tf_module)
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_weights = policy.get_weights()

    for epoch_index in range(epochs):
        dataset = _build_path_dataset(
            train_paths,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch_index,
            tf_module=tf_module,
        )
        for batch_paths in dataset:
            _run_black_scholes_warm_start_step(
                config,
                batch_paths,
                policy,
                optimizer,
                tf_module=tf_module,
            )

        train_loss = _evaluate_black_scholes_anchor_mse(config, train_paths, policy, batch_size=batch_size)
        val_loss = _evaluate_black_scholes_anchor_mse(config, val_paths, policy, batch_size=batch_size)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_weights = policy.get_weights()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > config.training.patience:
                break

    policy.set_weights(best_weights)
    return {"train_losses": train_losses, "val_losses": val_losses}


def _run_black_scholes_warm_start_step(
    config: RuntimeConfig,
    batch_paths,
    policy,
    optimizer,
    *,
    tf_module,
) -> float:
    """Run one supervised warm-start update toward the BS hedge benchmark."""

    _, target_delta = black_scholes_price_and_delta_from_config(config, np.asarray(batch_paths))
    target_tensor = tf_module.convert_to_tensor(target_delta, dtype=tf_module.float32)

    with tf_module.GradientTape() as tape:
        hedge_tensor = _rollout_dataset_for_training(config, batch_paths, policy)
        loss = tf_module.reduce_mean(tf_module.square(hedge_tensor - target_tensor))

    gradients = tape.gradient(loss, policy.trainable_variables)
    gradient_pairs = [(gradient, variable) for gradient, variable in zip(gradients, policy.trainable_variables) if gradient is not None]
    if len(gradient_pairs) == 0:
        raise RuntimeError("CVaR warm start produced no gradients for the policy network")

    clipped_gradients, _ = tf_module.clip_by_global_norm(
        [gradient for gradient, _ in gradient_pairs],
        config.training.gradient_clip,
    )
    optimizer.apply_gradients(zip(clipped_gradients, [variable for _, variable in gradient_pairs]))
    return float(loss.numpy())


def _evaluate_black_scholes_anchor_mse(
    config: RuntimeConfig,
    path_tensor: np.ndarray,
    policy,
    *,
    batch_size: int,
) -> float:
    """Evaluate rollout MSE against Black-Scholes deltas on a dataset."""

    learned_hedges = _rollout_dataset(
        config,
        path_tensor,
        policy,
        batch_size=batch_size,
    )
    _, black_scholes_delta = black_scholes_price_and_delta_from_config(config, path_tensor)
    return float(np.mean(np.square(learned_hedges - black_scholes_delta), dtype=np.float64))


def _evaluate_cvar_dataset_risk(
    config: RuntimeConfig,
    path_tensor: np.ndarray,
    policy,
    *,
    alpha: float,
    batch_size: int,
    smoothing: float,
) -> float:
    """Evaluate the exact smoothed CVaR risk on a full dataset by batching rollout."""

    pnl_sample = _collect_dataset_pnl(
        config,
        path_tensor,
        policy,
        batch_size=batch_size,
    )
    return float(cvar_risk(pnl_sample, alpha=alpha, smoothing=smoothing))


def _rollout_dataset_for_training(config: RuntimeConfig, batch_paths, policy):
    """Roll out the policy inside the GradientTape graph for one batch."""

    from policies import rollout_policy_from_config

    return rollout_policy_from_config(config, batch_paths, policy, training=True)


def _cvar_risk_tensor(
    pnl_sample,
    *,
    eta,
    alpha: float,
    smoothing: float,
    tf_module,
):
    """Differentiable TensorFlow CVaR objective used during training."""

    pnl_tensor = tf_module.convert_to_tensor(pnl_sample, dtype=tf_module.float32)
    eta_tensor = tf_module.cast(eta, dtype=pnl_tensor.dtype)
    alpha_tensor = tf_module.cast(alpha, dtype=pnl_tensor.dtype)
    smoothing_tensor = tf_module.cast(smoothing, dtype=pnl_tensor.dtype)
    losses = -pnl_tensor
    excess = losses - eta_tensor
    if smoothing == 0.0:
        tail_penalty = tf_module.nn.relu(excess)
    else:
        tail_penalty = smoothing_tensor * tf_module.nn.softplus(excess / smoothing_tensor)
    return eta_tensor + tf_module.reduce_mean(tail_penalty) / alpha_tensor


def _format_alpha_label(alpha: float) -> str:
    """Format alpha so it stays readable inside artifact directory names."""

    return format(alpha, "g").replace("-", "neg_").replace(".", "p")