"""Empirical decomposition helpers for saved policy artifacts.

The empirical extension compares a no-liability policy ``delta^{0,*}`` and a
with-liability policy ``delta^{Z,*}`` trained under the same market regime and
cost setting. The resulting empirical hedge component is

    eta^Z = delta^{Z,*} - delta^{0,*}.

This module validates that the two policies are matched on the dimensions that
make the subtraction meaningful, computes ``eta^Z`` pathwise, and compares the
resulting hedge component against the Black-Scholes delta benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from baselines.black_scholes import black_scholes_price_and_delta_from_config
from config import RuntimeConfig, load_config
from evaluation.benchmark_decomposition import compute_pathwise_normalized_gap, summarize_pathwise_metric
from simulators.gbm import simulate_gbm_paths_from_config


@dataclass(frozen=True)
class EmpiricalDecompositionMetrics:
    """Pathwise decomposition artifacts for one matched policy pair."""

    no_liability_config_hash: str
    with_liability_config_hash: str
    risk_kind: str
    cost_proportional_rate: float
    theta: float | None
    alpha: float | None
    n_paths: int
    n_steps: int
    no_liability_hedge: np.ndarray
    with_liability_hedge: np.ndarray
    hedge_component: np.ndarray
    benchmark_delta: np.ndarray
    benchmark_gap: np.ndarray
    reconstructed_with_liability_max_abs_error: float

    def to_summary_dict(self) -> dict[str, object]:
        """Return a JSON-serializable summary payload."""

        payload: dict[str, object] = {
            "no_liability_config_hash": self.no_liability_config_hash,
            "with_liability_config_hash": self.with_liability_config_hash,
            "risk_kind": self.risk_kind,
            "cost_proportional_rate": self.cost_proportional_rate,
            "n_paths": self.n_paths,
            "n_steps": self.n_steps,
            "path_norm": "euclidean_over_time",
            "benchmark_reference": "black_scholes_delta",
            "reconstructed_with_liability_max_abs_error": self.reconstructed_with_liability_max_abs_error,
            "mean_abs_no_liability_hedge": float(np.mean(np.abs(self.no_liability_hedge), dtype=np.float64)),
            "mean_abs_with_liability_hedge": float(np.mean(np.abs(self.with_liability_hedge), dtype=np.float64)),
            "mean_abs_hedge_component": float(np.mean(np.abs(self.hedge_component), dtype=np.float64)),
            "mean_abs_delta_gap": float(np.mean(np.abs(self.hedge_component - self.benchmark_delta), dtype=np.float64)),
            "benchmark_gap": summarize_pathwise_metric(self.benchmark_gap),
        }
        if self.theta is not None:
            payload["theta"] = self.theta
        if self.alpha is not None:
            payload["alpha"] = self.alpha
        return payload


def evaluate_empirical_decomposition(
    no_liability_config: RuntimeConfig,
    with_liability_config: RuntimeConfig,
    path_tensor: np.ndarray,
    no_liability_hedge_tensor: np.ndarray,
    with_liability_hedge_tensor: np.ndarray,
) -> EmpiricalDecompositionMetrics:
    """Compute the empirical decomposition for one matched policy pair."""

    _validate_matched_policy_configs(no_liability_config, with_liability_config)

    no_liability_hedge = np.asarray(no_liability_hedge_tensor, dtype=np.float64)
    with_liability_hedge = np.asarray(with_liability_hedge_tensor, dtype=np.float64)
    if no_liability_hedge.shape != with_liability_hedge.shape:
        raise ValueError("no_liability_hedge_tensor and with_liability_hedge_tensor must have the same shape")
    if no_liability_hedge.ndim != 2 or no_liability_hedge.shape[1] <= 0:
        raise ValueError("hedge tensors must have shape [n_paths, n_steps]")

    _, benchmark_delta = black_scholes_price_and_delta_from_config(with_liability_config, path_tensor)
    if benchmark_delta.shape != no_liability_hedge.shape:
        raise ValueError("hedge tensors do not align with the Black-Scholes hedge grid for the supplied paths")

    hedge_component = with_liability_hedge - no_liability_hedge
    benchmark_gap = compute_pathwise_normalized_gap(hedge_component, benchmark_delta)
    reconstruction_error = float(np.max(np.abs(no_liability_hedge + hedge_component - with_liability_hedge)))

    return EmpiricalDecompositionMetrics(
        no_liability_config_hash=no_liability_config.config_hash,
        with_liability_config_hash=with_liability_config.config_hash,
        risk_kind=with_liability_config.risk.kind.lower(),
        cost_proportional_rate=float(with_liability_config.costs.proportional_rate),
        theta=None if with_liability_config.risk.theta is None else float(with_liability_config.risk.theta),
        alpha=None if with_liability_config.risk.alpha is None else float(with_liability_config.risk.alpha),
        n_paths=int(no_liability_hedge.shape[0]),
        n_steps=int(no_liability_hedge.shape[1]),
        no_liability_hedge=no_liability_hedge,
        with_liability_hedge=with_liability_hedge,
        hedge_component=hedge_component,
        benchmark_delta=np.asarray(benchmark_delta, dtype=np.float64),
        benchmark_gap=benchmark_gap,
        reconstructed_with_liability_max_abs_error=reconstruction_error,
    )


def evaluate_saved_empirical_decomposition(
    no_liability_config_path: Path,
    no_liability_hedges_path: Path,
    with_liability_config_path: Path,
    with_liability_hedges_path: Path,
) -> EmpiricalDecompositionMetrics:
    """Load matched saved hedges and evaluate the empirical decomposition."""

    no_liability_config = load_config(no_liability_config_path)
    with_liability_config = load_config(with_liability_config_path)
    path_tensor = simulate_gbm_paths_from_config(
        no_liability_config,
        split="test",
        seed=no_liability_config.paths.seed + 2,
    )
    no_liability_hedge = np.load(no_liability_hedges_path)
    with_liability_hedge = np.load(with_liability_hedges_path)
    return evaluate_empirical_decomposition(
        no_liability_config,
        with_liability_config,
        path_tensor,
        no_liability_hedge,
        with_liability_hedge,
    )


def _validate_matched_policy_configs(
    no_liability_config: RuntimeConfig,
    with_liability_config: RuntimeConfig,
) -> None:
    """Ensure the two saved policies belong to the same decomposition regime."""

    if no_liability_config.experiment.with_liability:
        raise ValueError("no_liability_config must have experiment.with_liability=false")
    if not with_liability_config.experiment.with_liability:
        raise ValueError("with_liability_config must have experiment.with_liability=true")
    if no_liability_config.risk.kind.lower() != with_liability_config.risk.kind.lower():
        raise ValueError("both configs must use the same risk kind")

    _require_close(
        float(no_liability_config.costs.proportional_rate),
        float(with_liability_config.costs.proportional_rate),
        field_name="costs.proportional_rate",
    )
    _require_close(float(no_liability_config.market.s0), float(with_liability_config.market.s0), field_name="market.s0")
    _require_close(float(no_liability_config.market.mu), float(with_liability_config.market.mu), field_name="market.mu")
    _require_close(float(no_liability_config.market.sigma), float(with_liability_config.market.sigma), field_name="market.sigma")
    _require_close(float(no_liability_config.market.maturity), float(with_liability_config.market.maturity), field_name="market.maturity")
    _require_close(float(no_liability_config.market.dt), float(with_liability_config.market.dt), field_name="market.dt")
    _require_close(float(no_liability_config.market.strike), float(with_liability_config.market.strike), field_name="market.strike")
    if int(no_liability_config.market.n_steps) != int(with_liability_config.market.n_steps):
        raise ValueError("market.n_steps must match across the decomposition pair")

    if int(no_liability_config.paths.test_paths) != int(with_liability_config.paths.test_paths):
        raise ValueError("paths.test_paths must match across the decomposition pair")
    if int(no_liability_config.paths.seed) != int(with_liability_config.paths.seed):
        raise ValueError("paths.seed must match across the decomposition pair")

    if list(no_liability_config.model.feature_names) != list(with_liability_config.model.feature_names):
        raise ValueError("model.feature_names must match across the decomposition pair")

    no_theta = no_liability_config.risk.theta
    with_theta = with_liability_config.risk.theta
    if (no_theta is None) != (with_theta is None):
        raise ValueError("risk.theta presence must match across the decomposition pair")
    if no_theta is not None and with_theta is not None:
        _require_close(float(no_theta), float(with_theta), field_name="risk.theta")

    no_alpha = no_liability_config.risk.alpha
    with_alpha = with_liability_config.risk.alpha
    if (no_alpha is None) != (with_alpha is None):
        raise ValueError("risk.alpha presence must match across the decomposition pair")
    if no_alpha is not None and with_alpha is not None:
        _require_close(float(no_alpha), float(with_alpha), field_name="risk.alpha")


def _require_close(left: float, right: float, *, field_name: str, atol: float = 1e-12) -> None:
    if not np.isclose(left, right, atol=atol, rtol=0.0):
        raise ValueError(f"{field_name} must match across the decomposition pair")