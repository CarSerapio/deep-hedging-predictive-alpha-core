"""Residual diagnostics for empirical hedge decomposition.

These diagnostics operate on the empirical liability hedge component

    eta^Z = delta^{Z,*} - delta^{0,*},

and study the residual relative to the Black-Scholes delta benchmark,

    r = eta^Z - Delta.

The current implementation keeps the diagnostics deliberately compact: it
summarizes the residual pathwise, by hedge step, by moneyness bucket, and by
time bucket on the exact rollout grid used to construct the saved decomposition
artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baselines.black_scholes import build_time_to_maturity_grid
from config import RuntimeConfig
from evaluation.benchmark_decomposition import compute_pathwise_normalized_gap, summarize_pathwise_metric


@dataclass(frozen=True)
class ResidualDiagnosticsMetrics:
    """Pathwise residual diagnostics for one saved decomposition regime."""

    config_hash: str
    risk_kind: str
    cost_proportional_rate: float
    theta: float | None
    alpha: float | None
    n_paths: int
    n_steps: int
    hedge_component: np.ndarray
    benchmark_delta: np.ndarray
    residual: np.ndarray
    moneyness: np.ndarray
    time_to_maturity: np.ndarray
    normalized_gap: np.ndarray

    def to_summary_dict(self) -> dict[str, object]:
        """Return a JSON-serializable residual-diagnostics summary."""

        mean_residual_profile = np.mean(self.residual, axis=0, dtype=np.float64)
        mean_abs_residual_profile = np.mean(np.abs(self.residual), axis=0, dtype=np.float64)
        peak_step = int(np.argmax(mean_abs_residual_profile))

        payload: dict[str, object] = {
            "config_hash": self.config_hash,
            "risk_kind": self.risk_kind,
            "cost_proportional_rate": self.cost_proportional_rate,
            "n_paths": self.n_paths,
            "n_steps": self.n_steps,
            "residual_definition": "eta^Z_minus_black_scholes_delta",
            "path_norm": "euclidean_over_time",
            "normalization_reference": "black_scholes_delta",
            "benchmark_gap": summarize_pathwise_metric(self.normalized_gap),
            "mean_residual": float(np.mean(self.residual, dtype=np.float64)),
            "mean_abs_residual": float(np.mean(np.abs(self.residual), dtype=np.float64)),
            "rmse_residual": float(np.sqrt(np.mean(np.square(self.residual), dtype=np.float64))),
            "max_abs_residual": float(np.max(np.abs(self.residual))),
            "peak_mean_abs_residual_step": peak_step,
            "peak_mean_abs_residual_time_to_maturity": float(self.time_to_maturity[peak_step]),
            "peak_mean_abs_residual_value": float(mean_abs_residual_profile[peak_step]),
            "abs_residual_moneyness_distance_correlation": _safe_correlation(
                np.abs(self.residual).reshape(-1),
                np.abs(self.moneyness - 1.0).reshape(-1),
            ),
            "abs_residual_time_to_maturity_correlation": _safe_correlation(
                np.abs(self.residual).reshape(-1),
                np.broadcast_to(self.time_to_maturity[np.newaxis, :], self.residual.shape).reshape(-1),
            ),
            "mean_residual_profile": mean_residual_profile.tolist(),
            "mean_abs_residual_profile": mean_abs_residual_profile.tolist(),
            "moneyness_buckets": _summarize_moneyness_buckets(self.residual, self.moneyness),
            "time_buckets": _summarize_time_buckets(self.residual, self.time_to_maturity),
        }
        if self.theta is not None:
            payload["theta"] = self.theta
        if self.alpha is not None:
            payload["alpha"] = self.alpha
        return payload


def evaluate_residual_diagnostics(
    config: RuntimeConfig,
    path_tensor: np.ndarray,
    hedge_component_tensor: np.ndarray,
    benchmark_delta_tensor: np.ndarray,
) -> ResidualDiagnosticsMetrics:
    """Evaluate residual diagnostics for one saved regime."""

    paths = np.asarray(path_tensor, dtype=np.float64)
    hedge_component = np.asarray(hedge_component_tensor, dtype=np.float64)
    benchmark_delta = np.asarray(benchmark_delta_tensor, dtype=np.float64)
    if paths.ndim != 2:
        raise ValueError("path_tensor must have shape [n_paths, n_steps + 1]")
    if hedge_component.shape != benchmark_delta.shape:
        raise ValueError("hedge_component_tensor and benchmark_delta_tensor must have the same shape")
    if hedge_component.ndim != 2 or hedge_component.shape[1] <= 0:
        raise ValueError("hedge_component_tensor must have shape [n_paths, n_steps]")
    if paths.shape[0] != hedge_component.shape[0] or paths.shape[1] != hedge_component.shape[1] + 1:
        raise ValueError("path_tensor must align with hedge_component_tensor on the saved rollout grid")

    residual = hedge_component - benchmark_delta
    moneyness = paths[:, :-1] / float(config.market.strike)
    time_to_maturity = build_time_to_maturity_grid(
        float(config.market.maturity),
        hedge_component.shape[1],
        dtype=np.float64,
    )
    normalized_gap = compute_pathwise_normalized_gap(hedge_component, benchmark_delta)

    return ResidualDiagnosticsMetrics(
        config_hash=config.config_hash,
        risk_kind=config.risk.kind.lower(),
        cost_proportional_rate=float(config.costs.proportional_rate),
        theta=None if config.risk.theta is None else float(config.risk.theta),
        alpha=None if config.risk.alpha is None else float(config.risk.alpha),
        n_paths=int(hedge_component.shape[0]),
        n_steps=int(hedge_component.shape[1]),
        hedge_component=hedge_component,
        benchmark_delta=benchmark_delta,
        residual=residual,
        moneyness=moneyness,
        time_to_maturity=time_to_maturity,
        normalized_gap=normalized_gap,
    )


def _summarize_moneyness_buckets(residual: np.ndarray, moneyness: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        "lt_0p95": _summarize_masked_values(residual, moneyness < 0.95),
        "between_0p95_and_1p05": _summarize_masked_values(residual, (moneyness >= 0.95) & (moneyness <= 1.05)),
        "gt_1p05": _summarize_masked_values(residual, moneyness > 1.05),
    }


def _summarize_time_buckets(residual: np.ndarray, time_to_maturity: np.ndarray) -> dict[str, dict[str, float]]:
    n_steps = residual.shape[1]
    step_indices = np.arange(n_steps)
    step_groups = np.array_split(step_indices, 3)
    labels = ("early", "middle", "late")
    summary: dict[str, dict[str, float]] = {}
    for label, group in zip(labels, step_groups):
        mask = np.zeros_like(residual, dtype=bool)
        if group.size > 0:
            mask[:, group] = True
        bucket_summary = _summarize_masked_values(residual, mask)
        bucket_summary["mean_time_to_maturity"] = float(np.mean(time_to_maturity[group], dtype=np.float64)) if group.size > 0 else 0.0
        summary[label] = bucket_summary
    return summary


def _summarize_masked_values(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    flat_values = np.asarray(values, dtype=np.float64)[mask]
    count = int(flat_values.size)
    if count == 0:
        return {
            "count": 0,
            "fraction": 0.0,
            "mean_residual": 0.0,
            "mean_abs_residual": 0.0,
            "rmse_residual": 0.0,
        }

    total = int(np.asarray(mask, dtype=bool).size)
    return {
        "count": count,
        "fraction": float(count / total),
        "mean_residual": float(np.mean(flat_values, dtype=np.float64)),
        "mean_abs_residual": float(np.mean(np.abs(flat_values), dtype=np.float64)),
        "rmse_residual": float(np.sqrt(np.mean(np.square(flat_values), dtype=np.float64))),
    }


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("correlation inputs must have the same shape")
    if x.size == 0:
        return 0.0
    x_std = float(np.std(x, dtype=np.float64))
    y_std = float(np.std(y, dtype=np.float64))
    if x_std <= 0.0 or y_std <= 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])