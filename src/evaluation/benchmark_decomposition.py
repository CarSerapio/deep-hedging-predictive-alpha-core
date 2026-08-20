"""Benchmark decomposition metrics for learned hedge tensors.

The benchmark diagnostics compare a learned hedge against the Black-Scholes
delta baseline using Horikawa's normalized pathwise gap metric,

    ||delta^DH - Delta|| / ||Delta||,

and, for entropic runs, against the analytic decomposition benchmark,

    ||delta^DH - Delta - delta^SA|| / ||Delta||.

The norm is the Euclidean norm over hedge times on each path, and pathwise
statistics are summarized across the test split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from baselines.black_scholes import black_scholes_price_and_delta_from_config
from baselines.stat_arb_entropic import entropic_stat_arb_benchmark_from_config
from config import RuntimeConfig, load_config
from simulators.gbm import simulate_gbm_paths_from_config


_DENOMINATOR_FLOOR = 1e-12


@dataclass(frozen=True)
class BenchmarkDecompositionMetrics:
    """Pathwise benchmark-decomposition metrics for one saved hedge."""

    config_hash: str
    risk_kind: str
    n_paths: int
    n_steps: int
    raw_gap: np.ndarray
    adjusted_gap: np.ndarray | None

    def to_summary_dict(self) -> dict[str, object]:
        """Return a JSON-serializable summary payload."""

        payload: dict[str, object] = {
            "config_hash": self.config_hash,
            "risk_kind": self.risk_kind,
            "n_paths": self.n_paths,
            "n_steps": self.n_steps,
            "path_norm": "euclidean_over_time",
            "normalization_reference": "black_scholes_delta",
            "raw_gap": summarize_pathwise_metric(self.raw_gap),
        }
        if self.adjusted_gap is not None:
            payload["adjusted_reference"] = "black_scholes_delta_plus_stat_arb"
            payload["adjusted_gap"] = summarize_pathwise_metric(self.adjusted_gap)
        return payload


def compute_pathwise_normalized_gap(
    hedge_tensor: np.ndarray,
    benchmark_tensor: np.ndarray,
    *,
    denominator_tensor: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Horikawa-style normalized pathwise hedge gaps.

    Args:
        hedge_tensor: Learned or analytic hedge tensor of shape
            ``[n_paths, n_steps]``.
        benchmark_tensor: Reference hedge tensor in the same shape.
        denominator_tensor: Optional normalization tensor. When omitted, the
            benchmark tensor is also used in the denominator.

    Returns:
        One normalized Euclidean gap per path.

    Raises:
        ValueError: If the tensors do not share the same shape contract.
    """

    hedges = np.asarray(hedge_tensor, dtype=np.float64)
    benchmark = np.asarray(benchmark_tensor, dtype=np.float64)
    if hedges.shape != benchmark.shape:
        raise ValueError("hedge_tensor and benchmark_tensor must have the same shape")

    denominator_source = benchmark if denominator_tensor is None else np.asarray(denominator_tensor, dtype=np.float64)
    if denominator_source.shape != benchmark.shape:
        raise ValueError("denominator_tensor must have the same shape as benchmark_tensor")

    numerator = np.linalg.norm(hedges - benchmark, axis=1)
    denominator = np.linalg.norm(denominator_source, axis=1)
    safe_denominator = np.maximum(denominator, _DENOMINATOR_FLOOR)
    return numerator / safe_denominator


def summarize_pathwise_metric(pathwise_values: np.ndarray) -> dict[str, float]:
    """Summarize one pathwise diagnostic vector for JSON reporting."""

    values = np.asarray(pathwise_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("pathwise_values must be a non-empty one-dimensional array")

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def evaluate_hedge_tensor_decomposition(
    config: RuntimeConfig,
    path_tensor: np.ndarray,
    hedge_tensor: np.ndarray,
) -> BenchmarkDecompositionMetrics:
    """Evaluate raw and adjusted benchmark gaps for one hedge tensor.

    The raw metric always compares the supplied hedge against Black-Scholes
    delta. For entropic configs, the adjusted metric additionally subtracts the
    analytic statistical-arbitrage component and therefore compares against the
    theorem-implied benchmark ``Delta + delta^SA`` while keeping the same
    denominator ``||Delta||``.
    """

    _, black_scholes_delta = black_scholes_price_and_delta_from_config(config, path_tensor)
    learned_hedges = np.asarray(hedge_tensor)
    if learned_hedges.shape != black_scholes_delta.shape:
        raise ValueError(
            "hedge_tensor shape does not match the Black-Scholes delta grid for the supplied paths"
        )

    raw_gap = compute_pathwise_normalized_gap(learned_hedges, black_scholes_delta)
    adjusted_gap: np.ndarray | None = None
    if config.risk.kind.lower() == "entropic":
        stat_arb_delta = entropic_stat_arb_benchmark_from_config(config, path_tensor)
        analytic_benchmark = black_scholes_delta + stat_arb_delta
        adjusted_gap = compute_pathwise_normalized_gap(
            learned_hedges,
            analytic_benchmark,
            denominator_tensor=black_scholes_delta,
        )

    return BenchmarkDecompositionMetrics(
        config_hash=config.config_hash,
        risk_kind=config.risk.kind.lower(),
        n_paths=int(learned_hedges.shape[0]),
        n_steps=int(learned_hedges.shape[1]),
        raw_gap=raw_gap,
        adjusted_gap=adjusted_gap,
    )


def evaluate_saved_hedge_decomposition(
    config_path: Path,
    hedges_path: Path,
) -> BenchmarkDecompositionMetrics:
    """Load a saved hedge tensor and evaluate the benchmark diagnostics."""

    config = load_config(config_path)
    test_paths = simulate_gbm_paths_from_config(config, split="test", seed=config.paths.seed + 2)
    learned_hedges = np.load(hedges_path)
    return evaluate_hedge_tensor_decomposition(config, test_paths, learned_hedges)