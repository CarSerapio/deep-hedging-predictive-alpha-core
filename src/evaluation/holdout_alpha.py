"""Holdout evaluation helpers for the candidate no-liability alpha policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import RuntimeConfig
from finance.costs import compute_proportional_transaction_cost_from_config
from finance.pnl import compute_portfolio_pnl_from_config


@dataclass(frozen=True)
class StrategyPerformanceMetrics:
    """Pathwise no-liability performance metrics for one hedge policy."""

    label: str
    pnl: np.ndarray
    hedge: np.ndarray
    transaction_cost: np.ndarray

    def to_summary_dict(self) -> dict[str, object]:
        """Return a JSON-serializable summary payload."""

        trades = compute_trade_tensor(self.hedge)
        return {
            "label": self.label,
            "n_paths": int(self.hedge.shape[0]),
            "n_steps": int(self.hedge.shape[1]),
            "mean_abs_hedge": float(np.mean(np.abs(self.hedge), dtype=np.float64)),
            "mean_abs_trade": float(np.mean(np.abs(trades), dtype=np.float64)),
            "mean_pathwise_turnover": float(np.mean(np.sum(np.abs(trades), axis=1), dtype=np.float64)),
            "mean_transaction_cost": float(np.mean(self.transaction_cost, dtype=np.float64)),
            **summarize_pnl_distribution(self.pnl),
        }


@dataclass(frozen=True)
class HoldoutAlphaMetrics:
    """Candidate-alpha performance on one holdout regime and simple controls."""

    regime_label: str
    candidate: StrategyPerformanceMetrics
    passive: StrategyPerformanceMetrics
    long_only: StrategyPerformanceMetrics
    excess_vs_passive: np.ndarray
    excess_vs_long_only: np.ndarray

    def to_summary_dict(self) -> dict[str, object]:
        """Return a compact JSON-serializable regime summary."""

        return {
            "regime_label": self.regime_label,
            "candidate": self.candidate.to_summary_dict(),
            "controls": {
                "passive": self.passive.to_summary_dict(),
                "long_only": self.long_only.to_summary_dict(),
            },
            "benchmark_adjusted": {
                "vs_passive": summarize_excess_pnl(self.excess_vs_passive),
                "vs_long_only": summarize_excess_pnl(self.excess_vs_long_only),
            },
        }


def evaluate_holdout_alpha(
    config: RuntimeConfig,
    path_tensor: np.ndarray,
    candidate_hedge_tensor: np.ndarray,
    *,
    long_only_exposure: float = 1.0,
) -> HoldoutAlphaMetrics:
    """Evaluate a no-liability candidate policy against simple holdout controls."""

    candidate_hedge = _validate_hedge_tensor(candidate_hedge_tensor)
    passive_hedge = np.zeros_like(candidate_hedge)
    long_only_hedge = np.full(candidate_hedge.shape, float(long_only_exposure), dtype=np.float64)

    candidate = evaluate_strategy_performance(config, path_tensor, candidate_hedge, label="candidate_alpha")
    passive = evaluate_strategy_performance(config, path_tensor, passive_hedge, label="passive_cash")
    long_only = evaluate_strategy_performance(config, path_tensor, long_only_hedge, label="long_only")

    return HoldoutAlphaMetrics(
        regime_label=config.experiment.regime_label,
        candidate=candidate,
        passive=passive,
        long_only=long_only,
        excess_vs_passive=np.asarray(candidate.pnl - passive.pnl, dtype=np.float64),
        excess_vs_long_only=np.asarray(candidate.pnl - long_only.pnl, dtype=np.float64),
    )


def evaluate_strategy_performance(
    config: RuntimeConfig,
    path_tensor: np.ndarray,
    hedge_tensor: np.ndarray,
    *,
    label: str,
) -> StrategyPerformanceMetrics:
    """Compute pathwise no-liability performance metrics for one hedge tensor."""

    hedge = _validate_hedge_tensor(hedge_tensor)
    pnl = np.asarray(compute_portfolio_pnl_from_config(config, path_tensor, hedge), dtype=np.float64)
    transaction_cost = np.asarray(
        compute_proportional_transaction_cost_from_config(config, path_tensor, hedge),
        dtype=np.float64,
    )
    return StrategyPerformanceMetrics(
        label=label,
        pnl=pnl,
        hedge=hedge,
        transaction_cost=transaction_cost,
    )


def summarize_pnl_distribution(pnl: np.ndarray) -> dict[str, float]:
    """Summarize a terminal PnL distribution with holdout risk statistics."""

    pnl_array = _validate_vector(pnl, name="pnl")
    downside = np.minimum(pnl_array, 0.0)
    pnl_std = float(np.std(pnl_array, dtype=np.float64))
    downside_std = float(np.sqrt(np.mean(np.square(downside), dtype=np.float64)))
    quantiles = np.quantile(pnl_array, [0.01, 0.05, 0.5, 0.95, 0.99])
    p05 = float(quantiles[1])
    tail_slice = pnl_array[pnl_array <= p05]
    expected_shortfall_5 = p05 if tail_slice.size == 0 else float(np.mean(tail_slice, dtype=np.float64))
    mean_value = float(np.mean(pnl_array, dtype=np.float64))

    return {
        "pnl_mean": mean_value,
        "pnl_std": pnl_std,
        "sharpe": _safe_ratio(mean_value, pnl_std),
        "sortino": _safe_ratio(mean_value, downside_std),
        "pnl_min": float(np.min(pnl_array)),
        "pnl_p01": float(quantiles[0]),
        "pnl_p05": p05,
        "pnl_median": float(quantiles[2]),
        "pnl_p95": float(quantiles[3]),
        "pnl_p99": float(quantiles[4]),
        "pnl_max": float(np.max(pnl_array)),
        "expected_shortfall_5": expected_shortfall_5,
        "positive_rate": float(np.mean(pnl_array > 0.0, dtype=np.float64)),
        "loss_rate": float(np.mean(pnl_array < 0.0, dtype=np.float64)),
    }


def summarize_excess_pnl(excess_pnl: np.ndarray) -> dict[str, float]:
    """Summarize benchmark-adjusted holdout PnL against one simple control."""

    excess_array = _validate_vector(excess_pnl, name="excess_pnl")
    summary = summarize_pnl_distribution(excess_array)
    return {
        "mean_excess_pnl": float(summary["pnl_mean"]),
        "std_excess_pnl": float(summary["pnl_std"]),
        "excess_sharpe": float(summary["sharpe"]),
        "excess_sortino": float(summary["sortino"]),
        "excess_p05": float(summary["pnl_p05"]),
        "excess_expected_shortfall_5": float(summary["expected_shortfall_5"]),
        "positive_excess_rate": float(summary["positive_rate"]),
    }


def compute_trade_tensor(hedge_tensor: np.ndarray) -> np.ndarray:
    """Convert hedge positions into per-step trades charged by proportional costs."""

    hedges = _validate_hedge_tensor(hedge_tensor)
    trades = np.empty_like(hedges, dtype=np.float64)
    trades[:, 0] = hedges[:, 0]
    if hedges.shape[1] > 1:
        trades[:, 1:] = hedges[:, 1:] - hedges[:, :-1]
    return trades


def _validate_hedge_tensor(hedge_tensor: np.ndarray) -> np.ndarray:
    hedges = np.asarray(hedge_tensor, dtype=np.float64)
    if hedges.ndim != 2 or hedges.shape[1] <= 0:
        raise ValueError("hedge_tensor must have shape [n_paths, n_steps]")
    return hedges


def _validate_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.shape[0] <= 0:
        raise ValueError(f"{name} must have shape [n_paths]")
    return vector


def _safe_ratio(numerator: float, denominator: float, *, atol: float = 1e-12) -> float:
    if abs(denominator) <= atol:
        return 0.0
    return float(numerator / denominator)