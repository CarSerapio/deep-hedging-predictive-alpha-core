"""Evaluation helpers for benchmark diagnostics and saved-artifact analysis."""

from .anti_spurious_controls import build_shuffled_signal_tensor, build_unavailable_signal_control_entry, compute_control_weakening, summarize_hedge_response, summarize_no_liability_control_from_hedge, summarize_saved_no_liability_control, uses_only_benchmark_features
from .benchmark_decomposition import (
    BenchmarkDecompositionMetrics,
    compute_pathwise_normalized_gap,
    evaluate_hedge_tensor_decomposition,
    evaluate_saved_hedge_decomposition,
    summarize_pathwise_metric,
)
from .benchmark_explanations import BenchmarkExplanationMetrics, evaluate_benchmark_explanations
from .decomposition import EmpiricalDecompositionMetrics, evaluate_empirical_decomposition, evaluate_saved_empirical_decomposition
from .holdout_alpha import HoldoutAlphaMetrics, StrategyPerformanceMetrics, evaluate_holdout_alpha, summarize_excess_pnl, summarize_pnl_distribution
from .residual_diagnostics import ResidualDiagnosticsMetrics, evaluate_residual_diagnostics

__all__ = [
    "BenchmarkExplanationMetrics",
    "BenchmarkDecompositionMetrics",
    "EmpiricalDecompositionMetrics",
    "HoldoutAlphaMetrics",
    "ResidualDiagnosticsMetrics",
    "StrategyPerformanceMetrics",
    "build_shuffled_signal_tensor",
    "build_unavailable_signal_control_entry",
    "compute_pathwise_normalized_gap",
    "compute_control_weakening",
    "evaluate_benchmark_explanations",
    "evaluate_empirical_decomposition",
    "evaluate_hedge_tensor_decomposition",
    "evaluate_holdout_alpha",
    "evaluate_residual_diagnostics",
    "evaluate_saved_empirical_decomposition",
    "evaluate_saved_hedge_decomposition",
    "summarize_hedge_response",
    "summarize_no_liability_control_from_hedge",
    "summarize_saved_no_liability_control",
    "summarize_excess_pnl",
    "summarize_pnl_distribution",
    "summarize_pathwise_metric",
    "uses_only_benchmark_features",
]