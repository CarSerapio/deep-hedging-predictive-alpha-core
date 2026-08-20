"""High-level experiment workflows built on top of the core package."""

from .anti_spurious_controls import build_anti_spurious_control_report
from .benchmark_explanations import build_benchmark_explanation_report
from .holdout_evaluation import build_holdout_evaluation_report
from .holdout_regimes import build_holdout_regime_report
from .predictive_signal_controls import build_predictive_signal_controls_report
from .transaction_cost_sweep import run_transaction_cost_sweep

__all__ = [
    "build_anti_spurious_control_report",
    "build_benchmark_explanation_report",
    "build_holdout_evaluation_report",
    "build_holdout_regime_report",
    "build_predictive_signal_controls_report",
    "run_transaction_cost_sweep",
]