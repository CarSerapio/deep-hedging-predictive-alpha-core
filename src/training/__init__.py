"""Training entry points for learned deep-hedging experiments.

This package implements entropic-risk and CVaR training loops that turn the
benchmark policy and rollout modules into trained deep hedgers with saved
checkpoints and evaluation artifacts.
"""

from .cvar import CVaRTrainingResult, train_cvar_deep_hedger, train_cvar_variants
from .entropic import EntropicTrainingResult, train_entropic_deep_hedger, train_entropic_variants

__all__ = [
    "CVaRTrainingResult",
    "EntropicTrainingResult",
    "train_cvar_deep_hedger",
    "train_cvar_variants",
    "train_entropic_deep_hedger",
    "train_entropic_variants",
]