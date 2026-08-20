"""Policy-network entry points for learned hedging modules.

This package contains trainable policy models that map hedge-time features to
the next trading position. The current baseline exposes a feed-forward
TensorFlow MLP matching the benchmark methodology.
"""

from .mlp_policy import MLPPolicy, build_mlp_policy, build_mlp_policy_from_config, is_tensorflow_available
from .rollout import build_hedge_features, rollout_policy, rollout_policy_from_config

__all__ = [
    "MLPPolicy",
    "build_hedge_features",
    "build_mlp_policy",
    "build_mlp_policy_from_config",
    "is_tensorflow_available",
    "rollout_policy",
    "rollout_policy_from_config",
]