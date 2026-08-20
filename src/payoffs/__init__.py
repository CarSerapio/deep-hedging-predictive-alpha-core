"""Liability payoff entry points for the benchmark pipeline.

The payoff layer transforms simulated price paths into terminal derivative
cashflows that the hedge is meant to offset. The current benchmark supports
only the single-asset European call used throughout the reported experiments.
"""

from .european_call import compute_payoff, european_call_payoff, european_call_payoff_from_config

__all__ = ["compute_payoff", "european_call_payoff", "european_call_payoff_from_config"]