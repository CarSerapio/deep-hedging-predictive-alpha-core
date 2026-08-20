"""Reproducibility helpers for stochastic simulation and future model training.

This module belongs to the experiment-control stage of the research pipeline.
It centralizes seed handling for Python's standard RNG, NumPy, and TensorFlow
when TensorFlow is installed, so Monte Carlo simulations and future training
runs can be reproduced more reliably from one entry point.

Important limitation:
- Setting the same seed improves repeatability but does not guarantee full
    cross-hardware determinism, especially once TensorFlow kernels enter the
    pipeline.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed the main random number generators used by the project.

    Args:
        seed: Integer seed applied to Python hashing, ``random``, NumPy, and
            TensorFlow when available.
        deterministic: Whether to request deterministic TensorFlow operators
            when the installed version exposes that control.

    Side Effects:
        Mutates process-wide RNG state and sets ``PYTHONHASHSEED`` in the
        current process environment.

    Notes:
        TensorFlow seeding is optional because the current baseline modules do
        not require TensorFlow yet. The function returns silently when that
        dependency is absent.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed) # Seed the built-in Python random number generator for reproducibility.
    np.random.seed(seed) # Seed the NumPy random number generator for reproducibility.

    try:
        import tensorflow as tf  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return

    tf.random.set_seed(seed) # Seed the TensorFlow random number generator for reproducibility.
    if deterministic:
        # TensorFlow determinism is version-dependent, so the capability is
        # checked dynamically instead of assuming a specific API surface.
        enable_determinism = getattr(tf.config.experimental, "enable_op_determinism", None) # Check if the TensorFlow version supports enabling deterministic operations.
        if callable(enable_determinism):
            enable_determinism() # Enable deterministic TensorFlow operations for reproducibility.
