"""Validation tests for the scalar risk objectives.

These tests verify the scalar batch objectives used for later deep-hedging
training. They focus on cash translation, increasing conservativeness for
entropic risk, and left-tail sensitivity for CVaR.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import load_config  # noqa: E402
from risk import compute_risk_objective_from_config, cvar_risk, entropic_risk  # noqa: E402


class RiskObjectiveTests(unittest.TestCase):
    """Checks the current scalar risk objectives on simulated PnL batches."""

    def test_entropic_risk_is_cash_translation_invariant(self) -> None:
        pnl = np.array([-2.0, -0.5, 0.5, 1.0, 3.0], dtype=np.float64)
        cash = 1.25

        base = entropic_risk(pnl, theta=0.8)
        shifted = entropic_risk(pnl + cash, theta=0.8)

        self.assertAlmostEqual(shifted, base - cash, places=10)

    def test_entropic_risk_becomes_more_conservative_as_theta_increases(self) -> None:
        pnl = np.array([-4.0, -1.5, 0.0, 1.0, 3.5], dtype=np.float64)

        mild = entropic_risk(pnl, theta=0.5)
        severe = entropic_risk(pnl, theta=2.0)

        self.assertGreaterEqual(severe, mild)

    def test_cvar_risk_is_cash_translation_invariant(self) -> None:
        pnl = np.array([-3.0, -1.0, 0.5, 1.5, 2.0], dtype=np.float64)
        cash = 0.75

        base = cvar_risk(pnl, alpha=0.25)
        shifted = cvar_risk(pnl + cash, alpha=0.25)

        self.assertAlmostEqual(shifted, base - cash, places=6)

    def test_cvar_risk_focuses_on_left_tail_deterioration(self) -> None:
        base_pnl = np.array([-1.5, -0.8, 0.1, 0.5, 1.0, 1.2], dtype=np.float64)
        perturbed_pnl = np.array([-4.0, -0.8, 0.1, 0.5, 1.0, 1.2], dtype=np.float64)

        base_risk = cvar_risk(base_pnl, alpha=0.2)
        perturbed_risk = cvar_risk(perturbed_pnl, alpha=0.2)

        self.assertGreater(perturbed_risk, base_risk)

    def test_config_wrapper_dispatches_entropic_and_cvar(self) -> None:
        pnl = np.array([-2.5, -0.2, 0.3, 1.1], dtype=np.float64)

        entropic_config = load_config(PROJECT_ROOT / "configs" / "benchmark_entropic.yaml")
        cvar_config = load_config(PROJECT_ROOT / "configs" / "benchmark_cvar.yaml")

        entropic_value = compute_risk_objective_from_config(entropic_config, pnl)
        cvar_value = compute_risk_objective_from_config(cvar_config, pnl)

        self.assertAlmostEqual(entropic_value, entropic_risk(pnl, theta=entropic_config.risk.theta), places=10)
        self.assertAlmostEqual(cvar_value, cvar_risk(pnl, alpha=cvar_config.risk.alpha), places=10)


if __name__ == "__main__":
    unittest.main()