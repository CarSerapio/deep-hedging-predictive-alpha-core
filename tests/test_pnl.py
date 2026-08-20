"""Validation tests for discrete trading gain and terminal liability PnL.

These tests verify the self-financing accounting used by the benchmark
implementation. They check toy-path arithmetic, the no-hedge sign convention
for a liability writer, tensor-alignment guards, and the benchmark intuition
that a Black-Scholes hedge should reduce dispersion relative to no hedge in a
dense simulated setting.
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

from baselines.black_scholes import black_scholes_price_and_delta_from_config  # noqa: E402
from config import build_runtime_config_from_dict, load_config, load_config_dict  # noqa: E402
from finance.pnl import (  # noqa: E402
    compute_portfolio_pnl_from_config,
    compute_terminal_pnl,
    compute_terminal_pnl_from_config,
    compute_trading_gain,
)
from payoffs import european_call_payoff_from_config  # noqa: E402
from simulators import simulate_gbm_paths  # noqa: E402


class TerminalPnlTests(unittest.TestCase):
    """Checks the benchmark self-financing accounting and seller PnL sign."""

    def test_trading_gain_matches_hand_computation_on_toy_paths(self) -> None:
        paths = np.array(
            [
                [100.0, 102.0, 101.0],
                [100.0, 98.0, 105.0],
            ],
            dtype=np.float32,
        )
        hedges = np.array(
            [
                [1.0, 0.5],
                [0.0, 2.0],
            ],
            dtype=np.float32,
        )

        gain = compute_trading_gain(paths, hedges)

        expected = np.array([1.5, 14.0], dtype=np.float32)
        self.assertTrue(np.allclose(gain, expected))
        self.assertEqual(gain.dtype, np.float32)

    def test_zero_strategy_produces_zero_trading_gain(self) -> None:
        paths = np.array(
            [
                [100.0, 101.0, 99.0],
                [100.0, 97.0, 98.0],
            ],
            dtype=np.float32,
        )
        hedges = np.zeros((2, 2), dtype=np.float32)

        gain = compute_trading_gain(paths, hedges)

        self.assertTrue(np.array_equal(gain, np.zeros(2, dtype=np.float32)))

    def test_unhedged_liability_is_adverse_on_large_up_path(self) -> None:
        paths = np.array(
            [
                [100.0, 140.0],
                [100.0, 95.0],
            ],
            dtype=np.float32,
        )
        hedges = np.zeros((2, 1), dtype=np.float32)

        pnl = compute_terminal_pnl_from_config(load_config(PROJECT_ROOT / "configs" / "base.yaml"), paths, hedges)

        expected = np.array([-40.0, 0.0], dtype=np.float32)
        self.assertTrue(np.array_equal(pnl, expected))

    def test_black_scholes_delta_reduces_pnl_dispersion_against_no_hedge(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
        paths = simulate_gbm_paths(
            s0=config.market.s0,
            mu=0.0,
            sigma=config.market.sigma,
            maturity=config.market.maturity,
            n_steps=200,
            n_paths=4096,
            seed=11,
        )
        _, deltas = black_scholes_price_and_delta_from_config(config, paths)
        payoff = european_call_payoff_from_config(config, paths)

        hedged_pnl = compute_terminal_pnl(paths, deltas, payoff)
        unhedged_pnl = compute_terminal_pnl(paths, np.zeros_like(deltas), payoff)

        # Delta hedging should reduce the dispersion of terminal PnL relative to no hedge, even if the mean is similar.
        # Unhedged PNL = -Z + 0 = -Z, Hedged PNL = -Z + (delta · S)_T. The variance of the hedged PnL should be less than that of the unhedged PnL 
        # as trading gains offset much of the option payoff.
        self.assertLess(float(np.std(hedged_pnl)), float(np.std(unhedged_pnl)))

    def test_pnl_rejects_misaligned_hedge_tensor(self) -> None:
        paths = np.array([[100.0, 101.0, 102.0]], dtype=np.float32)
        hedges = np.array([[0.1]], dtype=np.float32) # This hedge tensor has shape (1, 1) while the path tensor has shape (1, 3). The hedge tensor should have shape (1, 2) to match the number of steps in the path tensor minus one. This test checks that the function raises a ValueError when the hedge tensor is misaligned with the path tensor.
        payoff = np.array([2.0], dtype=np.float32) # This payoff tensor has shape (1,) which is correct for a single path. The test is focused on the misalignment of the hedge tensor, so the payoff tensor is valid.
 
        with self.assertRaises(ValueError):
            compute_terminal_pnl(paths, hedges, payoff)

    def test_cost_aware_liability_pnl_from_config_subtracts_transaction_costs(self) -> None:
        payload = load_config_dict(PROJECT_ROOT / "configs" / "base.yaml")
        payload["costs"]["proportional_rate"] = 0.01
        config = build_runtime_config_from_dict(payload, source_path="cost-liability")

        paths = np.array([[100.0, 105.0, 110.0]], dtype=np.float32)
        hedges = np.array([[1.0, 0.5]], dtype=np.float32)

        pnl = compute_terminal_pnl_from_config(config, paths, hedges)

        # Gain = 1*(105-100) + 0.5*(110-105) = 7.5
        # Payoff = 10, transaction cost = 0.01*(100*1.0 + 105*0.5) = 1.525
        self.assertTrue(np.allclose(pnl, np.array([-4.025], dtype=np.float32), atol=1e-6))

    def test_cost_aware_no_liability_pnl_from_config_subtracts_transaction_costs(self) -> None:
        payload = load_config_dict(PROJECT_ROOT / "configs" / "base.yaml")
        payload["experiment"]["with_liability"] = False
        payload["costs"]["proportional_rate"] = 0.01
        config = build_runtime_config_from_dict(payload, source_path="cost-no-liability")

        paths = np.array([[100.0, 105.0, 110.0]], dtype=np.float32)
        hedges = np.array([[1.0, 0.5]], dtype=np.float32)

        pnl = compute_portfolio_pnl_from_config(config, paths, hedges)

        # Gain = 7.5, transaction cost = 1.525
        self.assertTrue(np.allclose(pnl, np.array([5.975], dtype=np.float32), atol=1e-6))


if __name__ == "__main__":
    unittest.main()