"""Focused tests for the anti-spurious-alpha helpers."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import load_config  # noqa: E402
from evaluation.anti_spurious_controls import (  # noqa: E402
    build_unavailable_signal_control_entry,
    compute_control_weakening,
    summarize_saved_no_liability_control,
    uses_only_benchmark_features,
)


class AntiSpuriousControlsTests(unittest.TestCase):
    """Check the non-training helper logic used by the control study."""

    def test_unavailable_signal_control_is_marked_not_applicable(self) -> None:
        entry = build_unavailable_signal_control_entry(
            control_name="zero_predictive_signal",
            feature_names=["spot", "time_to_maturity", "previous_hedge"],
        )

        self.assertEqual(entry["status"], "not_applicable")
        self.assertIn("No predictive signal inputs exist", entry["reason"])

    def test_control_weakening_reports_signed_differences(self) -> None:
        benchmark_only = {
            "candidate": {"pnl_mean": 0.002, "mean_abs_hedge": 0.4, "pnl_std": 0.02},
            "benchmark_adjusted": {"vs_passive": {"mean_excess_pnl": 0.002}},
        }
        zero_drift = {
            "candidate": {"pnl_mean": -0.001, "mean_abs_hedge": 0.3, "pnl_std": 0.01},
            "benchmark_adjusted": {"vs_passive": {"mean_excess_pnl": -0.001}},
        }

        weakening = compute_control_weakening(benchmark_only, zero_drift)

        self.assertAlmostEqual(weakening["candidate_pnl_mean_change"], -0.003)
        self.assertAlmostEqual(weakening["candidate_mean_abs_hedge_change"], -0.1)
        self.assertAlmostEqual(weakening["vs_passive_mean_excess_change"], -0.003)

    def test_selected_config_uses_only_benchmark_features(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "entropic_no_liability_unit_spot_cost_0p0025.yaml")

        self.assertTrue(uses_only_benchmark_features(config))

    def test_saved_control_summary_includes_vs_passive_entry(self) -> None:
        config_path = PROJECT_ROOT / "configs" / "entropic_no_liability_unit_spot_zero_drift_cost_0p0025.yaml"
        config = load_config(config_path)

        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            hedge_tensor = np.zeros((config.paths.test_paths, config.market.n_steps), dtype=np.float64)
            np.save(run_dir / "test_hedges.npy", hedge_tensor)

            summary = summarize_saved_no_liability_control(
                config_path=config_path,
                run_dir=run_dir,
                reference_sigma=0.2,
                control_name="zero_drift",
                note="test",
            )

            self.assertIn("benchmark_adjusted", summary)
            self.assertIn("vs_passive", summary["benchmark_adjusted"])
            self.assertAlmostEqual(
                summary["benchmark_adjusted"]["vs_passive"]["mean_excess_pnl"],
                summary["candidate"]["pnl_mean"],
            )


if __name__ == "__main__":
    unittest.main()