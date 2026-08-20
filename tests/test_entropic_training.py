"""Validation tests for the entropic training loop."""

from __future__ import annotations

from dataclasses import replace
import json
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
from finance.pnl import compute_portfolio_pnl_from_config, compute_terminal_pnl_from_config  # noqa: E402
from policies import build_mlp_policy_from_config, is_tensorflow_available, rollout_policy_from_config  # noqa: E402
from training import train_entropic_deep_hedger, train_entropic_variants  # noqa: E402


class EntropicTrainingTests(unittest.TestCase):
    """Checks that entropic training emits coherent artifacts and checkpoints."""

    def test_single_run_writes_checkpoint_history_and_reproducible_test_hedges(self) -> None:
        if not is_tensorflow_available():
            self.skipTest("TensorFlow is unavailable in the current Python environment.")

        config = _small_entropic_config()
        with TemporaryDirectory() as tmpdir:
            result = train_entropic_deep_hedger(config, tmpdir, deterministic=True)

            self.assertEqual(result.test_hedges.shape, (config.paths.test_paths, config.market.n_steps))
            self.assertTrue(Path(result.checkpoint_path).exists())
            self.assertTrue(Path(result.history_path).exists())
            self.assertTrue(Path(result.test_hedges_path).exists())
            self.assertGreaterEqual(result.best_epoch, 1)
            self.assertLessEqual(result.best_epoch, result.epochs_ran)
            self.assertTrue(np.isfinite(result.test_risk))

            restored_policy = build_mlp_policy_from_config(config, seed=config.paths.seed)
            restored_policy.load_weights(result.checkpoint_path)
            restored_hedges = rollout_policy_from_config(config, result.test_paths, restored_policy).numpy()
            np.testing.assert_allclose(restored_hedges, result.test_hedges, atol=1e-5)

            history = json.loads(Path(result.history_path).read_text(encoding="utf-8"))
            self.assertEqual(history["theta"], 1.0)
            self.assertEqual(history["best_epoch"], result.best_epoch)
            self.assertEqual(history["epochs_ran"], result.epochs_ran)
            self.assertEqual(result.checkpoint_metric, "val_loss")
            self.assertEqual(history["checkpoint_metric"], "val_loss")
            self.assertEqual(len(result.val_adjusted_gap_means), result.epochs_ran)
            self.assertEqual(len(history["val_adjusted_gap_means"]), result.epochs_ran)
            self.assertTrue(np.isfinite(result.initial_val_adjusted_gap_mean))
            self.assertTrue(np.isfinite(result.best_val_adjusted_gap_mean))

    def test_adjusted_gap_checkpointing_tracks_validation_gap_history(self) -> None:
        if not is_tensorflow_available():
            self.skipTest("TensorFlow is unavailable in the current Python environment.")

        config = _small_entropic_config(epochs=3)
        with TemporaryDirectory() as tmpdir:
            result = train_entropic_deep_hedger(
                config,
                tmpdir,
                deterministic=True,
                checkpoint_metric="adjusted_gap",
            )

            self.assertEqual(result.checkpoint_metric, "adjusted_gap")
            self.assertEqual(len(result.val_adjusted_gap_means), result.epochs_ran)
            self.assertAlmostEqual(result.best_checkpoint_score, min(result.val_adjusted_gap_means), places=7)
            self.assertAlmostEqual(result.checkpoint_val_adjusted_gap_mean, result.best_checkpoint_score, places=7)

            history = json.loads(Path(result.history_path).read_text(encoding="utf-8"))
            self.assertEqual(history["checkpoint_metric"], "adjusted_gap")
            self.assertAlmostEqual(history["best_checkpoint_score"], min(history["val_adjusted_gap_means"]), places=7)
            self.assertAlmostEqual(history["checkpoint_val_adjusted_gap_mean"], history["best_checkpoint_score"], places=7)

    def test_variant_wrapper_returns_one_result_per_theta(self) -> None:
        if not is_tensorflow_available():
            self.skipTest("TensorFlow is unavailable in the current Python environment.")

        config = _small_entropic_config(epochs=2)
        with TemporaryDirectory() as tmpdir:
            results = train_entropic_variants(config, tmpdir, theta_values=(1.0, 5.0), deterministic=True)

            self.assertEqual(len(results), 2)
            self.assertEqual([result.theta for result in results], [1.0, 5.0])
            self.assertNotEqual(results[0].artifact_dir, results[1].artifact_dir)

    def test_no_liability_run_uses_portfolio_pnl_and_records_friction_metadata(self) -> None:
        if not is_tensorflow_available():
            self.skipTest("TensorFlow is unavailable in the current Python environment.")

        config = _small_no_liability_entropic_config()
        with TemporaryDirectory() as tmpdir:
            result = train_entropic_deep_hedger(config, tmpdir, deterministic=True)

            expected_test_pnl = compute_portfolio_pnl_from_config(config, result.test_paths, result.test_hedges)
            np.testing.assert_allclose(result.test_pnl, expected_test_pnl, atol=1e-6)

            history = json.loads(Path(result.history_path).read_text(encoding="utf-8"))
            self.assertFalse(history["with_liability"])
            self.assertEqual(history["experiment_name"], config.experiment.name)
            self.assertAlmostEqual(history["cost_proportional_rate"], config.costs.proportional_rate, places=12)

    def test_with_liability_run_uses_terminal_pnl_and_records_friction_metadata(self) -> None:
        if not is_tensorflow_available():
            self.skipTest("TensorFlow is unavailable in the current Python environment.")

        config = _small_with_liability_entropic_config()
        with TemporaryDirectory() as tmpdir:
            result = train_entropic_deep_hedger(config, tmpdir, deterministic=True)

            expected_test_pnl = compute_terminal_pnl_from_config(config, result.test_paths, result.test_hedges)
            np.testing.assert_allclose(result.test_pnl, expected_test_pnl, atol=1e-6)

            history = json.loads(Path(result.history_path).read_text(encoding="utf-8"))
            self.assertTrue(history["with_liability"])
            self.assertEqual(history["experiment_name"], config.experiment.name)
            self.assertAlmostEqual(history["cost_proportional_rate"], config.costs.proportional_rate, places=12)


def _small_entropic_config(*, epochs: int = 4):
    """Build a tiny but valid entropic config for focused training tests."""

    config = load_config(PROJECT_ROOT / "configs" / "benchmark_entropic_unit_spot.yaml")
    return replace(
        config,
        market=replace(config.market, mu=0.10),
        paths=replace(config.paths, train_paths=96, val_paths=48, test_paths=48, seed=19),
        training=replace(
            config.training,
            batch_size=24,
            epochs=epochs,
            learning_rate=0.005,
            gradient_clip=1.0,
            patience=1,
        ),
        risk=replace(config.risk, theta=1.0),
    )


def _small_no_liability_entropic_config(*, epochs: int = 4):
    """Build a tiny no-liability config for focused training tests."""

    config = load_config(PROJECT_ROOT / "configs" / "entropic_no_liability_unit_spot.yaml")
    return replace(
        config,
        market=replace(config.market, mu=0.10),
        paths=replace(config.paths, train_paths=96, val_paths=48, test_paths=48, seed=23),
        training=replace(
            config.training,
            batch_size=24,
            epochs=epochs,
            learning_rate=0.005,
            gradient_clip=1.0,
            patience=1,
        ),
        risk=replace(config.risk, theta=1.0),
    )


def _small_with_liability_entropic_config(*, epochs: int = 4):
    """Build a tiny with-liability config for focused training tests."""

    config = load_config(PROJECT_ROOT / "configs" / "entropic_with_liability_unit_spot.yaml")
    return replace(
        config,
        market=replace(config.market, mu=0.10),
        paths=replace(config.paths, train_paths=96, val_paths=48, test_paths=48, seed=29),
        training=replace(
            config.training,
            batch_size=24,
            epochs=epochs,
            learning_rate=0.005,
            gradient_clip=1.0,
            patience=1,
        ),
        risk=replace(config.risk, theta=1.0),
    )


if __name__ == "__main__":
    unittest.main()