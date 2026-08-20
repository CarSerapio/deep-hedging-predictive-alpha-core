"""Validation tests for the CVaR training loop."""

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
from policies import build_mlp_policy_from_config, is_tensorflow_available, rollout_policy_from_config  # noqa: E402
from training import train_cvar_deep_hedger, train_cvar_variants  # noqa: E402


class CVaRTrainingTests(unittest.TestCase):
    """Checks that CVaR training emits coherent artifacts and checkpoints."""

    def test_single_run_writes_checkpoint_history_and_reproducible_test_hedges(self) -> None:
        if not is_tensorflow_available():
            self.skipTest("TensorFlow is unavailable in the current Python environment.")

        config = _small_cvar_config()
        with TemporaryDirectory() as tmpdir:
            result = train_cvar_deep_hedger(config, tmpdir, deterministic=True)

            self.assertEqual(result.test_hedges.shape, (config.paths.test_paths, config.market.n_steps))
            self.assertTrue(Path(result.checkpoint_path).exists())
            self.assertTrue(Path(result.history_path).exists())
            self.assertTrue(Path(result.test_hedges_path).exists())
            self.assertGreaterEqual(result.best_epoch, 1)
            self.assertLessEqual(result.best_epoch, result.epochs_ran)
            self.assertTrue(np.isfinite(result.test_risk))
            self.assertTrue(np.isfinite(result.best_eta))

            restored_policy = build_mlp_policy_from_config(config, seed=config.paths.seed)
            restored_policy.load_weights(result.checkpoint_path)
            restored_hedges = rollout_policy_from_config(config, result.test_paths, restored_policy).numpy()
            np.testing.assert_allclose(restored_hedges, result.test_hedges, atol=1e-5)

            history = json.loads(Path(result.history_path).read_text(encoding="utf-8"))
            self.assertEqual(history["alpha"], 0.5)
            self.assertEqual(history["best_epoch"], result.best_epoch)
            self.assertEqual(history["epochs_ran"], result.epochs_ran)

    def test_variant_wrapper_returns_one_result_per_alpha(self) -> None:
        if not is_tensorflow_available():
            self.skipTest("TensorFlow is unavailable in the current Python environment.")

        config = _small_cvar_config(epochs=2)
        with TemporaryDirectory() as tmpdir:
            results = train_cvar_variants(config, tmpdir, alpha_values=(0.25, 0.5), deterministic=True)

            self.assertEqual(len(results), 2)
            self.assertEqual([result.alpha for result in results], [0.25, 0.5])
            self.assertNotEqual(results[0].artifact_dir, results[1].artifact_dir)

    def test_cvar_updates_auxiliary_eta_and_records_metric_history(self) -> None:
        if not is_tensorflow_available():
            self.skipTest("TensorFlow is unavailable in the current Python environment.")

        cvar_config = _small_cvar_config(epochs=3)
        with TemporaryDirectory() as tmpdir:
            cvar_result = train_cvar_deep_hedger(cvar_config, Path(tmpdir) / "cvar", deterministic=True)

            self.assertEqual(len(cvar_result.eta_history), cvar_result.epochs_ran)
            self.assertTrue(all(np.isfinite(value) for value in cvar_result.eta_history))
            self.assertTrue(np.isfinite(cvar_result.best_eta))
            self.assertTrue(np.isfinite(cvar_result.mean_abs_delta_gap))
            self.assertNotAlmostEqual(cvar_result.eta_history[-1], 0.0, places=6)


def _small_cvar_config(*, epochs: int = 4):
    """Build a tiny but valid CVaR config for focused training tests."""

    config = load_config(PROJECT_ROOT / "configs" / "benchmark_cvar.yaml")
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
        risk=replace(config.risk, alpha=0.5),
    )


if __name__ == "__main__":
    unittest.main()