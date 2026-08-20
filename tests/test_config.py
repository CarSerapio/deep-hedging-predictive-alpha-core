"""Validation tests for typed configuration loading and benchmark assumptions.

These tests form the first reproducibility checkpoint of the research pipeline.
They confirm that config inheritance works, that the baseline benchmark values
load correctly, and that invalid experimental splits are rejected before any
simulation or training code runs.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import ConfigError, build_runtime_config_from_dict, load_config, load_config_dict  # noqa: E402


class ConfigTests(unittest.TestCase):
    """Regression tests for the configuration layer used by all later modules."""

    def test_load_base_config(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "base.yaml")

        self.assertEqual(config.experiment.name, "base")
        self.assertEqual(config.market.n_steps, 20)
        self.assertEqual(config.model.hidden_layers, 4)
        self.assertEqual(config.model.feature_names, ["spot", "time_to_maturity", "previous_hedge"])
        self.assertEqual(config.costs.proportional_rate, 0.0)
        self.assertEqual(config.risk.kind, "entropic")
        self.assertEqual(config.splits.train, 0.70)
        self.assertTrue(config.config_hash)

    def test_extends_merges_nested_values(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "benchmark_cvar.yaml")

        self.assertEqual(config.experiment.name, "benchmark_cvar")
        self.assertEqual(config.market.s0, 100.0)
        self.assertEqual(config.risk.kind, "cvar")
        self.assertEqual(config.risk.alpha, 0.5)
        self.assertIsNone(config.risk.theta)

    def test_invalid_split_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.yaml"
            payload = load_config_dict(PROJECT_ROOT / "configs" / "base.yaml")
            payload["splits"] = {"train": 0.8, "validation": 0.2, "test": 0.2}
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(ConfigError):
                load_config(path)

    def test_build_runtime_config_from_dict_recomputes_hash_for_modified_payload(self) -> None:
        payload = load_config_dict(PROJECT_ROOT / "configs" / "benchmark_entropic_unit_spot.yaml")
        baseline = build_runtime_config_from_dict(payload, source_path="baseline")

        modified_payload = json.loads(json.dumps(payload))
        modified_payload["market"]["mu"] = 0.10
        modified = build_runtime_config_from_dict(modified_payload, source_path="modified")

        self.assertEqual(baseline.market.mu, 0.05)
        self.assertEqual(modified.market.mu, 0.10)
        self.assertNotEqual(baseline.config_hash, modified.config_hash)

    def test_negative_cost_rate_raises(self) -> None:
        payload = load_config_dict(PROJECT_ROOT / "configs" / "base.yaml")
        payload["costs"]["proportional_rate"] = -0.001

        with self.assertRaises(ConfigError):
            build_runtime_config_from_dict(payload, source_path="negative-cost")

if __name__ == "__main__":
    unittest.main()