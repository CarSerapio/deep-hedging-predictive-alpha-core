"""Build holdout-regime definitions from clean config inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
REPO_ROOT = SRC_ROOT.parent

from config import RuntimeConfig, load_config
DEFAULT_HOLDOUT_CONFIGS = (
    REPO_ROOT / "configs" / "holdout_no_liability_cost_0p004.yaml",
    REPO_ROOT / "configs" / "holdout_no_liability_low_drift_high_vol.yaml",
    REPO_ROOT / "configs" / "holdout_no_liability_high_drift_low_vol.yaml",
)


def build_holdout_regime_report(
    *,
    in_sample_config_path: Path,
    holdout_config_paths: list[Path],
) -> dict[str, Any]:
    """Build a holdout-regime summary report."""

    if len(holdout_config_paths) == 0:
        raise ValueError("holdout_config_paths must contain at least one holdout config")

    in_sample_config = load_config(in_sample_config_path)
    holdout_regimes = [
        summarize_holdout_regime(in_sample_config, load_config(path), config_path=path)
        for path in holdout_config_paths
    ]

    covered_dimensions = sorted(
        {
            dimension
            for regime in holdout_regimes
            for dimension in regime["changed_dimensions"]
            if dimension != "seed"
        }
    )
    verification = {
        "all_holdouts_materially_different": all(bool(regime["materially_different_from_in_sample"]) for regime in holdout_regimes),
        "no_split_id_leakage": all(len(regime["overlapping_split_ids"]) == 0 for regime in holdout_regimes),
        "holdout_names_are_unique": len({str(regime["regime_label"]) for regime in holdout_regimes}) == len(holdout_regimes),
        "covers_required_dimensions": len(covered_dimensions) > 0,
    }

    return {
        "report_type": "holdout_regimes",
        "verification_passed": all(bool(value) for value in verification.values()),
        "verification": verification,
        "in_sample_config": str(in_sample_config_path.expanduser().resolve()),
        "in_sample_regime_label": in_sample_config.experiment.regime_label,
        "covered_dimensions": covered_dimensions,
        "holdout_regimes": holdout_regimes,
    }


def summarize_holdout_regime(
    in_sample_config: RuntimeConfig,
    holdout_config: RuntimeConfig,
    *,
    config_path: Path,
) -> dict[str, Any]:
    """Summarize one holdout config against the selected in-sample baseline."""

    changed_dimensions = _compute_changed_dimensions(in_sample_config, holdout_config)
    in_sample_split_ids = build_split_ids(in_sample_config)
    holdout_split_ids = build_split_ids(holdout_config)
    overlapping_split_ids = sorted(set(in_sample_split_ids.values()) & set(holdout_split_ids.values()))

    return {
        "config_path": str(config_path.expanduser().resolve()),
        "experiment_name": holdout_config.experiment.name,
        "regime_label": holdout_config.experiment.regime_label,
        "with_liability": bool(holdout_config.experiment.with_liability),
        "cost_proportional_rate": float(holdout_config.costs.proportional_rate),
        "market": {
            "s0": float(holdout_config.market.s0),
            "mu": float(holdout_config.market.mu),
            "sigma": float(holdout_config.market.sigma),
            "maturity": float(holdout_config.market.maturity),
            "dt": float(holdout_config.market.dt),
            "n_steps": int(holdout_config.market.n_steps),
            "strike": float(holdout_config.market.strike),
        },
        "paths_seed": int(holdout_config.paths.seed),
        "changed_dimensions": changed_dimensions,
        "materially_different_from_in_sample": any(dimension != "seed" for dimension in changed_dimensions),
        "delta_from_in_sample": {
            "cost_proportional_rate": float(holdout_config.costs.proportional_rate - in_sample_config.costs.proportional_rate),
            "mu": float(holdout_config.market.mu - in_sample_config.market.mu),
            "sigma": float(holdout_config.market.sigma - in_sample_config.market.sigma),
            "dt": float(holdout_config.market.dt - in_sample_config.market.dt),
            "n_steps": int(holdout_config.market.n_steps - in_sample_config.market.n_steps),
            "seed": int(holdout_config.paths.seed - in_sample_config.paths.seed),
        },
        "split_ids": holdout_split_ids,
        "overlapping_split_ids": overlapping_split_ids,
    }


def build_split_ids(config: RuntimeConfig) -> dict[str, str]:
    """Build deterministic split identifiers for leakage checks."""

    base_seed = int(config.paths.seed)
    return {
        "train": f"seed:{base_seed}:split:train",
        "validation": f"seed:{base_seed + 1}:split:validation",
        "test": f"seed:{base_seed + 2}:split:test",
    }


def _compute_changed_dimensions(in_sample_config: RuntimeConfig, holdout_config: RuntimeConfig) -> list[str]:
    changed: list[str] = []
    if abs(float(holdout_config.costs.proportional_rate - in_sample_config.costs.proportional_rate)) > 1e-12:
        changed.append("cost_level")
    if abs(float(holdout_config.market.mu - in_sample_config.market.mu)) > 1e-12:
        changed.append("drift")
    if abs(float(holdout_config.market.sigma - in_sample_config.market.sigma)) > 1e-12:
        changed.append("volatility")
    if abs(float(holdout_config.market.dt - in_sample_config.market.dt)) > 1e-12 or int(holdout_config.market.n_steps) != int(in_sample_config.market.n_steps):
        changed.append("simulator_grid")
    if int(holdout_config.paths.seed) != int(in_sample_config.paths.seed):
        changed.append("seed")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a holdout-regime summary report.")
    parser.add_argument(
        "--in-sample-config",
        type=Path,
        default=REPO_ROOT / "configs" / "entropic_no_liability_unit_spot_cost_0p0025.yaml",
        help="Selected in-sample no-liability config used as the holdout baseline.",
    )
    parser.add_argument(
        "--holdout-configs",
        type=Path,
        nargs="+",
        default=list(DEFAULT_HOLDOUT_CONFIGS),
        help="One or more holdout config paths to include in the regime collection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "holdout_regimes",
        help="Directory where the holdout-regime summary is written.",
    )
    args = parser.parse_args()

    report = build_holdout_regime_report(
        in_sample_config_path=args.in_sample_config,
        holdout_config_paths=args.holdout_configs,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"report_path={report_path}")
    print(f"verification_passed={report['verification_passed']}")
    print(f"n_holdout_regimes={len(report['holdout_regimes'])}")
    print(f"covered_dimensions={','.join(report['covered_dimensions'])}")


if __name__ == "__main__":
    main()