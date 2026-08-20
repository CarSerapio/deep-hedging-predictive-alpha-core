"""Evaluate a saved policy on a collection of holdout regimes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Any

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
REPO_ROOT = SRC_ROOT.parent

from config import RuntimeConfig, load_config
from evaluation.holdout_alpha import HoldoutAlphaMetrics, evaluate_holdout_alpha
from policies import build_mlp_policy_from_config, rollout_policy_from_config
from simulators import simulate_market_data_from_config


def build_holdout_evaluation_report(
    *,
    holdout_summary_path: Path,
    policy_config_path: Path | None = None,
    policy_run_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, HoldoutAlphaMetrics]]:
    """Evaluate a selected no-liability policy on configured holdouts."""

    holdout_summary_path = holdout_summary_path.expanduser().resolve()
    holdout_summary = json.loads(holdout_summary_path.read_text(encoding="utf-8"))

    resolved_policy_config_path = Path(
        holdout_summary["in_sample_config"] if policy_config_path is None else policy_config_path
    ).expanduser().resolve()
    resolved_policy_run_dir = (
        REPO_ROOT / "artifacts" / "entropic_no_liability_unit_spot_cost_0p0025" / "theta_1"
        if policy_run_dir is None
        else policy_run_dir.expanduser().resolve()
    )

    policy_config = load_config(resolved_policy_config_path)
    _validate_selected_policy_config(policy_config)
    history = json.loads((resolved_policy_run_dir / "history.json").read_text(encoding="utf-8"))
    policy = _load_saved_policy(policy_config, resolved_policy_run_dir / "policy.weights.h5")

    holdout_regimes: list[dict[str, Any]] = []
    metrics_by_regime: dict[str, HoldoutAlphaMetrics] = {}
    for holdout_entry in holdout_summary["holdout_regimes"]:
        holdout_config_path = Path(holdout_entry["config_path"]).expanduser().resolve()
        holdout_config = load_config(holdout_config_path)
        _validate_holdout_compatibility(policy_config, holdout_config)

        test_market = simulate_market_data_from_config(
            holdout_config,
            split="test",
            seed=holdout_config.paths.seed + 2,
        )
        candidate_hedge = np.asarray(
            rollout_policy_from_config(
                holdout_config,
                test_market.path_tensor,
                policy,
                predictive_signal_tensor=test_market.predictive_signal_tensor,
                training=False,
            ),
            dtype=np.float64,
        )
        metrics = evaluate_holdout_alpha(holdout_config, test_market.path_tensor, candidate_hedge)
        metrics_by_regime[holdout_config.experiment.regime_label] = metrics

        regime_summary = metrics.to_summary_dict()
        regime_summary["config_path"] = str(holdout_config_path)
        regime_summary["changed_dimensions"] = list(holdout_entry.get("changed_dimensions", []))
        regime_summary["materially_different_from_in_sample"] = bool(
            holdout_entry.get("materially_different_from_in_sample", False)
        )
        regime_summary["overlapping_split_ids"] = list(holdout_entry.get("overlapping_split_ids", []))
        holdout_regimes.append(regime_summary)

    verification = _build_verification(holdout_summary, holdout_regimes)
    report = {
        "report_type": "holdout_evaluation",
        "holdout_summary_path": str(holdout_summary_path),
        "policy_config": str(resolved_policy_config_path),
        "policy_run_dir": str(resolved_policy_run_dir),
        "policy_checkpoint_path": str((resolved_policy_run_dir / "policy.weights.h5").resolve()),
        "policy_history_config_hash": history.get("config_hash"),
        "policy_theta": float(policy_config.risk.theta) if policy_config.risk.theta is not None else None,
        "verification_passed": all(bool(value) for value in verification.values()),
        "verification": verification,
        "aggregate": _build_aggregate_summary(holdout_regimes),
        "holdout_regimes": holdout_regimes,
    }
    return report, metrics_by_regime


def _load_saved_policy(config: RuntimeConfig, checkpoint_path: Path):
    policy = build_mlp_policy_from_config(config, name="holdout_policy")
    policy.load_weights(checkpoint_path.expanduser().resolve())
    return policy


def _build_aggregate_summary(holdout_regimes: list[dict[str, Any]]) -> dict[str, Any]:
    positive_vs_passive_count = sum(
        1 for regime in holdout_regimes if float(regime["benchmark_adjusted"]["vs_passive"]["mean_excess_pnl"]) > 0.0
    )
    positive_vs_long_only_count = sum(
        1 for regime in holdout_regimes if float(regime["benchmark_adjusted"]["vs_long_only"]["mean_excess_pnl"]) > 0.0
    )
    positive_after_costs_count = sum(1 for regime in holdout_regimes if float(regime["candidate"]["pnl_mean"]) > 0.0)
    return {
        "n_holdout_regimes": len(holdout_regimes),
        "positive_after_costs_count": positive_after_costs_count,
        "positive_vs_passive_count": positive_vs_passive_count,
        "positive_vs_long_only_count": positive_vs_long_only_count,
    }


def _build_verification(holdout_summary: dict[str, Any], holdout_regimes: list[dict[str, Any]]) -> dict[str, bool]:
    return {
        "selected_policy_is_no_liability": True,
        "holdout_summary_verification_passed": bool(holdout_summary.get("verification_passed", False)),
        "all_holdouts_remain_seed_separated": all(len(regime["overlapping_split_ids"]) == 0 for regime in holdout_regimes),
        "alpha_positive_after_costs_in_at_least_one_holdout": any(float(regime["candidate"]["pnl_mean"]) > 0.0 for regime in holdout_regimes),
        "alpha_outperforms_passive_in_at_least_one_holdout": any(
            float(regime["benchmark_adjusted"]["vs_passive"]["mean_excess_pnl"]) > 0.0 for regime in holdout_regimes
        ),
        "alpha_does_not_disappear_in_all_holdouts": any(
            abs(float(regime["candidate"]["pnl_mean"])) > 1e-8 and float(regime["candidate"]["pnl_std"]) > 1e-8
            for regime in holdout_regimes
        ),
    }


def _validate_selected_policy_config(config: RuntimeConfig) -> None:
    if config.experiment.with_liability:
        raise ValueError("Holdout evaluation requires a no-liability selected policy config")
    if config.risk.theta is None:
        raise ValueError("Holdout evaluation currently expects an entropic no-liability policy with theta set")


def _validate_holdout_compatibility(selected_policy_config: RuntimeConfig, holdout_config: RuntimeConfig) -> None:
    if holdout_config.experiment.with_liability:
        raise ValueError("Holdout configs must set experiment.with_liability=false")
    if list(selected_policy_config.model.feature_names) != list(holdout_config.model.feature_names):
        raise ValueError("Holdout config feature_names must match the selected in-sample policy")
    if int(selected_policy_config.market.n_steps) != int(holdout_config.market.n_steps):
        raise ValueError("Holdout config n_steps must match the selected in-sample policy")
    if abs(float(selected_policy_config.market.maturity - holdout_config.market.maturity)) > 1e-12:
        raise ValueError("Holdout config maturity must match the selected in-sample policy")


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a selected policy on holdout regimes.")
    parser.add_argument(
        "--holdout-summary",
        type=Path,
        default=REPO_ROOT / "artifacts" / "holdout_regimes" / "summary.json",
        help="Holdout-regime summary JSON used to define the evaluation set.",
    )
    parser.add_argument(
        "--policy-config",
        type=Path,
        default=None,
        help="Optional override for the selected policy config used to rebuild the model.",
    )
    parser.add_argument(
        "--policy-run-dir",
        type=Path,
        default=None,
        help="Optional override for the selected run directory containing policy.weights.h5.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "holdout_evaluation",
        help="Directory where the holdout report and pathwise arrays are written.",
    )
    args = parser.parse_args()

    report, metrics_by_regime = build_holdout_evaluation_report(
        holdout_summary_path=args.holdout_summary,
        policy_config_path=args.policy_config,
        policy_run_dir=args.policy_run_dir,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for regime_label, metrics in metrics_by_regime.items():
        array_path = output_dir / f"{_slugify(regime_label)}_arrays.npz"
        np.savez(
            array_path,
            candidate_pnl=metrics.candidate.pnl,
            passive_pnl=metrics.passive.pnl,
            long_only_pnl=metrics.long_only.pnl,
            candidate_hedge=metrics.candidate.hedge,
        )

    print(f"report_path={report_path}")
    print(f"verification_passed={report['verification_passed']}")
    print(f"n_holdout_regimes={len(report['holdout_regimes'])}")
    print(f"positive_vs_passive_count={report['aggregate']['positive_vs_passive_count']}")
    print(f"positive_vs_long_only_count={report['aggregate']['positive_vs_long_only_count']}")


if __name__ == "__main__":
    main()