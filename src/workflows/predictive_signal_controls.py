"""Evaluate signal-destruction controls for predictive-signal policies."""

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

from config import load_config
from evaluation import (
    build_shuffled_signal_tensor,
    compute_control_weakening,
    summarize_hedge_response,
    summarize_no_liability_control_from_hedge,
)
from policies import build_mlp_policy_from_config, rollout_policy_from_config
from simulators import simulate_market_data_from_config
def build_predictive_signal_controls_report(
    *,
    holdout_evaluation_summary_path: Path,
    policy_run_dir: Path | None = None,
) -> dict[str, Any]:
    """Build predictive-signal destruction controls for a saved policy."""

    summary_path = holdout_evaluation_summary_path.expanduser().resolve()
    holdout_evaluation_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    policy_config_path = Path(holdout_evaluation_summary["policy_config"]).expanduser().resolve()
    policy_config = load_config(policy_config_path)
    if "predictive_signal" not in list(policy_config.model.feature_names):
        raise ValueError("Predictive-signal controls require a policy config whose feature_names include predictive_signal")

    resolved_policy_run_dir = (
        policy_run_dir.expanduser().resolve()
        if policy_run_dir is not None
        else Path(holdout_evaluation_summary["policy_run_dir"]).expanduser().resolve()
    )
    policy = _load_saved_policy(policy_config, resolved_policy_run_dir / "policy.weights.h5")
    reference_sigma = float(policy_config.market.sigma)
    arrays_dir = summary_path.parent

    holdout_regimes: list[dict[str, Any]] = []
    for holdout_entry in holdout_evaluation_summary["holdout_regimes"]:
        regime_label = str(holdout_entry["regime_label"])
        holdout_config_path = Path(holdout_entry["config_path"]).expanduser().resolve()
        holdout_config = load_config(holdout_config_path)
        market_data = simulate_market_data_from_config(
            holdout_config,
            split="test",
            seed=holdout_config.paths.seed + 2,
        )
        predictive_signal_tensor = market_data.predictive_signal_tensor
        if predictive_signal_tensor is None:
            raise ValueError("Predictive holdout configs must simulate predictive signals")

        saved_arrays = np.load(arrays_dir / f"{_slugify(regime_label)}_arrays.npz")
        saved_candidate_hedge = np.asarray(saved_arrays["candidate_hedge"], dtype=np.float64)
        baseline_hedge = np.asarray(
            rollout_policy_from_config(
                holdout_config,
                market_data.path_tensor,
                policy,
                predictive_signal_tensor=predictive_signal_tensor,
                training=False,
            ),
            dtype=np.float64,
        )
        zero_signal_hedge = np.asarray(
            rollout_policy_from_config(
                holdout_config,
                market_data.path_tensor,
                policy,
                predictive_signal_tensor=np.zeros_like(predictive_signal_tensor, dtype=np.float64),
                training=False,
            ),
            dtype=np.float64,
        )
        shuffled_signal_hedge = np.asarray(
            rollout_policy_from_config(
                holdout_config,
                market_data.path_tensor,
                policy,
                predictive_signal_tensor=build_shuffled_signal_tensor(
                    predictive_signal_tensor,
                    seed=holdout_config.paths.seed + 2500,
                ),
                training=False,
            ),
            dtype=np.float64,
        )

        baseline_summary = summarize_no_liability_control_from_hedge(
            config=holdout_config,
            path_tensor=market_data.path_tensor,
            hedge_tensor=baseline_hedge,
            reference_sigma=reference_sigma,
            control_name="baseline_predictive_policy",
            note="Saved predictive policy replayed on the holdout under the observed predictive signal.",
            config_path=holdout_config_path,
            run_dir=resolved_policy_run_dir,
        )
        zero_signal_summary = summarize_no_liability_control_from_hedge(
            config=holdout_config,
            path_tensor=market_data.path_tensor,
            hedge_tensor=zero_signal_hedge,
            reference_sigma=reference_sigma,
            control_name="zero_predictive_signal",
            note="The saved policy is replayed on the same holdout paths with the predictive-signal input replaced by zeros at every hedge step.",
            config_path=holdout_config_path,
        )
        shuffled_signal_summary = summarize_no_liability_control_from_hedge(
            config=holdout_config,
            path_tensor=market_data.path_tensor,
            hedge_tensor=shuffled_signal_hedge,
            reference_sigma=reference_sigma,
            control_name="shuffled_predictive_signal",
            note="The saved policy is replayed on the same holdout paths with predictive signals shuffled across paths at each hedge step.",
            config_path=holdout_config_path,
        )

        zero_signal_summary["weakening_vs_baseline"] = compute_control_weakening(baseline_summary, zero_signal_summary)
        zero_signal_summary["hedge_response_vs_baseline"] = summarize_hedge_response(baseline_hedge, zero_signal_hedge)
        shuffled_signal_summary["weakening_vs_baseline"] = compute_control_weakening(
            baseline_summary,
            shuffled_signal_summary,
        )
        shuffled_signal_summary["hedge_response_vs_baseline"] = summarize_hedge_response(
            baseline_hedge,
            shuffled_signal_hedge,
        )

        holdout_regimes.append(
            {
                "regime_label": regime_label,
                "config_path": str(holdout_config_path),
                "changed_dimensions": list(holdout_entry.get("changed_dimensions", [])),
                "saved_holdout_replay_max_abs_diff": float(np.max(np.abs(baseline_hedge - saved_candidate_hedge))),
                "baseline": baseline_summary,
                "zero_predictive_signal": zero_signal_summary,
                "shuffled_predictive_signal": shuffled_signal_summary,
            }
        )

    verification = _build_verification(holdout_evaluation_summary, policy_config, holdout_regimes)
    return {
        "report_type": "predictive_signal_controls",
        "holdout_evaluation_summary_path": str(summary_path),
        "policy_config": str(policy_config_path),
        "policy_run_dir": str(resolved_policy_run_dir),
        "verification_passed": all(bool(value) for value in verification.values()),
        "verification": verification,
        "aggregate": _build_aggregate_summary(holdout_regimes),
        "holdout_regimes": holdout_regimes,
    }


def _build_aggregate_summary(holdout_regimes: list[dict[str, Any]]) -> dict[str, Any]:
    replay_match_count = sum(1 for regime in holdout_regimes if float(regime["saved_holdout_replay_max_abs_diff"]) <= 1e-6)
    zero_signal_weaker_pnl_count = sum(
        1
        for regime in holdout_regimes
        if float(regime["zero_predictive_signal"]["candidate"]["pnl_mean"])
        <= float(regime["baseline"]["candidate"]["pnl_mean"]) + 1e-12
    )
    shuffled_signal_weaker_pnl_count = sum(
        1
        for regime in holdout_regimes
        if float(regime["shuffled_predictive_signal"]["candidate"]["pnl_mean"])
        <= float(regime["baseline"]["candidate"]["pnl_mean"]) + 1e-12
    )
    signal_response_detected_count = sum(
        1
        for regime in holdout_regimes
        if max(
            float(regime["zero_predictive_signal"]["hedge_response_vs_baseline"]["mean_abs_hedge_shift"]),
            float(regime["shuffled_predictive_signal"]["hedge_response_vs_baseline"]["mean_abs_hedge_shift"]),
        )
        > 1e-6
    )
    return {
        "n_holdout_regimes": len(holdout_regimes),
        "baseline_replay_match_count": replay_match_count,
        "zero_signal_weaker_pnl_count": zero_signal_weaker_pnl_count,
        "shuffled_signal_weaker_pnl_count": shuffled_signal_weaker_pnl_count,
        "signal_response_detected_count": signal_response_detected_count,
    }


def _build_verification(
    holdout_evaluation_summary: dict[str, Any],
    policy_config,
    holdout_regimes: list[dict[str, Any]],
) -> dict[str, bool]:
    return {
        "holdout_evaluation_verification_passed": bool(holdout_evaluation_summary.get("verification_passed", False)),
        "policy_uses_predictive_signal_feature": "predictive_signal" in list(policy_config.model.feature_names),
        "baseline_replay_matches_saved_holdout_hedges": all(
            float(regime["saved_holdout_replay_max_abs_diff"]) <= 1e-6 for regime in holdout_regimes
        ),
        "zero_signal_weakens_pnl_in_at_least_one_holdout": any(
            float(regime["zero_predictive_signal"]["candidate"]["pnl_mean"])
            < float(regime["baseline"]["candidate"]["pnl_mean"]) - 1e-12
            for regime in holdout_regimes
        ),
        "shuffled_signal_weakens_pnl_in_at_least_one_holdout": any(
            float(regime["shuffled_predictive_signal"]["candidate"]["pnl_mean"])
            < float(regime["baseline"]["candidate"]["pnl_mean"]) - 1e-12
            for regime in holdout_regimes
        ),
        "signal_destruction_changes_hedge_in_at_least_one_holdout": any(
            max(
                float(regime["zero_predictive_signal"]["hedge_response_vs_baseline"]["mean_abs_hedge_shift"]),
                float(regime["shuffled_predictive_signal"]["hedge_response_vs_baseline"]["mean_abs_hedge_shift"]),
            )
            > 1e-6
            for regime in holdout_regimes
        ),
    }


def _load_saved_policy(config, checkpoint_path: Path):
    policy = build_mlp_policy_from_config(config, name="predictive_policy")
    policy.load_weights(checkpoint_path.expanduser().resolve())
    return policy


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build predictive-signal destruction controls.")
    parser.add_argument(
        "--holdout-evaluation-summary",
        type=Path,
        default=REPO_ROOT / "artifacts" / "holdout_evaluation" / "summary.json",
        help="Holdout-evaluation summary JSON for the predictive policy.",
    )
    parser.add_argument(
        "--policy-run-dir",
        type=Path,
        default=None,
        help="Optional override for the predictive policy run directory containing policy.weights.h5.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "predictive_signal_controls",
        help="Directory where the predictive-signal control summary is written.",
    )
    args = parser.parse_args()

    report = build_predictive_signal_controls_report(
        holdout_evaluation_summary_path=args.holdout_evaluation_summary,
        policy_run_dir=args.policy_run_dir,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"report_path={report_path}")
    print(f"verification_passed={report['verification_passed']}")
    print(f"zero_signal_weaker_count={report['aggregate']['zero_signal_weaker_pnl_count']}")
    print(f"shuffled_signal_weaker_count={report['aggregate']['shuffled_signal_weaker_pnl_count']}")
    print(f"signal_response_detected_count={report['aggregate']['signal_response_detected_count']}")


if __name__ == "__main__":
    main()