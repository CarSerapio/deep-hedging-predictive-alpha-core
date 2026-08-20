"""Explain holdout performance using benchmark exposure proxies."""

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
from evaluation import evaluate_benchmark_explanations
from simulators import simulate_market_data_from_config


def build_benchmark_explanation_report(
    *,
    holdout_evaluation_summary_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Build a benchmark-explanation report from holdout evaluation artifacts."""

    summary_path = holdout_evaluation_summary_path.expanduser().resolve()
    holdout_evaluation_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    policy_config = load_config(Path(holdout_evaluation_summary["policy_config"]))
    arrays_dir = summary_path.parent
    signal_feature_available = "predictive_signal" in list(policy_config.model.feature_names)

    holdout_regimes: list[dict[str, Any]] = []
    arrays_by_regime: dict[str, dict[str, np.ndarray]] = {}
    for holdout_entry in holdout_evaluation_summary["holdout_regimes"]:
        regime_label = str(holdout_entry["regime_label"])
        config_path = Path(holdout_entry["config_path"]).expanduser().resolve()
        holdout_config = load_config(config_path)
        candidate_arrays = np.load(arrays_dir / f"{_slugify(regime_label)}_arrays.npz")
        candidate_hedge = np.asarray(candidate_arrays["candidate_hedge"], dtype=np.float64)
        test_market = simulate_market_data_from_config(
            holdout_config,
            split="test",
            seed=holdout_config.paths.seed + 2,
        )

        metrics = evaluate_benchmark_explanations(
            holdout_config,
            test_market.path_tensor,
            candidate_hedge,
            reference_sigma=float(policy_config.market.sigma),
        )
        regime_summary = metrics.to_summary_dict()
        regime_summary["config_path"] = str(config_path)
        regime_summary["changed_dimensions"] = list(holdout_entry.get("changed_dimensions", []))
        regime_summary["holdout_candidate_pnl_mean"] = float(holdout_entry["candidate"]["pnl_mean"])
        regime_summary["vs_passive_mean_excess_pnl"] = float(
            holdout_entry["benchmark_adjusted"]["vs_passive"]["mean_excess_pnl"]
        )
        holdout_regimes.append(regime_summary)

        arrays_by_regime[regime_label] = {
            "candidate_pnl": metrics.candidate.pnl,
            "buy_and_hold_pnl": metrics.buy_and_hold.pnl,
            "constant_long_only_pnl": metrics.constant_long_only.pnl,
            "volatility_scaled_long_only_pnl": metrics.volatility_scaled_long_only.pnl,
            "cumulative_spot_return": metrics.cumulative_spot_return,
            "candidate_hedge": metrics.candidate.hedge,
        }

    verification = _build_verification(
        holdout_evaluation_summary,
        holdout_regimes,
        signal_feature_available=signal_feature_available,
    )
    report = {
        "report_type": "benchmark_explanations",
        "holdout_evaluation_summary_path": str(summary_path),
        "policy_config": str(Path(holdout_evaluation_summary["policy_config"]).expanduser().resolve()),
        "reference_sigma": float(policy_config.market.sigma),
        "verification_passed": all(bool(value) for value in verification.values()),
        "verification": verification,
        "aggregate": _build_aggregate_summary(holdout_regimes),
        "holdout_regimes": holdout_regimes,
    }
    return report, arrays_by_regime


def _build_aggregate_summary(holdout_regimes: list[dict[str, Any]]) -> dict[str, Any]:
    positive_vs_volatility_scaled_count = sum(
        1
        for regime in holdout_regimes
        if float(regime["benchmark_adjusted"]["vs_volatility_scaled_long_only"]["mean_excess_pnl"]) > 0.0
    )
    positive_vs_buy_and_hold_count = sum(
        1
        for regime in holdout_regimes
        if float(regime["benchmark_adjusted"]["vs_buy_and_hold"]["mean_excess_pnl"]) > 0.0
    )
    distinct_factor_r2_below_095_count = sum(
        1
        for regime in holdout_regimes
        if float(regime["exposure_regressions"]["distinct_factor_regression"]["r2"]) < 0.95
    )
    simple_signal_rule_unavailable_count = sum(
        1 for regime in holdout_regimes if not bool(regime["limitations"]["simple_signal_rule_available"])
    )
    return {
        "n_holdout_regimes": len(holdout_regimes),
        "positive_vs_buy_and_hold_count": positive_vs_buy_and_hold_count,
        "positive_vs_volatility_scaled_long_only_count": positive_vs_volatility_scaled_count,
        "distinct_factor_r2_below_095_count": distinct_factor_r2_below_095_count,
        "simple_signal_rule_unavailable_count": simple_signal_rule_unavailable_count,
    }


def _build_verification(
    holdout_evaluation_summary: dict[str, Any],
    holdout_regimes: list[dict[str, Any]],
    *,
    signal_feature_available: bool,
) -> dict[str, bool]:
    return {
        "holdout_evaluation_verification_passed": bool(holdout_evaluation_summary.get("verification_passed", False)),
        "all_holdouts_include_required_controls": all(_has_required_controls(regime) for regime in holdout_regimes),
        "signal_rule_availability_matches_policy_state": all(
            bool(regime["limitations"]["simple_signal_rule_available"]) == signal_feature_available
            for regime in holdout_regimes
        ),
        "candidate_outperforms_buy_and_hold_in_at_least_one_holdout": any(
            float(regime["benchmark_adjusted"]["vs_buy_and_hold"]["mean_excess_pnl"]) > 0.0 for regime in holdout_regimes
        ),
        "candidate_outperforms_volatility_scaled_long_only_in_at_least_one_holdout": any(
            float(regime["benchmark_adjusted"]["vs_volatility_scaled_long_only"]["mean_excess_pnl"]) > 0.0
            for regime in holdout_regimes
        ),
        "distinct_exposure_regression_not_perfect_in_at_least_one_holdout": any(
            float(regime["exposure_regressions"]["distinct_factor_regression"]["r2"]) < 0.95
            for regime in holdout_regimes
        ),
    }


def _has_required_controls(regime: dict[str, Any]) -> bool:
    controls = regime.get("controls", {})
    return all(name in controls for name in ("buy_and_hold", "constant_long_only", "volatility_scaled_long_only"))


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a benchmark-explanation report.")
    parser.add_argument(
        "--holdout-evaluation-summary",
        type=Path,
        default=REPO_ROOT / "artifacts" / "holdout_evaluation" / "summary.json",
        help="Holdout-evaluation summary JSON used as the source artifact.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "benchmark_explanations",
        help="Directory where the explanation report and arrays are written.",
    )
    args = parser.parse_args()

    report, arrays_by_regime = build_benchmark_explanation_report(
        holdout_evaluation_summary_path=args.holdout_evaluation_summary,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for regime_label, arrays in arrays_by_regime.items():
        np.savez(output_dir / f"{_slugify(regime_label)}_arrays.npz", **arrays)

    print(f"report_path={report_path}")
    print(f"verification_passed={report['verification_passed']}")
    print(f"n_holdout_regimes={len(report['holdout_regimes'])}")
    print(f"positive_vs_vol_scaled_count={report['aggregate']['positive_vs_volatility_scaled_long_only_count']}")
    print(f"distinct_factor_r2_below_095_count={report['aggregate']['distinct_factor_r2_below_095_count']}")


if __name__ == "__main__":
    main()