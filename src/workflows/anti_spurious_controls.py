"""Run anti-spurious control summaries for saved no-liability policies."""

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

from config import load_config
from evaluation import (
    build_unavailable_signal_control_entry,
    compute_control_weakening,
    summarize_saved_no_liability_control,
    uses_only_benchmark_features,
)
def build_anti_spurious_control_report(
    *,
    benchmark_only_config: Path,
    benchmark_only_run_dir: Path,
    zero_drift_config: Path,
    zero_drift_run_dir: Path,
) -> dict[str, Any]:
    """Build an anti-spurious-control report for the current benchmark state."""

    benchmark_config = load_config(benchmark_only_config)
    reference_sigma = float(benchmark_config.market.sigma)

    benchmark_only = summarize_saved_no_liability_control(
        config_path=benchmark_only_config,
        run_dir=benchmark_only_run_dir,
        reference_sigma=reference_sigma,
        control_name="benchmark_only_state",
        note="The current state already uses only benchmark observables: spot, time_to_maturity, and previous_hedge.",
    )
    zero_drift = summarize_saved_no_liability_control(
        config_path=zero_drift_config,
        run_dir=zero_drift_run_dir,
        reference_sigma=reference_sigma,
        control_name="zero_drift",
        note="The same no-liability experiment retrained under mu = 0 to test whether the candidate component survives after drift is removed.",
    )
    zero_drift["weakening_vs_benchmark_only"] = compute_control_weakening(benchmark_only, zero_drift)

    zero_predictive_signal = build_unavailable_signal_control_entry(
        control_name="zero_predictive_signal",
        feature_names=list(benchmark_config.model.feature_names),
    )
    shuffled_signals = build_unavailable_signal_control_entry(
        control_name="shuffled_signals",
        feature_names=list(benchmark_config.model.feature_names),
    )

    available_controls = {
        "benchmark_only_state": benchmark_only,
        "zero_drift": zero_drift,
    }
    unavailable_controls = {
        "zero_predictive_signal": zero_predictive_signal,
        "shuffled_signals": shuffled_signals,
    }
    verification = _build_verification(benchmark_config, benchmark_only, zero_drift, unavailable_controls)

    return {
        "report_type": "anti_spurious_controls",
        "benchmark_only_config": str(benchmark_only_config.expanduser().resolve()),
        "benchmark_only_run_dir": str(benchmark_only_run_dir.expanduser().resolve()),
        "zero_drift_config": str(zero_drift_config.expanduser().resolve()),
        "zero_drift_run_dir": str(zero_drift_run_dir.expanduser().resolve()),
        "verification_passed": all(bool(value) for value in verification.values()),
        "verification": verification,
        "aggregate": {
            "available_control_count": len(available_controls),
            "unavailable_control_count": len(unavailable_controls),
            "zero_drift_pnl_mean": float(zero_drift["candidate"]["pnl_mean"]),
            "zero_drift_vs_passive_mean_excess": float(zero_drift["benchmark_adjusted"]["vs_passive"]["mean_excess_pnl"]),
        },
        "available_controls": available_controls,
        "unavailable_controls": unavailable_controls,
    }


def _build_verification(
    benchmark_config,
    benchmark_only: dict[str, Any],
    zero_drift: dict[str, Any],
    unavailable_controls: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    benchmark_pnl_mean = float(benchmark_only["candidate"]["pnl_mean"])
    zero_drift_pnl_mean = float(zero_drift["candidate"]["pnl_mean"])
    benchmark_excess = float(benchmark_only["benchmark_adjusted"]["vs_passive"]["mean_excess_pnl"])
    zero_drift_excess = float(zero_drift["benchmark_adjusted"]["vs_passive"]["mean_excess_pnl"])

    return {
        "benchmark_only_state_uses_only_benchmark_features": uses_only_benchmark_features(benchmark_config),
        "signal_destruction_controls_are_unavailable_in_current_state_space": all(
            entry["status"] == "not_applicable" for entry in unavailable_controls.values()
        ),
        "zero_drift_candidate_mean_pnl_is_not_higher_than_benchmark_only": zero_drift_pnl_mean <= benchmark_pnl_mean + 1e-12,
        "zero_drift_candidate_vs_passive_mean_excess_is_not_higher_than_benchmark_only": zero_drift_excess <= benchmark_excess + 1e-12,
        "zero_drift_candidate_is_not_positive_after_costs": zero_drift_pnl_mean <= 0.0,
        "zero_drift_candidate_does_not_outperform_passive": zero_drift_excess <= 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an anti-spurious-control report.")
    parser.add_argument(
        "--benchmark-only-config",
        type=Path,
        default=REPO_ROOT / "configs" / "entropic_no_liability_unit_spot_cost_0p0025.yaml",
        help="Selected no-liability config, which already uses only benchmark observables.",
    )
    parser.add_argument(
        "--benchmark-only-run-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "entropic_no_liability_unit_spot_cost_0p0025" / "theta_1",
        help="Run directory for the selected no-liability policy.",
    )
    parser.add_argument(
        "--zero-drift-config",
        type=Path,
        default=REPO_ROOT / "configs" / "entropic_no_liability_unit_spot_zero_drift_cost_0p0025.yaml",
        help="Zero-drift anti-spurious-control config.",
    )
    parser.add_argument(
        "--zero-drift-run-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "entropic_no_liability_unit_spot_zero_drift_cost_0p0025" / "theta_1",
        help="Run directory for the zero-drift control policy.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "anti_spurious_controls",
        help="Directory where the anti-spurious summary is written.",
    )
    args = parser.parse_args()

    report = build_anti_spurious_control_report(
        benchmark_only_config=args.benchmark_only_config,
        benchmark_only_run_dir=args.benchmark_only_run_dir,
        zero_drift_config=args.zero_drift_config,
        zero_drift_run_dir=args.zero_drift_run_dir,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"report_path={report_path}")
    print(f"verification_passed={report['verification_passed']}")
    print(f"available_controls={report['aggregate']['available_control_count']}")
    print(f"unavailable_controls={report['aggregate']['unavailable_control_count']}")
    print(f"zero_drift_pnl_mean={report['aggregate']['zero_drift_pnl_mean']:.12f}")


if __name__ == "__main__":
    main()