"""Anti-spurious-alpha control helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from config import RuntimeConfig, load_config
from evaluation.benchmark_explanations import evaluate_benchmark_explanations
from evaluation.holdout_alpha import summarize_excess_pnl
from simulators import simulate_market_data_from_config


def summarize_saved_no_liability_control(
    *,
    config_path: Path,
    run_dir: Path,
    reference_sigma: float,
    control_name: str,
    note: str,
) -> dict[str, Any]:
    """Load one saved no-liability run and summarize it as a control."""

    resolved_config_path = config_path.expanduser().resolve()
    resolved_run_dir = run_dir.expanduser().resolve()
    config = load_config(resolved_config_path)
    if config.experiment.with_liability:
        raise ValueError("Control runs must use experiment.with_liability=false")

    hedge_tensor = np.asarray(np.load(resolved_run_dir / "test_hedges.npy"), dtype=np.float64)
    market_data = simulate_market_data_from_config(
        config,
        split="test",
        seed=config.paths.seed + 2,
    )
    return summarize_no_liability_control_from_hedge(
        config=config,
        path_tensor=market_data.path_tensor,
        hedge_tensor=hedge_tensor,
        reference_sigma=reference_sigma,
        control_name=control_name,
        note=note,
        config_path=resolved_config_path,
        run_dir=resolved_run_dir,
    )


def summarize_no_liability_control_from_hedge(
    *,
    config: RuntimeConfig,
    path_tensor: np.ndarray,
    hedge_tensor: np.ndarray,
    reference_sigma: float,
    control_name: str,
    note: str,
    config_path: Path | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Summarize one no-liability control from a supplied hedge tensor."""

    metrics = evaluate_benchmark_explanations(
        config,
        path_tensor,
        hedge_tensor,
        reference_sigma=reference_sigma,
    )
    summary = metrics.to_summary_dict()
    summary.setdefault("benchmark_adjusted", {})["vs_passive"] = summarize_excess_pnl(metrics.candidate.pnl)
    summary.update(
        {
            "control_name": control_name,
            "status": "available",
            "config_path": None if config_path is None else str(config_path),
            "run_dir": None if run_dir is None else str(run_dir),
            "note": note,
        }
    )
    return summary


def build_shuffled_signal_tensor(
    predictive_signal_tensor: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Shuffle predictive signals across paths at each hedge step."""

    signal_tensor = np.asarray(predictive_signal_tensor, dtype=np.float64)
    if signal_tensor.ndim != 2 or signal_tensor.shape[1] <= 0:
        raise ValueError("predictive_signal_tensor must have shape [n_paths, n_steps]")

    rng = np.random.default_rng(int(seed))
    shuffled = np.empty_like(signal_tensor, dtype=np.float64)
    for step in range(signal_tensor.shape[1]):
        shuffled[:, step] = rng.permutation(signal_tensor[:, step])
    return shuffled


def summarize_hedge_response(
    baseline_hedge_tensor: np.ndarray,
    comparison_hedge_tensor: np.ndarray,
) -> dict[str, float]:
    """Summarize how much a control changes the hedge relative to baseline."""

    baseline = np.asarray(baseline_hedge_tensor, dtype=np.float64)
    comparison = np.asarray(comparison_hedge_tensor, dtype=np.float64)
    if baseline.ndim != 2 or comparison.ndim != 2 or baseline.shape != comparison.shape:
        raise ValueError("baseline_hedge_tensor and comparison_hedge_tensor must share shape [n_paths, n_steps]")

    delta = comparison - baseline
    return {
        "mean_abs_hedge_shift": float(np.mean(np.abs(delta), dtype=np.float64)),
        "max_abs_hedge_shift": float(np.max(np.abs(delta))),
        "rmse_hedge_shift": float(np.sqrt(np.mean(np.square(delta), dtype=np.float64))),
    }


def build_unavailable_signal_control_entry(
    *,
    control_name: str,
    feature_names: list[str],
) -> dict[str, Any]:
    """Return a formal entry for unavailable signal-destruction controls."""

    return {
        "control_name": control_name,
        "status": "not_applicable",
        "feature_names": list(feature_names),
        "reason": (
            "No predictive signal inputs exist in the current benchmark state; "
            "model.feature_names is fixed to [spot, time_to_maturity, previous_hedge], "
            "so this control cannot be instantiated without widening the repository state space."
        ),
    }


def compute_control_weakening(
    benchmark_only_entry: dict[str, Any],
    comparison_entry: dict[str, Any],
) -> dict[str, float]:
    """Compute how strongly one control weakens the benchmark baseline."""

    benchmark_candidate = benchmark_only_entry["candidate"]
    comparison_candidate = comparison_entry["candidate"]
    benchmark_vs_passive = benchmark_only_entry["benchmark_adjusted"]["vs_passive"]
    comparison_vs_passive = comparison_entry["benchmark_adjusted"]["vs_passive"]

    return {
        "candidate_pnl_mean_change": float(comparison_candidate["pnl_mean"] - benchmark_candidate["pnl_mean"]),
        "candidate_abs_pnl_mean_change": float(
            abs(comparison_candidate["pnl_mean"]) - abs(benchmark_candidate["pnl_mean"])
        ),
        "candidate_mean_abs_hedge_change": float(
            comparison_candidate["mean_abs_hedge"] - benchmark_candidate["mean_abs_hedge"]
        ),
        "vs_passive_mean_excess_change": float(
            comparison_vs_passive["mean_excess_pnl"] - benchmark_vs_passive["mean_excess_pnl"]
        ),
        "pnl_std_change": float(comparison_candidate["pnl_std"] - benchmark_candidate["pnl_std"]),
    }


def uses_only_benchmark_features(config: RuntimeConfig) -> bool:
    """Return whether the config uses the current benchmark-only state."""

    return list(config.model.feature_names) == ["spot", "time_to_maturity", "previous_hedge"]