"""Sweep proportional transaction-cost regimes for matched training pairs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any
from xml.sax.saxutils import escape

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
REPO_ROOT = SRC_ROOT.parent

from config import RuntimeConfig, build_runtime_config_from_dict, load_config
from evaluation import evaluate_empirical_decomposition, evaluate_residual_diagnostics
from training import train_entropic_variants


def run_transaction_cost_sweep(
    *,
    no_liability_config_path: Path,
    with_liability_config_path: Path,
    cost_rates: list[float],
    theta: float,
    seed: int | None,
    deterministic: bool,
    output_dir: Path,
) -> dict[str, Any]:
    """Train and summarize a matched transaction-cost sweep."""

    if len(cost_rates) == 0:
        raise ValueError("cost_rates must contain at least one value")

    no_liability_base = load_config(no_liability_config_path)
    with_liability_base = load_config(with_liability_config_path)
    output_root = output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    point_summaries: list[dict[str, Any]] = []
    component_means: list[float] = []
    residual_means: list[float] = []
    path_match_flags: list[bool] = []

    for cost_rate in cost_rates:
        point_summary = _run_cost_point(
            no_liability_base=no_liability_base,
            with_liability_base=with_liability_base,
            cost_rate=float(cost_rate),
            theta=theta,
            seed=seed,
            deterministic=deterministic,
            output_root=output_root,
        )
        point_summaries.append(point_summary)
        component_means.append(float(point_summary["decomposition"]["mean_abs_hedge_component"]))
        residual_means.append(float(point_summary["residual_diagnostics"]["mean_abs_residual"]))
        path_match_flags.append(bool(point_summary["path_grids_match"]))

    verification = {
        "all_path_grids_match": all(path_match_flags),
        "component_mean_abs_non_increasing": _is_non_increasing(component_means),
        "residual_mean_abs_non_decreasing": _is_non_decreasing(residual_means),
    }

    report = {
        "report_type": "transaction_cost_sweep",
        "verification_passed": all(bool(value) for value in verification.values()),
        "verification": verification,
        "no_liability_base_config": str(no_liability_config_path.expanduser().resolve()),
        "with_liability_base_config": str(with_liability_config_path.expanduser().resolve()),
        "theta": float(theta),
        "seed": seed,
        "deterministic": bool(deterministic),
        "cost_rates": [float(value) for value in cost_rates],
        "points": point_summaries,
        "trend_summary": {
            "component_mean_abs_values": component_means,
            "residual_mean_abs_values": residual_means,
            "component_mean_abs_drop_first_to_last": component_means[-1] - component_means[0],
            "residual_mean_abs_change_first_to_last": residual_means[-1] - residual_means[0],
        },
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_sweep_svg(point_summaries, output_root / "transaction_cost_sweep.svg")
    report["summary_path"] = str(summary_path)
    return report


def build_cost_swept_config(
    base_config: RuntimeConfig,
    *,
    cost_rate: float,
    experiment_suffix: str,
) -> RuntimeConfig:
    """Clone a validated config and override only the proportional cost regime."""

    if cost_rate < 0.0:
        raise ValueError("cost_rate must be non-negative")

    payload = copy.deepcopy(base_config.as_dict())
    payload["costs"]["proportional_rate"] = float(cost_rate)
    payload["experiment"]["name"] = f"{base_config.experiment.name}_{experiment_suffix}"
    payload["experiment"]["description"] = (
        f"{base_config.experiment.description} Transaction-cost sweep override with proportional rate={cost_rate:.6g}."
    )
    payload["experiment"]["regime_label"] = f"{base_config.experiment.regime_label}_{experiment_suffix}"
    return build_runtime_config_from_dict(payload, source_path=f"{base_config.source_path}::{experiment_suffix}")


def _run_cost_point(
    *,
    no_liability_base: RuntimeConfig,
    with_liability_base: RuntimeConfig,
    cost_rate: float,
    theta: float,
    seed: int | None,
    deterministic: bool,
    output_root: Path,
) -> dict[str, Any]:
    cost_label = _format_cost_label(cost_rate)
    point_dir = output_root / f"cost_{cost_label}"
    point_dir.mkdir(parents=True, exist_ok=True)

    no_config = build_cost_swept_config(
        no_liability_base,
        cost_rate=cost_rate,
        experiment_suffix=f"cost_{cost_label}",
    )
    with_config = build_cost_swept_config(
        with_liability_base,
        cost_rate=cost_rate,
        experiment_suffix=f"cost_{cost_label}",
    )

    no_results = train_entropic_variants(
        no_config,
        point_dir / "no_liability_training",
        theta_values=(theta,),
        seed=seed,
        deterministic=deterministic,
    )
    with_results = train_entropic_variants(
        with_config,
        point_dir / "with_liability_training",
        theta_values=(theta,),
        seed=seed,
        deterministic=deterministic,
    )
    no_result = no_results[0]
    with_result = with_results[0]
    path_grids_match = bool(np.allclose(no_result.test_paths, with_result.test_paths, atol=0.0, rtol=0.0))
    if not path_grids_match:
        raise ValueError(f"test path grids do not match for cost_rate={cost_rate}")

    decomposition_metrics = evaluate_empirical_decomposition(
        no_config,
        with_config,
        with_result.test_paths,
        no_result.test_hedges,
        with_result.test_hedges,
    )
    residual_metrics = evaluate_residual_diagnostics(
        with_config,
        with_result.test_paths,
        decomposition_metrics.hedge_component,
        decomposition_metrics.benchmark_delta,
    )

    np.savez(
        point_dir / "decomposition_arrays.npz",
        no_liability_hedge=decomposition_metrics.no_liability_hedge.astype(np.float64, copy=False),
        with_liability_hedge=decomposition_metrics.with_liability_hedge.astype(np.float64, copy=False),
        hedge_component=decomposition_metrics.hedge_component.astype(np.float64, copy=False),
        benchmark_delta=decomposition_metrics.benchmark_delta.astype(np.float64, copy=False),
        benchmark_gap=decomposition_metrics.benchmark_gap.astype(np.float64, copy=False),
    )
    np.savez(
        point_dir / "residual_arrays.npz",
        residual=residual_metrics.residual.astype(np.float64, copy=False),
        moneyness=residual_metrics.moneyness.astype(np.float64, copy=False),
        benchmark_delta=residual_metrics.benchmark_delta.astype(np.float64, copy=False),
        hedge_component=residual_metrics.hedge_component.astype(np.float64, copy=False),
    )

    decomposition_summary = decomposition_metrics.to_summary_dict()
    residual_summary = residual_metrics.to_summary_dict()
    point_summary = {
        "cost_rate": float(cost_rate),
        "cost_label": cost_label,
        "path_grids_match": path_grids_match,
        "artifact_dir": str(point_dir),
        "no_liability_training": {
            "config_hash": no_config.config_hash,
            "artifact_dir": no_result.artifact_dir,
            "best_epoch": no_result.best_epoch,
            "epochs_ran": no_result.epochs_ran,
            "best_val_loss": no_result.best_val_loss,
            "test_risk": no_result.test_risk,
            "mean_abs_hedge": no_result.mean_abs_hedge,
            "mean_abs_delta_gap": no_result.mean_abs_delta_gap,
        },
        "with_liability_training": {
            "config_hash": with_config.config_hash,
            "artifact_dir": with_result.artifact_dir,
            "best_epoch": with_result.best_epoch,
            "epochs_ran": with_result.epochs_ran,
            "best_val_loss": with_result.best_val_loss,
            "test_risk": with_result.test_risk,
            "mean_abs_hedge": with_result.mean_abs_hedge,
            "mean_abs_delta_gap": with_result.mean_abs_delta_gap,
        },
        "decomposition": {
            "config_hash_no_liability": decomposition_summary["no_liability_config_hash"],
            "config_hash_with_liability": decomposition_summary["with_liability_config_hash"],
            "mean_abs_hedge_component": decomposition_summary["mean_abs_hedge_component"],
            "mean_abs_delta_gap": decomposition_summary["mean_abs_delta_gap"],
            "benchmark_gap": decomposition_summary["benchmark_gap"],
        },
        "residual_diagnostics": {
            "mean_abs_residual": residual_summary["mean_abs_residual"],
            "rmse_residual": residual_summary["rmse_residual"],
            "peak_mean_abs_residual_step": residual_summary["peak_mean_abs_residual_step"],
            "peak_mean_abs_residual_value": residual_summary["peak_mean_abs_residual_value"],
            "abs_residual_moneyness_distance_correlation": residual_summary["abs_residual_moneyness_distance_correlation"],
            "abs_residual_time_to_maturity_correlation": residual_summary["abs_residual_time_to_maturity_correlation"],
            "time_buckets": residual_summary["time_buckets"],
        },
    }
    (point_dir / "point_summary.json").write_text(json.dumps(point_summary, indent=2), encoding="utf-8")
    return point_summary


def _parse_float_list(raw_value: str) -> list[float]:
    values = [float(item.strip()) for item in raw_value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def _format_cost_label(value: float) -> str:
    text = f"{value:.6g}"
    return text.replace("-", "m").replace(".", "p")


def _is_non_increasing(values: list[float], *, tolerance: float = 1e-9) -> bool:
    return all(next_value <= current_value + tolerance for current_value, next_value in zip(values, values[1:]))


def _is_non_decreasing(values: list[float], *, tolerance: float = 1e-9) -> bool:
    return all(next_value >= current_value - tolerance for current_value, next_value in zip(values, values[1:]))


def _write_sweep_svg(points: list[dict[str, Any]], output_path: Path) -> None:
    x_values = [float(point["cost_rate"]) for point in points]
    series = [
        ("Mean abs hedge component", [float(point["decomposition"]["mean_abs_hedge_component"]) for point in points], "#2a9d8f"),
        ("Mean abs residual", [float(point["residual_diagnostics"]["mean_abs_residual"]) for point in points], "#e76f51"),
        ("Benchmark gap mean", [float(point["decomposition"]["benchmark_gap"]["mean"]) for point in points], "#264653"),
    ]

    width = 960
    height = 540
    left = 90
    right = 30
    top = 70
    bottom = 90
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_min = min(x_values)
    x_max = max(x_values)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    y_points = [value for _, values, _ in series for value in values]
    y_min = min(y_points)
    y_max = max(y_points)
    if y_min == y_max:
        padding = 1.0 if y_min == 0.0 else abs(y_min) * 0.1
        y_min -= padding
        y_max += padding

    def x_coord(value: float) -> float:
        ratio = (value - x_min) / (x_max - x_min)
        return left + ratio * plot_width

    def y_coord(value: float) -> float:
        ratio = (value - y_min) / (y_max - y_min)
        return top + (1.0 - ratio) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933} .axis{stroke:#52606d;stroke-width:1.2} .grid{stroke:#d9e2ec;stroke-width:1} .legend{font-size:13px}</style>',
        '<text x="480" y="36" text-anchor="middle" font-size="24">Transaction-Cost Sweep</text>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" />',
        f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" />',
    ]

    for tick in range(6):
        tick_value = y_min + (y_max - y_min) * tick / 5.0
        y = y_coord(tick_value)
        elements.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" />')
        elements.append(f'<text x="{left - 10}" y="{y + 5:.2f}" text-anchor="end" font-size="12">{tick_value:.3f}</text>')

    for x_value in x_values:
        x = x_coord(x_value)
        elements.append(f'<text x="{x:.2f}" y="{top + plot_height + 24:.2f}" text-anchor="middle" font-size="12">{x_value:.4f}</text>')

    legend_x = left + 10
    legend_y = top - 24
    for index, (label, values, color) in enumerate(series):
        polyline = " ".join(f"{x_coord(x_value):.2f},{y_coord(y_value):.2f}" for x_value, y_value in zip(x_values, values))
        elements.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline}" />')
        for x_value, y_value in zip(x_values, values):
            elements.append(f'<circle cx="{x_coord(x_value):.2f}" cy="{y_coord(y_value):.2f}" r="4" fill="{color}" />')
        legend_offset = index * 220
        elements.append(f'<line x1="{legend_x + legend_offset}" y1="{legend_y}" x2="{legend_x + legend_offset + 28}" y2="{legend_y}" stroke="{color}" stroke-width="3" />')
        elements.append(f'<text class="legend" x="{legend_x + legend_offset + 36}" y="{legend_y + 4}">{escape(label)}</text>')

    elements.append(f'<text x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle" font-size="14">Proportional transaction-cost rate</text>')
    elements.append(f'<text x="26" y="{height / 2:.1f}" text-anchor="middle" font-size="14" transform="rotate(-90 26 {height / 2:.1f})">Metric value</text>')
    elements.append('</svg>')
    output_path.write_text("\n".join(elements), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lower-cost frictional sweep.")
    parser.add_argument(
        "--no-liability-config",
        type=Path,
        default=REPO_ROOT / "configs" / "entropic_no_liability_unit_spot.yaml",
        help="Base no-liability entropic config.",
    )
    parser.add_argument(
        "--with-liability-config",
        type=Path,
        default=REPO_ROOT / "configs" / "entropic_with_liability_unit_spot.yaml",
        help="Base with-liability entropic config.",
    )
    parser.add_argument(
        "--cost-rates",
        type=_parse_float_list,
        default=[0.001, 0.0025, 0.005],
        help="Comma-separated proportional cost rates to sweep, for example '0.001,0.0025,0.005'.",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=1.0,
        help="Entropic theta value used for all sweep points.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for the shared simulation and initialization seed.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic TensorFlow kernels when the runtime supports them.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "transaction_cost_sweep",
        help="Directory where sweep artifacts are written.",
    )
    args = parser.parse_args()

    summary = run_transaction_cost_sweep(
        no_liability_config_path=args.no_liability_config,
        with_liability_config_path=args.with_liability_config,
        cost_rates=args.cost_rates,
        theta=args.theta,
        seed=args.seed,
        deterministic=args.deterministic,
        output_dir=args.output_dir,
    )
    print(f"summary_path={summary['summary_path']}")
    print(f"verification_passed={summary['verification_passed']}")
    print(f"cost_rates={','.join(f'{value:.6g}' for value in summary['cost_rates'])}")
    print(f"component_mean_abs_non_increasing={summary['verification']['component_mean_abs_non_increasing']}")
    print(f"residual_mean_abs_non_decreasing={summary['verification']['residual_mean_abs_non_decreasing']}")


if __name__ == "__main__":
    main()