"""Benchmark-adjustment and risk-premium explanation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import RuntimeConfig
from evaluation.holdout_alpha import (
    StrategyPerformanceMetrics,
    evaluate_strategy_performance,
    summarize_excess_pnl,
)


@dataclass(frozen=True)
class BenchmarkExplanationMetrics:
    """Benchmark explanation diagnostics for one holdout regime."""

    regime_label: str
    candidate: StrategyPerformanceMetrics
    buy_and_hold: StrategyPerformanceMetrics
    constant_long_only: StrategyPerformanceMetrics
    volatility_scaled_long_only: StrategyPerformanceMetrics
    cumulative_spot_return: np.ndarray
    univariate_regressions: dict[str, dict[str, Any]]
    distinct_factor_regression: dict[str, Any]
    risk_premium_diagnostics: dict[str, Any]
    limitations: dict[str, Any]

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary payload."""

        return {
            "regime_label": self.regime_label,
            "candidate": self.candidate.to_summary_dict(),
            "controls": {
                "buy_and_hold": self.buy_and_hold.to_summary_dict(),
                "constant_long_only": self.constant_long_only.to_summary_dict(),
                "volatility_scaled_long_only": self.volatility_scaled_long_only.to_summary_dict(),
            },
            "benchmark_adjusted": {
                "vs_buy_and_hold": summarize_excess_pnl(self.candidate.pnl - self.buy_and_hold.pnl),
                "vs_constant_long_only": summarize_excess_pnl(self.candidate.pnl - self.constant_long_only.pnl),
                "vs_volatility_scaled_long_only": summarize_excess_pnl(
                    self.candidate.pnl - self.volatility_scaled_long_only.pnl
                ),
            },
            "exposure_regressions": {
                "univariate": self.univariate_regressions,
                "distinct_factor_regression": self.distinct_factor_regression,
            },
            "risk_premium_diagnostics": self.risk_premium_diagnostics,
            "limitations": self.limitations,
        }


def evaluate_benchmark_explanations(
    config: RuntimeConfig,
    path_tensor: np.ndarray,
    candidate_hedge_tensor: np.ndarray,
    *,
    reference_sigma: float,
    constant_long_only_exposure: float = 1.0,
    buy_and_hold_exposure: float = 1.0,
) -> BenchmarkExplanationMetrics:
    """Evaluate benchmark and risk-premium explanations."""

    candidate_hedge = _validate_hedge_tensor(candidate_hedge_tensor)
    n_paths, n_steps = candidate_hedge.shape
    signal_feature_available = "predictive_signal" in list(config.model.feature_names)

    buy_and_hold_hedge = build_constant_long_only_hedge(
        n_paths=n_paths,
        n_steps=n_steps,
        exposure=buy_and_hold_exposure,
    )
    constant_long_only_hedge = build_constant_long_only_hedge(
        n_paths=n_paths,
        n_steps=n_steps,
        exposure=constant_long_only_exposure,
    )
    volatility_scaled_hedge = build_volatility_scaled_long_only_hedge(
        n_paths=n_paths,
        n_steps=n_steps,
        reference_sigma=reference_sigma,
        current_sigma=float(config.market.sigma),
    )

    candidate = evaluate_strategy_performance(config, path_tensor, candidate_hedge, label="candidate_alpha")
    buy_and_hold = evaluate_strategy_performance(config, path_tensor, buy_and_hold_hedge, label="buy_and_hold")
    constant_long_only = evaluate_strategy_performance(
        config,
        path_tensor,
        constant_long_only_hedge,
        label="constant_long_only",
    )
    volatility_scaled_long_only = evaluate_strategy_performance(
        config,
        path_tensor,
        volatility_scaled_hedge,
        label="volatility_scaled_long_only",
    )

    cumulative_spot_return = compute_cumulative_spot_return(path_tensor)
    univariate_regressions = {
        "candidate_on_buy_and_hold_pnl": fit_linear_exposure_regression(
            candidate.pnl,
            {"buy_and_hold_pnl": buy_and_hold.pnl},
        ),
        "candidate_on_constant_long_only_pnl": fit_linear_exposure_regression(
            candidate.pnl,
            {"constant_long_only_pnl": constant_long_only.pnl},
        ),
        "candidate_on_volatility_scaled_long_only_pnl": fit_linear_exposure_regression(
            candidate.pnl,
            {"volatility_scaled_long_only_pnl": volatility_scaled_long_only.pnl},
        ),
        "candidate_on_equity_risk_premium_proxy": fit_linear_exposure_regression(
            candidate.pnl,
            {"cumulative_spot_return": cumulative_spot_return},
        ),
    }

    distinct_factors, aliases = deduplicate_factor_vectors(
        {
            "buy_and_hold_pnl": buy_and_hold.pnl,
            "constant_long_only_pnl": constant_long_only.pnl,
            "volatility_scaled_long_only_pnl": volatility_scaled_long_only.pnl,
            "cumulative_spot_return": cumulative_spot_return,
        }
    )
    distinct_factor_regression = fit_linear_exposure_regression(candidate.pnl, distinct_factors)
    distinct_factor_regression["aliased_factors"] = aliases

    pathwise_mean_hedge = np.mean(candidate_hedge, axis=1, dtype=np.float64)
    terminal_hedge = candidate_hedge[:, -1]
    risk_premium_diagnostics = {
        "proxy_name": "cumulative_spot_return",
        "theoretical_expected_proxy_mean": float(np.exp(float(config.market.mu) * float(config.market.maturity)) - 1.0),
        "realized_proxy_mean": float(np.mean(cumulative_spot_return, dtype=np.float64)),
        "realized_proxy_std": float(np.std(cumulative_spot_return, dtype=np.float64)),
        "candidate_pnl_proxy_correlation": safe_correlation(candidate.pnl, cumulative_spot_return),
        "candidate_mean_hedge_proxy_correlation": safe_correlation(pathwise_mean_hedge, cumulative_spot_return),
        "candidate_terminal_hedge_proxy_correlation": safe_correlation(terminal_hedge, cumulative_spot_return),
        "candidate_pnl_proxy_beta": float(
            univariate_regressions["candidate_on_equity_risk_premium_proxy"]["coefficients"]["cumulative_spot_return"]
        ),
        "candidate_mean_hedge_proxy_beta": float(
            fit_linear_exposure_regression(pathwise_mean_hedge, {"cumulative_spot_return": cumulative_spot_return})[
                "coefficients"
            ]["cumulative_spot_return"]
        ),
    }

    limitations = {
        "simple_signal_rule_available": signal_feature_available,
        "simple_signal_rule_reason": (
            "Predictive signal is available in the current model state, so signal-based follow-up controls can now be run on this track."
            if signal_feature_available
            else "No synthetic predictive signal is available in the current benchmark state; model.feature_names contains only spot, time_to_maturity, and previous_hedge."
        ),
        "buy_and_hold_equals_constant_long_only": bool(
            np.allclose(buy_and_hold.pnl, constant_long_only.pnl, atol=1e-12, rtol=0.0)
        ),
    }

    return BenchmarkExplanationMetrics(
        regime_label=config.experiment.regime_label,
        candidate=candidate,
        buy_and_hold=buy_and_hold,
        constant_long_only=constant_long_only,
        volatility_scaled_long_only=volatility_scaled_long_only,
        cumulative_spot_return=cumulative_spot_return,
        univariate_regressions=univariate_regressions,
        distinct_factor_regression=distinct_factor_regression,
        risk_premium_diagnostics=risk_premium_diagnostics,
        limitations=limitations,
    )


def build_constant_long_only_hedge(*, n_paths: int, n_steps: int, exposure: float) -> np.ndarray:
    """Build a constant long-only hedge tensor."""

    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive")
    return np.full((n_paths, n_steps), float(exposure), dtype=np.float64)


def build_volatility_scaled_long_only_hedge(
    *,
    n_paths: int,
    n_steps: int,
    reference_sigma: float,
    current_sigma: float,
) -> np.ndarray:
    """Build a long-only hedge scaled inversely with current volatility."""

    reference = float(reference_sigma)
    current = float(current_sigma)
    if reference <= 0.0 or current <= 0.0:
        raise ValueError("reference_sigma and current_sigma must be positive")
    return build_constant_long_only_hedge(
        n_paths=n_paths,
        n_steps=n_steps,
        exposure=reference / current,
    )


def compute_cumulative_spot_return(path_tensor: np.ndarray) -> np.ndarray:
    """Return the pathwise cumulative stock return used as an equity-premium proxy."""

    paths = np.asarray(path_tensor, dtype=np.float64)
    if paths.ndim != 2 or paths.shape[1] <= 1:
        raise ValueError("path_tensor must have shape [n_paths, n_steps + 1]")
    if np.any(paths[:, 0] <= 0.0):
        raise ValueError("initial spot must be positive for all paths")
    return np.asarray(paths[:, -1] / paths[:, 0] - 1.0, dtype=np.float64)


def deduplicate_factor_vectors(
    factors: dict[str, np.ndarray],
    *,
    atol: float = 1e-12,
    residual_atol: float = 1e-10,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Remove affine-equivalent factor vectors so the joint regression is well-defined."""

    distinct: dict[str, np.ndarray] = {}
    aliases: dict[str, str] = {}
    for factor_name, values in factors.items():
        factor_vector = _validate_vector(values, name=factor_name)
        matched_name = None
        for existing_name, existing_values in distinct.items():
            if np.allclose(factor_vector, existing_values, atol=atol, rtol=0.0) or _is_affine_equivalent(
                factor_vector,
                existing_values,
                residual_atol=residual_atol,
            ):
                matched_name = existing_name
                break
        if matched_name is None:
            distinct[factor_name] = factor_vector
        else:
            aliases[factor_name] = matched_name
    return distinct, aliases


def fit_linear_exposure_regression(target: np.ndarray, factors: dict[str, np.ndarray]) -> dict[str, Any]:
    """Fit a simple OLS exposure regression with an intercept."""

    y = _validate_vector(target, name="target")
    if len(factors) == 0:
        raise ValueError("factors must contain at least one regressor")

    factor_names = list(factors.keys())
    factor_arrays = [_validate_vector(factors[name], name=name) for name in factor_names]
    for factor_array in factor_arrays:
        if factor_array.shape[0] != y.shape[0]:
            raise ValueError("all factor vectors must align with target")

    design = np.column_stack([np.ones(y.shape[0], dtype=np.float64), *factor_arrays])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residual = y - fitted
    ss_res = float(np.sum(np.square(residual), dtype=np.float64))
    centered_y = y - np.mean(y, dtype=np.float64)
    ss_tot = float(np.sum(np.square(centered_y), dtype=np.float64))
    r2 = 0.0 if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)

    return {
        "factor_names": factor_names,
        "intercept": float(coefficients[0]),
        "coefficients": {
            factor_name: float(coefficients[index + 1])
            for index, factor_name in enumerate(factor_names)
        },
        "r2": r2,
        "residual_std": float(np.std(residual, dtype=np.float64)),
        "condition_number": float(np.linalg.cond(design)),
        "target_mean": float(np.mean(y, dtype=np.float64)),
        "factor_correlations": {
            factor_name: safe_correlation(y, factor_values)
            for factor_name, factor_values in zip(factor_names, factor_arrays, strict=False)
        },
    }


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Return a stable Pearson correlation for two aligned vectors."""

    left_vector = _validate_vector(left, name="left")
    right_vector = _validate_vector(right, name="right")
    if left_vector.shape[0] != right_vector.shape[0]:
        raise ValueError("left and right must align")
    left_std = float(np.std(left_vector, dtype=np.float64))
    right_std = float(np.std(right_vector, dtype=np.float64))
    if left_std <= 1e-12 or right_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(left_vector, right_vector)[0, 1])


def _is_affine_equivalent(left: np.ndarray, right: np.ndarray, *, residual_atol: float) -> bool:
    left_vector = _validate_vector(left, name="left")
    right_vector = _validate_vector(right, name="right")
    if left_vector.shape[0] != right_vector.shape[0]:
        raise ValueError("left and right must align")

    design = np.column_stack(
        [
            np.ones(left_vector.shape[0], dtype=np.float64),
            right_vector,
        ]
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, left_vector, rcond=None)
    residual = left_vector - design @ coefficients
    return float(np.std(residual, dtype=np.float64)) <= residual_atol


def _validate_hedge_tensor(hedge_tensor: np.ndarray) -> np.ndarray:
    hedges = np.asarray(hedge_tensor, dtype=np.float64)
    if hedges.ndim != 2 or hedges.shape[1] <= 0:
        raise ValueError("hedge_tensor must have shape [n_paths, n_steps]")
    return hedges


def _validate_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.shape[0] <= 0:
        raise ValueError(f"{name} must have shape [n_paths]")
    return vector