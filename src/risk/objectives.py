"""Scalar risk objectives for batches of terminal PnL samples.

This module implements the two risk measures used by the benchmark pipeline:

- entropic risk ``ENT_theta(X) = (1 / theta) log E[exp(-theta X)])``;
- CVaR written as an auxiliary-variable optimization over terminal PnL.

The input variable ``pnl_sample`` represents terminal PnL for the liability
writer, so larger values are better. Both risk measures are therefore written
so that adding deterministic cash lowers the reported risk value.
"""

from __future__ import annotations

import numpy as np

from config import RuntimeConfig


def entropic_risk(pnl_sample: np.ndarray, *, theta: float) -> float:
    """Compute entropic risk from a batch of terminal PnL samples.

    Args:
        pnl_sample: One-dimensional array of terminal PnL samples with shape
            ``[n_paths]``.
        theta: Positive risk-aversion parameter.

    Returns:
        Scalar entropic risk value.

    Raises:
        ValueError: If ``pnl_sample`` is empty or non-numeric, or if ``theta``
            is not positive.
    """

    samples = _validate_pnl_sample(pnl_sample)
    if theta <= 0:
        raise ValueError("theta must be positive")

    scaled = -theta * samples
    # A log-mean-exp evaluation keeps the entropic objective stable when the
    # batch contains strongly negative PnL observations.
    anchor = float(np.max(scaled))
    centered = np.exp(scaled - anchor)
    return float((np.log(np.mean(centered)) + anchor) / theta)


def cvar_risk_objective(
    pnl_sample: np.ndarray,
    *,
    alpha: float,
    eta: float,
    smoothing: float = 1e-3,
) -> float:
    """Evaluate the auxiliary-variable CVaR objective for terminal PnL.

    The objective is written as

    ``eta + (1 / alpha) E[(-X - eta)_+]``

    for terminal PnL ``X``. When ``smoothing > 0``, the positive part is
    replaced by a softplus approximation so the result is differentiable in
    ``eta`` and in the pathwise samples.

    Args:
        pnl_sample: One-dimensional array of terminal PnL samples.
        alpha: Tail probability in ``(0, 1]``.
        eta: Auxiliary scalar variable from the CVaR representation.
        smoothing: Non-negative softplus temperature. Set to ``0.0`` for the
            exact non-smooth positive part.

    Returns:
        Scalar CVaR objective value at the supplied ``eta``.

    Raises:
        ValueError: If ``pnl_sample`` is invalid, if ``alpha`` is outside
            ``(0, 1]``, or if ``smoothing`` is negative.
    """

    samples = _validate_pnl_sample(pnl_sample) # Validate the input PnL sample array to ensure it is a one-dimensional, non-empty, numeric array. 
    _validate_alpha(alpha) # Validate the alpha parameter to ensure it is in the range (0, 1]. This is necessary for the CVaR calculation to be meaningful, as alpha represents the tail probability.
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")

    losses = -samples
    excess = losses - float(eta) # Compute the excess losses relative to the auxiliary variable eta. This represents the amount by which each loss exceeds eta, which is used in the CVaR objective calculation.
    if smoothing == 0.0:
        tail_penalty = np.maximum(excess, 0.0)
    else:
        tail_penalty = smoothing * _softplus(excess / smoothing)

    return float(float(eta) + np.mean(tail_penalty) / alpha)


def cvar_risk(
    pnl_sample: np.ndarray,
    *,
    alpha: float,
    smoothing: float = 1e-3,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> float:
    """Compute CVaR risk from terminal PnL samples.

    Args:
        pnl_sample: One-dimensional array of terminal PnL samples with shape
            ``[n_paths]``.
        alpha: Tail probability in ``(0, 1]``. Smaller values focus more
            strongly on the worst left-tail outcomes.
        smoothing: Non-negative softplus temperature used to smooth the
            auxiliary-variable objective. A positive value yields a
            differentiable approximation.
        tolerance: Bisection tolerance for the smoothed auxiliary-variable
            solver.
        max_iterations: Maximum number of bisection iterations for the smoothed
            solver.

    Returns:
        Scalar CVaR risk value.

    Raises:
        ValueError: If ``pnl_sample`` is invalid, if ``alpha`` is outside
            ``(0, 1]``, or if ``smoothing`` is negative.
    """

    samples = _validate_pnl_sample(pnl_sample)
    _validate_alpha(alpha)
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")

    if alpha == 1.0:
        # With the full distribution in the tail, CVaR reduces to the expected
        # loss, which is the negative mean PnL under this sign convention.
        return float(-np.mean(samples, dtype=np.float64))

    losses = -samples
    if smoothing == 0.0:
        eta = float(np.quantile(losses, 1.0 - alpha))
        return cvar_risk_objective(samples, alpha=alpha, eta=eta, smoothing=0.0)

    eta = _solve_smoothed_cvar_eta(
        losses,
        alpha=alpha,
        smoothing=smoothing,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return cvar_risk_objective(samples, alpha=alpha, eta=eta, smoothing=smoothing)


def compute_risk_objective(
    pnl_sample: np.ndarray,
    *,
    kind: str,
    theta: float | None = None,
    alpha: float | None = None,
    cvar_smoothing: float = 1e-3,
) -> float:
    """Dispatch to the configured scalar risk objective.

    Args:
        pnl_sample: One-dimensional array of terminal PnL samples.
        kind: Risk-measure name. The current module supports ``"entropic"`` and
            ``"cvar"``.
        theta: Entropic risk-aversion parameter.
        alpha: CVaR tail probability.
        cvar_smoothing: Softplus temperature used by the CVaR approximation.

    Returns:
        Scalar objective value for the supplied batch.

    Raises:
        ValueError: If the requested ``kind`` is unsupported or if its required
            parameter is missing.
    """

    normalized_kind = kind.lower()
    if normalized_kind == "entropic":
        if theta is None:
            raise ValueError("theta is required for entropic risk")
        return entropic_risk(pnl_sample, theta=theta)
    if normalized_kind == "cvar":
        if alpha is None:
            raise ValueError("alpha is required for CVaR")
        return cvar_risk(pnl_sample, alpha=alpha, smoothing=cvar_smoothing)

    raise ValueError(f"Unsupported risk objective: {kind}")


def compute_risk_objective_from_config(
    config: RuntimeConfig,
    pnl_sample: np.ndarray,
    *,
    cvar_smoothing: float = 1e-3,
) -> float:
    """Compute the configured scalar risk objective from ``RuntimeConfig``.

    Args:
        config: Validated runtime configuration containing the ``risk`` block.
        pnl_sample: One-dimensional array of terminal PnL samples.
        cvar_smoothing: Softplus temperature used by the CVaR approximation.

    Returns:
        Scalar objective value for the batch.
    """

    return compute_risk_objective(
        pnl_sample,
        kind=config.risk.kind,
        theta=config.risk.theta,
        alpha=config.risk.alpha,
        cvar_smoothing=cvar_smoothing,
    )


def _validate_pnl_sample(pnl_sample: np.ndarray) -> np.ndarray:
    """Validate the PnL sample vector used by the risk objectives."""

    samples = np.asarray(pnl_sample, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError("pnl_sample must have shape [n_paths]")
    if samples.size == 0:
        raise ValueError("pnl_sample must be non-empty")
    if not np.all(np.isfinite(samples)):
        raise ValueError("pnl_sample must contain only finite values")
    return samples


def _validate_alpha(alpha: float) -> None:
    """Validate the CVaR tail probability parameter."""

    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")


def _solve_smoothed_cvar_eta(
    losses: np.ndarray,
    *,
    alpha: float,
    smoothing: float,
    tolerance: float,
    max_iterations: int,
) -> float:
    """Solve the one-dimensional smoothed CVaR auxiliary problem by bisection."""

    spread = max(float(np.max(losses) - np.min(losses)), 1.0)
    low = float(np.min(losses) - 10.0 * (spread + smoothing))
    high = float(np.max(losses) + 10.0 * (spread + smoothing))

    low_grad = _smoothed_cvar_derivative(losses, eta=low, alpha=alpha, smoothing=smoothing)
    high_grad = _smoothed_cvar_derivative(losses, eta=high, alpha=alpha, smoothing=smoothing)

    while low_grad > 0.0:
        low -= 2.0 * spread
        low_grad = _smoothed_cvar_derivative(losses, eta=low, alpha=alpha, smoothing=smoothing)
    while high_grad < 0.0:
        high += 2.0 * spread
        high_grad = _smoothed_cvar_derivative(losses, eta=high, alpha=alpha, smoothing=smoothing)

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        grad = _smoothed_cvar_derivative(losses, eta=mid, alpha=alpha, smoothing=smoothing)
        if abs(grad) <= tolerance or (high - low) <= tolerance:
            return float(mid)
        if grad < 0.0:
            low = mid
        else:
            high = mid

    return float(0.5 * (low + high))


def _smoothed_cvar_derivative(
    losses: np.ndarray,
    *,
    eta: float,
    alpha: float,
    smoothing: float,
) -> float:
    """Return the derivative of the smoothed CVaR objective in ``eta``."""

    logits = (losses - eta) / smoothing
    return float(1.0 - np.mean(_sigmoid(logits)) / alpha)


def _softplus(x: np.ndarray) -> np.ndarray:
    """Return a stable softplus transformation."""

    return np.logaddexp(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Return a stable logistic sigmoid used in the CVaR derivative."""

    return 0.5 * (1.0 + np.tanh(0.5 * x))