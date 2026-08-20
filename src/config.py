"""Typed experiment configuration for the benchmark pipeline.

This module loads JSON-compatible YAML, JSON, or TOML files, resolves
inheritance through an ``extends`` field, validates the assumptions used by the
current modules, and exposes a typed ``RuntimeConfig`` object consumed by
simulation, payoff, benchmark-hedge, and PnL code.

Important assumptions:
- The discrete hedge grid must satisfy ``n_steps = round(maturity / dt)`` so
    the market simulator and benchmark hedge use the same chronology.
- The market block represents a single-asset benchmark experiment with an ATM
    European call liability.
- YAML comments are only supported when PyYAML is installed because the loader
    first attempts strict JSON parsing for JSON-compatible YAML files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]


class ConfigError(ValueError):
    """Raised when a configuration file cannot be parsed or validated."""


@dataclass(frozen=True)
class ExperimentConfig:
    """High-level metadata describing one experiment run.

    Attributes:
        name: Stable experiment identifier used in logs or artifact names.
        tag: Short label for grouping related runs.
        description: Human-readable summary of the treatment or benchmark.
        with_liability: Whether the run includes the derivative liability ``Z``.
        regime_label: Dataset or regime label used to distinguish evaluations.
    """

    name: str
    tag: str
    description: str
    with_liability: bool
    regime_label: str


@dataclass(frozen=True)
class MarketConfig:
    """Single-asset market assumptions for the benchmark simulation.

    Attributes:
        s0: Initial spot price $S_0$.
        mu: Annualized physical-measure drift used for Monte Carlo simulation.
        sigma: Annualized volatility $\\sigma$.
        maturity: Option maturity $T$ in years.
        dt: Trading interval $\\Delta t$ in years.
        n_steps: Number of discrete hedge intervals on $[0, T]$.
        strike: Call strike $K$ in price units.
    """

    s0: float
    mu: float
    sigma: float
    maturity: float
    dt: float
    n_steps: int
    strike: float


@dataclass(frozen=True)
class PathConfig:
    """Monte Carlo sample sizes and seed settings for each data split."""

    train_paths: int
    val_paths: int
    test_paths: int
    seed: int


@dataclass(frozen=True)
class ModelConfig:
    """Neural-network hyperparameters reserved for later deep-hedging modules."""

    hidden_layers: int
    hidden_width: int
    activation: str
    feature_names: list[str]


@dataclass(frozen=True)
class TrainingConfig:
    """Optimizer and early-stopping hyperparameters for future training stages."""

    batch_size: int
    epochs: int
    learning_rate: float
    gradient_clip: float
    optimizer: str
    patience: int


@dataclass(frozen=True)
class CostConfig:
    """Transaction-cost settings for frictional extensions.

    Attributes:
        proportional_rate: Proportional spot trading cost applied to each
            absolute trade size ``|Delta delta_t|``.
    """

    proportional_rate: float


@dataclass(frozen=True)
class SignalConfig:
    """Optional predictive-signal settings for extension experiments.

    Attributes:
        enabled: Whether the predictive-signal extension is active.
        kind: Signal-process identifier. The first extension track uses an
            AR(1) signal that perturbs the physical-measure drift.
        ar1_phi: Autoregressive coefficient for the latent signal state.
        innovation_scale: Standard deviation of the AR(1) innovation.
        drift_scale: Loading from the current signal value into the next-step
            physical drift.
        initial_value: Deterministic initial signal value at hedge time 0.
    """

    enabled: bool
    kind: str
    ar1_phi: float
    innovation_scale: float
    drift_scale: float
    initial_value: float


@dataclass(frozen=True)
class RiskConfig:
    """Risk-measure settings used by benchmark and future learning objectives."""

    kind: str
    theta: float | None
    alpha: float | None


@dataclass(frozen=True)
class SplitConfig:
    """Relative proportions for train, validation, and test experiments."""

    train: float
    validation: float
    test: float


@dataclass(frozen=True)
class RuntimeConfig:
    """Fully validated experiment configuration consumed by all runtime modules.

    The object combines typed blocks for the experiment design, the market
    model, path counts, model/training placeholders, and risk settings. It also
    stores the resolved ``source_path`` and a short content hash so downstream
    outputs can be tied back to the exact normalized configuration.
    """

    experiment: ExperimentConfig
    market: MarketConfig
    paths: PathConfig
    model: ModelConfig
    signal: SignalConfig
    training: TrainingConfig
    costs: CostConfig
    risk: RiskConfig
    splits: SplitConfig
    source_path: str
    config_hash: str

    def as_dict(self) -> dict[str, Any]:
        """Return the typed configuration as a plain nested mapping."""

        return asdict(self)


def load_config(config_path: str | Path) -> RuntimeConfig:
    """Load, resolve, and validate a runtime configuration file.

    Args:
        config_path: Path to a JSON-compatible YAML, JSON, or TOML file.

    Returns:
        A validated ``RuntimeConfig`` object used throughout the research
        pipeline.

    Raises:
        ConfigError: If the file cannot be parsed or violates benchmark
            assumptions.
    """

    path = Path(config_path).expanduser().resolve() # Resolve the path to an absolute path, expanding any user home directory references.
    raw_config = _load_config_dict(path) # Load the configuration file into a raw dictionary, resolving any inheritance specified by the "extends" field.
    return _build_runtime_config(raw_config, path) # Build and return a typed RuntimeConfig object from the raw configuration dictionary, validating it against benchmark assumptions.


def load_config_dict(config_path: str | Path) -> dict[str, Any]:
    """Load a configuration file into an untyped nested mapping.

    This helper is mainly useful in tests or tooling that need to modify a raw
    config payload before re-validating it through ``load_config``.
    """

    path = Path(config_path).expanduser().resolve() 
    return _load_config_dict(path)


def build_runtime_config_from_dict(
    config: dict[str, Any],
    *,
    source_path: str | Path = "<in-memory>",
) -> RuntimeConfig:
    """Validate an in-memory config mapping and return a typed runtime config.

    This helper is useful for scripted parameter sweeps that need to adjust a
    loaded config payload while still recomputing validation and `config_hash`
    from the modified contents.
    """

    return _build_runtime_config(config, Path(source_path))


def _load_config_dict(path: Path) -> dict[str, Any]:
    """Recursively load a config file and resolve parent inheritance."""

    raw = _read_mapping(path) # Read the configuration file at the given path into a raw dictionary mapping.
    
    # Check if the configuration specifies an "extends" field, which indicates that it inherits from a parent configuration file. 
    # If present, remove this field from the raw dictionary and store its value in extends_value.
    extends_value = raw.pop("extends", None) 

    if extends_value is None:
        return raw # If there is no "extends" field, return the raw configuration dictionary as is.

    # Child configs override only the keys they specify, which lets benchmark
    # variants share one common market and training baseline.
    parent_path = (path.parent / extends_value).resolve() 
    parent = _load_config_dict(parent_path) 
    return _deep_merge(parent, raw)


def _read_mapping(path: Path) -> dict[str, Any]:
    """Parse a configuration file into a Python mapping."""

    if not path.exists(): 
        raise ConfigError(f"Configuration file does not exist: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml", ".json"}:
        return _parse_json_like_yaml(text, path)
    if suffix == ".toml":
        if tomllib is None:
            raise ConfigError("TOML loading requires Python 3.11+ or tomli.")
        parsed = tomllib.loads(text)
        if not isinstance(parsed, dict):
            raise ConfigError(f"Configuration root must be a mapping: {path}")
        return parsed

    raise ConfigError(f"Unsupported configuration extension: {path.suffix}")


def _parse_json_like_yaml(text: str, path: Path) -> dict[str, Any]:
    """Parse JSON-compatible YAML, falling back to PyYAML when necessary."""

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ConfigError(
                "YAML parsing requires PyYAML unless the file uses JSON-compatible YAML. "
                f"Offending file: {path}"
            ) from exc
        parsed = yaml.safe_load(text)

    if not isinstance(parsed, dict):
        raise ConfigError(f"Configuration root must be a mapping: {path}")
    return parsed


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge nested mappings so child configs can override selected fields."""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict): # If both the value in the override dictionary and the corresponding value in the base dictionary are dictionaries, recursively merge them.
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_runtime_config(config: dict[str, Any], source_path: Path) -> RuntimeConfig:
    """Convert a raw config mapping into typed dataclasses and metadata."""

    _require_sections(config, ["experiment", "market", "paths", "model", "training", "costs", "risk", "splits"])

    # ** unpacks the dictionary into keyword arguments for the dataclass constructors, creating instances of each configuration section.
    # experiment = ExperimentConfig(**config["experiment"]) equivalently creates 
    # experiment = ExperimentConfig(name=config["experiment"]["name"], tag=config["experiment"]["tag"], 
    # description=config["experiment"]["description"], with_liability=config["experiment"]["with_liability"], 
    # regime_label=config["experiment"]["regime_label"])
    experiment = ExperimentConfig(**config["experiment"])
    market = MarketConfig(**config["market"])
    paths = PathConfig(**config["paths"])
    model = ModelConfig(**config["model"])
    signal = SignalConfig(**_deep_merge(_default_signal_config_payload(), config.get("signal", {})))
    training = TrainingConfig(**config["training"])
    costs = CostConfig(**config["costs"])
    risk = RiskConfig(**config["risk"])
    splits = SplitConfig(**config["splits"])

    _validate_runtime_config(experiment, market, paths, model, signal, training, costs, risk, splits)

    # The hash is computed from the normalized typed payload rather than the raw
    # file text so logically equivalent inherited configs share the same ID.
    normalized = {
        "experiment": asdict(experiment),
        "market": asdict(market),
        "paths": asdict(paths),
        "model": asdict(model),
        "training": asdict(training),
        "costs": asdict(costs),
        "risk": asdict(risk),
        "splits": asdict(splits),
    }
    if signal.enabled:
        normalized["signal"] = asdict(signal)
    # The config_hash is generated by serializing the normalized configuration dictionary to a JSON string with sorted keys, 
    # encoding it to bytes, and then computing the SHA-256 hash of the resulting byte string. 
    # The first 16 characters of the hexadecimal representation of the hash are used as a short identifier for the configuration.
    config_hash = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    return RuntimeConfig(
        experiment=experiment,
        market=market,
        paths=paths,
        model=model,
        signal=signal,
        training=training,
        costs=costs,
        risk=risk,
        splits=splits,
        source_path=str(source_path),
        config_hash=config_hash,
    )


def _require_sections(config: dict[str, Any], sections: list[str]) -> None:
    """Enforce that all top-level experiment blocks are present."""

    missing = [section for section in sections if section not in config]
    if missing:
        raise ConfigError(f"Missing required configuration sections: {', '.join(missing)}")


def _validate_runtime_config(
    experiment: ExperimentConfig,
    market: MarketConfig,
    paths: PathConfig,
    model: ModelConfig,
    signal: SignalConfig,
    training: TrainingConfig,
    costs: CostConfig,
    risk: RiskConfig,
    splits: SplitConfig,
) -> None:
    """Validate benchmark assumptions used by the current implementation.

    The checks here protect the chronology shared by the simulator, the
    Black-Scholes hedge grid, and the terminal PnL accounting. They also ensure
    that the risk-measure parameters are internally consistent before training
    or evaluation code consumes them.
    """

    if not experiment.name:
        raise ConfigError("experiment.name must be non-empty")
    if market.s0 <= 0 or market.sigma <= 0 or market.maturity <= 0 or market.dt <= 0:
        raise ConfigError("market.s0, market.sigma, market.maturity, and market.dt must be positive")
    if market.n_steps <= 0:
        raise ConfigError("market.n_steps must be positive")
    expected_steps = round(market.maturity / market.dt)
    if expected_steps != market.n_steps:
        raise ConfigError(
            "market.n_steps must equal round(market.maturity / market.dt) for the benchmark grid"
        )
    if market.strike <= 0:
        raise ConfigError("market.strike must be positive")

    if min(paths.train_paths, paths.val_paths, paths.test_paths) <= 0:
        raise ConfigError("All path counts must be positive")
    if model.hidden_layers <= 0 or model.hidden_width <= 0:
        raise ConfigError("model.hidden_layers and model.hidden_width must be positive")
    if not model.feature_names:
        raise ConfigError("model.feature_names must be non-empty")
    benchmark_feature_names = ["spot", "time_to_maturity", "previous_hedge"]
    predictive_feature_names = benchmark_feature_names + ["predictive_signal"]
    if signal.enabled:
        if model.feature_names != predictive_feature_names:
            raise ConfigError(
                "signal-enabled configs must use model.feature_names equal to ['spot', 'time_to_maturity', 'previous_hedge', 'predictive_signal']"
            )
        if signal.kind.lower() != "ar1_drift":
            raise ConfigError("signal.kind must equal 'ar1_drift' when the predictive-signal extension is enabled")
        if abs(signal.ar1_phi) >= 1.0:
            raise ConfigError("signal.ar1_phi must satisfy |ar1_phi| < 1 for the predictive-signal extension")
        if signal.innovation_scale < 0.0:
            raise ConfigError("signal.innovation_scale must be non-negative")
    elif model.feature_names != benchmark_feature_names:
        raise ConfigError(
            "model.feature_names must equal ['spot', 'time_to_maturity', 'previous_hedge'] for the benchmark hedge state"
        )
    if training.batch_size <= 0 or training.epochs <= 0 or training.learning_rate <= 0:
        raise ConfigError("training.batch_size, training.epochs, and training.learning_rate must be positive")
    if training.gradient_clip <= 0:
        raise ConfigError("training.gradient_clip must be positive")
    if training.patience < 0:
        raise ConfigError("training.patience must be non-negative")
    if costs.proportional_rate < 0:
        raise ConfigError("costs.proportional_rate must be non-negative")

    split_total = splits.train + splits.validation + splits.test
    if abs(split_total - 1.0) > 1e-9:
        raise ConfigError("splits must sum to 1.0")

    # Later modules compare entropic and CVaR objectives, so the validator keeps
    # their parameter domains explicit at config-load time.
    risk_kind = risk.kind.lower()
    if risk_kind not in {"entropic", "cvar", "mse", "smse"}:
        raise ConfigError("risk.kind must be one of: entropic, cvar, mse, smse")
    if risk_kind == "entropic":
        if risk.theta is None or risk.theta <= 0:
            raise ConfigError("risk.theta must be positive for entropic risk")
    if risk_kind == "cvar":
        if risk.alpha is None or not (0 < risk.alpha <= 1):
            raise ConfigError("risk.alpha must be in (0, 1] for CVaR")


def _default_signal_config_payload() -> dict[str, Any]:
    return {
        "enabled": False,
        "kind": "ar1_drift",
        "ar1_phi": 0.9,
        "innovation_scale": 0.25,
        "drift_scale": 0.05,
        "initial_value": 0.0,
    }
