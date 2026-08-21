# Deep Hedging Predictive Alpha Core

This repository is the complete implementation of the methodology in "Deep Hedging, Statistical Arbitrage, and the Interpretation of Apparent Alpha in a Canonical GBM Benchmark". It includes:

- configuration loading and validation
- GBM market simulation with an optional predictive signal
- analytic hedge benchmarks
- entropic and CVaR risk objectives
- TensorFlow training loops
- holdout evaluation and benchmark explanations
- anti-spurious and predictive-signal control studies
- unit tests for the core methodology

Generated outputs are written to `artifacts/` on demand.

## Repository Layout

- `configs/`: canonical experiment configurations
- `src/`: implementation modules and command-line workflows
- `tests/`: unit tests for the core methodology and workflow logic

The source tree is organized by function rather than chronology:

- `baselines/`: analytic benchmark hedges
- `simulators/`: path generation and optional predictive-signal simulation
- `payoffs/`, `finance/`, `risk/`: payoff, PnL, and objective calculations
- `policies/`, `training/`: policy models and training loops
- `evaluation/`: decomposition, holdout, and diagnostic analyses
- `workflows/`: reproducible command-line entry points

## Environment Setup

The full workflow, including TensorFlow training, is intended for Python 3.10-3.12.

On Windows, create the virtual environment with an explicit compatible interpreter so you do not accidentally build the environment with an unsupported Python such as 3.13 or 3.14. Python 3.12 is recommended when available.

```powershell
py -0p
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[tensorflow]"
```

If Python 3.12 is not installed but Python 3.11 is, use `py -3.11 -m venv .venv` instead.

On Unix-like systems, the activation command is:

```bash
source .venv/bin/activate
```

If you only need configuration, simulation, analytics, and tests that do not train TensorFlow models, install the base project instead:

```powershell
python -m pip install -e .
```

After installation, you can either run the workflow files directly from the repository root or use the installed console scripts.

## Core Workflows

### 1. Train the entropic benchmark

```powershell
python src/workflows/train_entropic.py --config configs/benchmark_entropic.yaml --theta 1.0 --output-dir artifacts/entropic_training
```

Console script:

```powershell
dh-train-entropic --config configs/benchmark_entropic.yaml --theta 1.0 --output-dir artifacts/entropic_training
```

### 2. Train the CVaR benchmark

```powershell
python src/workflows/train_cvar.py --config configs/benchmark_cvar.yaml --alpha 0.5 --output-dir artifacts/cvar_training
```

Console script:

```powershell
dh-train-cvar --config configs/benchmark_cvar.yaml --alpha 0.5 --output-dir artifacts/cvar_training
```

### 3. Define holdout regimes

```powershell
python src/workflows/holdout_regimes.py --in-sample-config configs/entropic_no_liability_unit_spot_cost_0p0025.yaml --output-dir artifacts/holdout_regimes
```

Console script:

```powershell
dh-build-holdout-regimes --in-sample-config configs/entropic_no_liability_unit_spot_cost_0p0025.yaml --output-dir artifacts/holdout_regimes
```

### 4. Evaluate a saved policy on holdouts

```powershell
python src/workflows/holdout_evaluation.py --holdout-summary artifacts/holdout_regimes/summary.json --policy-run-dir artifacts/entropic_no_liability_unit_spot_cost_0p0025/theta_1 --output-dir artifacts/holdout_evaluation
```

Console script:

```powershell
dh-evaluate-holdouts --holdout-summary artifacts/holdout_regimes/summary.json --policy-run-dir artifacts/entropic_no_liability_unit_spot_cost_0p0025/theta_1 --output-dir artifacts/holdout_evaluation
```

### 5. Explain holdout performance with benchmark proxies

```powershell
python src/workflows/benchmark_explanations.py --holdout-evaluation-summary artifacts/holdout_evaluation/summary.json --output-dir artifacts/benchmark_explanations
```

Console script:

```powershell
dh-explain-benchmarks --holdout-evaluation-summary artifacts/holdout_evaluation/summary.json --output-dir artifacts/benchmark_explanations
```

### 6. Run anti-spurious controls

```powershell
python src/workflows/anti_spurious_controls.py --output-dir artifacts/anti_spurious_controls
```

Console script:

```powershell
dh-anti-spurious-controls --output-dir artifacts/anti_spurious_controls
```

### 7. Evaluate predictive-signal controls

```powershell
python src/workflows/predictive_signal_controls.py --holdout-evaluation-summary artifacts/holdout_evaluation/summary.json --output-dir artifacts/predictive_signal_controls
```

Console script:

```powershell
dh-predictive-signal-controls --holdout-evaluation-summary artifacts/holdout_evaluation/summary.json --output-dir artifacts/predictive_signal_controls
```

### 8. Sweep transaction-cost regimes

```powershell
python src/workflows/transaction_cost_sweep.py --output-dir artifacts/transaction_cost_sweep
```

Console script:

```powershell
dh-transaction-cost-sweep --output-dir artifacts/transaction_cost_sweep
```

Each workflow writes a `summary.json` plus any required arrays or intermediate artifacts under the selected output directory.

## Canonical Configurations

The main configs retained in `configs/` are the ones needed to reproduce the final methodology:

- `benchmark_entropic.yaml`: entropic benchmark training
- `benchmark_cvar.yaml`: CVaR benchmark training
- `entropic_no_liability_unit_spot_cost_0p0025.yaml`: selected no-liability frictional regime
- `entropic_with_liability_unit_spot_cost_0p0025.yaml`: selected with-liability frictional regime
- `predictive_signal_no_liability_unit_spot_cost_0p0025.yaml`: predictive-signal extension
- `holdout_*.yaml`: benchmark-only holdout regimes
- `predictive_holdout_*.yaml`: predictive-signal holdout regimes

## Testing

Run the full validation suite from the repository root:

```powershell
python -m unittest discover tests
```

For a narrow smoke test of configuration and the workflow-related control logic:

```powershell
python -m unittest tests.test_config tests.test_anti_spurious_controls tests.test_predictive_signal_controls
```

## Methodology Summary

The implementation matches the final project methodology:

- one underlying asset simulated under GBM in discrete time
- European call liability for with-liability experiments
- benchmark state based on `spot`, `time_to_maturity`, and `previous_hedge`
- optional AR(1) predictive signal for the predictive extension
- entropic and smoothed CVaR training objectives
- evaluation through benchmark decomposition, holdout alpha, benchmark explanations, residual diagnostics, and control studies