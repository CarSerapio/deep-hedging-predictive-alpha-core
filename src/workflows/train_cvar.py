"""Command-line entry point for CVaR deep-hedging training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
REPO_ROOT = SRC_ROOT.parent

from config import load_config
from training import train_cvar_variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one or more CVaR deep-hedging variants.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "benchmark_cvar.yaml",
        help="Path to a CVaR configuration file.",
    )
    parser.add_argument(
        "--alpha",
        dest="alpha_values",
        action="append",
        type=float,
        help="CVaR alpha value to train. Repeat the flag for multiple variants.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "cvar_training",
        help="Directory where checkpoints, histories, and test hedge tensors are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional override for the simulation and initialization seed.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic TensorFlow kernels when the runtime supports them.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    alpha_values = tuple(args.alpha_values) if args.alpha_values is not None else (0.5,)
    results = train_cvar_variants(
        config,
        args.output_dir,
        alpha_values=alpha_values,
        seed=args.seed,
        deterministic=args.deterministic,
    )

    print(f"config={args.config}")
    print(f"output_dir={args.output_dir}")
    print("alpha    best_epoch    best_val_loss    test_risk    mean_abs_delta_gap")
    for result in results:
        print(
            f"{result.alpha:5.2f} {result.best_epoch:12d} {result.best_val_loss:16.6f} "
            f"{result.test_risk:12.6f} {result.mean_abs_delta_gap:20.6f}"
        )


if __name__ == "__main__":
    main()