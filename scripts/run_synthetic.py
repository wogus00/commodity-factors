#!/usr/bin/env python3
"""Create a reproducible dashboard demo with training-only fits."""

import argparse
from pathlib import Path

from commodity_factors.analysis import run_analysis, save_bundle
from commodity_factors.demo import make_demo_panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ranks", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/demo"))
    parser.add_argument("--maxiter", type=int, default=300)
    args = parser.parse_args()
    bundle = run_analysis(
        make_demo_panel(args.seed, args.ranks), dataset_id="synthetic",
        dataset_label="Synthetic oil futures", source="Seeded Schwartz–Smith simulation",
        maxiter=args.maxiter, progress=lambda text: print(text, flush=True),
    )
    bundle.metadata["seed"] = args.seed
    bundle.metadata["assumptions"].append(
        "Demo quotes and monthly expiry dates are generated, not observed market data."
    )
    print(f"Saved demo to {save_bundle(bundle, args.output).resolve()}")


if __name__ == "__main__":
    main()
