#!/usr/bin/env python3
"""Calibrate local futures data once and save a dashboard bundle."""

import argparse
from pathlib import Path

from commodity_factors.analysis import run_analysis, save_bundle
from commodity_factors.data import load_raw_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/processed/wti_panel_2021_2025.parquet"))
    parser.add_argument("--output", type=Path, default=Path("results/wti"))
    parser.add_argument("--validation-start", default="2024-01-01")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 20])
    parser.add_argument("--dataset-id", default="wti")
    parser.add_argument("--dataset-label", default="WTI crude oil")
    parser.add_argument("--source", default="WRDS / LSEG Datastream")
    parser.add_argument("--maxiter", type=int, default=300)
    args = parser.parse_args()
    bundle = run_analysis(
        load_raw_data(args.panel), validation_start=args.validation_start,
        test_start=args.test_start, horizons=args.horizons,
        dataset_id=args.dataset_id, dataset_label=args.dataset_label, source=args.source,
        maxiter=args.maxiter, progress=lambda text: print(text, flush=True),
    )
    destination = save_bundle(bundle, args.output)
    headline = bundle.metrics.query("split == 'test' and horizon == 1 and maturity_rank == 0")
    print(headline[["model", "rmse_log", "mae_log", "n_dates"]].to_string(index=False))
    print(f"Saved analysis to {destination.resolve()}")


if __name__ == "__main__":
    main()
