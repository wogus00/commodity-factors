#!/usr/bin/env python3
"""Append completed settlement curves and record the next forecast; no fitting."""

import argparse
from pathlib import Path

import pandas as pd

from commodity_factors.analysis import load_bundle, save_bundle
from commodity_factors.daily import daily_snapshot
from commodity_factors.daily_update import append_daily_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True, help="Complete daily curves in CSV or Parquet")
    parser.add_argument("--next-session", required=True, help="Next settlement session, YYYY-MM-DD, from the feed calendar")
    parser.add_argument("--output", type=Path, help="Defaults to the existing bundle directory")
    args = parser.parse_args()
    raw = pd.read_csv(args.prices) if args.prices.suffix.lower() == ".csv" else pd.read_parquet(args.prices)
    bundle = append_daily_data(load_bundle(args.bundle), raw, next_session=args.next_session)
    save_bundle(bundle, args.output or args.bundle)
    snapshot = daily_snapshot(bundle)
    print(f"Observed through {snapshot['as_of']}; next forecast {snapshot['target_date']}")
    for row in snapshot["models"]:
        print(f"{row['model']}: {row['next_price']:.2f}" if row["next_price"] is not None
              else f"{row['model']}: selected contract expires before target")


if __name__ == "__main__":
    main()
