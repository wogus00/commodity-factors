#!/usr/bin/env python3
"""Export saved results to a browser-readable JSON file, without fitting models."""

import argparse
from pathlib import Path

from commodity_factors.analysis import load_bundle
from commodity_factors.web_bundle import browser_bytes


ROOT = Path(__file__).resolve().parents[1]


def export_bundle(directory, output):
    bundle = load_bundle(directory)
    output = Path(output).resolve()
    # The deployable tree only contains shareable generated quotes. Empirical
    # files can be opened in the browser without being uploaded to a host.
    if bundle.metadata["dataset_id"] != "synthetic" and output.is_relative_to(ROOT / "web"):
        raise ValueError("Keep empirical exports outside web/. Use results/<dataset>/browser.json for local import.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(browser_bytes(bundle))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "data/synthetic/demo")
    parser.add_argument("--output", type=Path, default=ROOT / "web/data/demo.json")
    args = parser.parse_args()
    print(f"Exported {export_bundle(args.bundle, args.output)}")


if __name__ == "__main__":
    main()
