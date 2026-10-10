"""Serialize a saved analysis for the static web workspace."""

import json

from commodity_factors.analysis import TABLES
from commodity_factors.daily import daily_snapshot


def browser_payload(bundle):
    return {
        "format": "commodity-factors-web", "version": 1,
        "metadata": bundle.metadata,
        "daily_latest": daily_snapshot(bundle),
        "tables": {name: json.loads(getattr(bundle, name).to_json(
            orient="split", index=False, date_format="iso", double_precision=12)) for name in TABLES},
    }


def browser_bytes(bundle):
    return (json.dumps(browser_payload(bundle), separators=(",", ":"), allow_nan=False) + "\n").encode()
