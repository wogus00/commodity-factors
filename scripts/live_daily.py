#!/usr/bin/env python3
"""Estimate costs, initialize once, or update the published daily WTI analysis."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from commodity_factors.blob_store import BlobStore, LiveRepository
from commodity_factors.live_feed import DatabentoFeed
from commodity_factors.live_update import expected_observation, request_window, update_from_feed
from commodity_factors.settlement_calendar import SettlementCalendar


ROOT = Path(__file__).resolve().parents[1]


def required(name):
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(f"Configure {name}; never put API keys in web/ or commit them")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("estimate", "initialize", "update"), default="update")
    parser.add_argument("--calendar", type=Path, default=ROOT / "configs/settlement_calendar.json")
    args = parser.parse_args()
    import databento as db
    feed = DatabentoFeed(db.Historical(key=required("DATABENTO_API_KEY")))
    repository, bundle = None, None
    if args.mode != "estimate":
        if os.environ.get("LIVE_DATA_DISPLAY_APPROVED") != "true":
            raise RuntimeError("Confirm public display rights, then set LIVE_DATA_DISPLAY_APPROVED=true")
        repository = LiveRepository(
            BlobStore(required("BLOB_PRIVATE_READ_WRITE_TOKEN"), "private"),
            BlobStore(required("BLOB_PUBLIC_READ_WRITE_TOKEN"), "public"),
        )
        bundle = repository.load()
        if args.mode == "initialize" and bundle is not None:
            raise RuntimeError("History already exists; use update. Initialization never overwrites it")
        if args.mode == "update" and bundle is None:
            raise RuntimeError("Run initialize once after reviewing the historical data cost")
    now = datetime.now(timezone.utc)
    delay = os.environ.get("LIVE_PUBLICATION_DELAY_HOURS", "0")
    start, end = request_window(bundle, now=now, available_end=feed.available_end(), publication_delay_hours=delay)
    if end <= start:
        raise RuntimeError("No new provider data is available in the requested window")
    if args.mode == "estimate":
        cost = feed.estimate(start.isoformat(), end.isoformat())
        print(json.dumps({"estimated_history_cost_usd": cost, "start": start.isoformat(), "end": end.isoformat(),
                          "price_data_requested": False}))
        return
    raw, cost = feed.fetch(start.isoformat(), end.isoformat(), start_date=str(start.date()),
                           end_date=str(end.date()), max_cost_usd=os.environ.get("LIVE_MAX_COST_USD", "0"))
    calendar = SettlementCalendar(json.loads(args.calendar.read_text()))
    expected = expected_observation(calendar, now, delay)
    updated = update_from_feed(bundle, raw, calendar=calendar, now=now, expected_date=expected)
    updated.metadata["live_feed"]["publication_delay_hours"] = float(delay)
    manifest_url = repository.publish(updated, display_approved=True)
    print(json.dumps({"as_of": str(updated.panel.date_.max().date()), "estimated_data_cost_usd": cost,
                      "next_session": updated.metadata["daily_forecast"]["next_session"],
                      "public_manifest_url": manifest_url}))


if __name__ == "__main__":
    main()
