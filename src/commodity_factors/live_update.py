"""Coordinate daily settlement ingestion, immutable forecasts, and publication."""

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from commodity_factors.analysis import run_analysis
from commodity_factors.daily_update import append_daily_data
from commodity_factors.live_feed import DATASET, SYMBOL, DATASET_ID


def request_window(bundle, *, now, available_end, history_start="2021-01-01", publication_delay_hours=0):
    delay = float(publication_delay_hours)
    if not 0 <= delay <= 720:
        raise ValueError("Publication delay must be between 0 and 720 hours")
    # A reference session can be published on a later UTC date. Fetch overlap
    # and filter by ts_ref, rather than using the receipt date as the price date.
    start_date = history_start if bundle is None else str((bundle.panel.date_.max() - pd.Timedelta(days=3)).date())
    cutoff = min(pd.Timestamp(now) - pd.Timedelta(hours=delay), pd.Timestamp(available_end))
    return pd.Timestamp(start_date, tz="UTC"), cutoff


def expected_observation(calendar, now, delay_hours):
    local = (now - timedelta(hours=float(delay_hours))).astimezone(ZoneInfo("America/Chicago"))
    # Run several hours after the normal 13:30 CT settlement calculation.
    day = local.date() if local.time() >= time(17, 30) else local.date() - timedelta(days=1)
    return calendar.previous_session(day)


def update_from_feed(bundle, raw, *, calendar, now, expected_date, calibration=None):
    if bundle is not None and bundle.metadata.get("dataset_id") != DATASET_ID:
        raise ValueError("WRDS and Databento histories must remain separate")
    if not raw.empty:
        dates = pd.to_datetime(raw.date_).dt.normalize()
        if bundle is None:
            options = calibration or {}
            bundle = run_analysis(raw, dataset_id=DATASET_ID, dataset_label="WTI crude oil",
                                  source="Databento / CME NYMEX final settlements", **options)
        else:
            latest = bundle.panel.date_.max()
            # Skip immutable overlap. Corrections require a deliberate new
            # analysis version, rather than silently rewriting prior results.
            additions = raw[dates > latest]
            if not additions.empty:
                required = latest + pd.Timedelta(days=1)
                while required <= dates.max():
                    if calendar.is_session(str(required.date())) and required not in set(dates):
                        raise RuntimeError(f"Missing settlement session {required.date()}; no partial catch-up published")
                    required += pd.Timedelta(days=1)
                bundle = append_daily_data(bundle, additions)
    if bundle is None:
        raise RuntimeError("No final settlement curves were available for initialization")
    latest = str(bundle.panel.date_.max().date())
    if latest < expected_date:
        raise RuntimeError(f"Final WTI data is behind schedule: expected {expected_date}, received {latest}")
    target, basis = calendar.next_session(latest)
    # A catch-up run must not claim it issued a forecast before a past target.
    target_cutoff = datetime.combine(datetime.fromisoformat(target).date(), time(13, 30), ZoneInfo("America/Chicago"))
    if now >= target_cutoff:
        raise RuntimeError("Next-session settlement has already passed; wait for the missing final data")
    bundle = append_daily_data(bundle, bundle.panel[bundle.panel.date_ == bundle.panel.date_.max()], next_session=target)
    bundle.metadata["daily_forecast"]["calendar_basis"] = basis
    bundle.metadata["live_feed"] = {
        "provider": "Databento", "dataset": DATASET, "symbol": SYMBOL,
        "mode": "automatic_daily_settlement", "price_kind": "Final actual settlement",
        "history_is_provider_separate": True,
        "expected_observation_date": expected_date,
    }
    return bundle
