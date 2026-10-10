"""Databento's official final daily WTI settlements, separate from WRDS data."""

import math

import numpy as np
import pandas as pd

from commodity_factors.data import prepare_panel


DATASET = "GLBX.MDP3"
SYMBOL = "CL.FUT"
DATASET_ID = "databento-wti"


def normalize_settlements(definitions, statistics, *, start_date, end_date, n=6):
    """Inputs must use SDK float prices/pretty_ts, not fixed-point raw prices.

    CME ts_ref is a UTC date label. Do not convert it to US local time.
    Settlement flags: bit 0 final, bit 1 actual, bit 3 intraday (excluded).
    Reject an incomplete near-month curve instead of promoting distant months.
    """
    columns = ["date_", "futcode", "lasttrddate", "settlement", "dsmnem"]
    if statistics.empty:
        return pd.DataFrame(columns=columns)
    definitions = definitions.copy()
    required = {"instrument_id", "instrument_class", "raw_symbol", "expiration"}
    if not required.issubset(definitions):
        raise ValueError("Instrument definitions are missing required fields")
    definitions = definitions[
        definitions.instrument_class.astype(str).eq("F")
        & definitions.raw_symbol.astype(str).str.fullmatch(r"CL[FGHJKMNQUVXZ]\d{1,2}")
    ].copy()
    definitions["lasttrddate"] = pd.to_datetime(definitions.expiration, utc=True).dt.tz_localize(None).dt.normalize()
    if definitions.groupby("instrument_id").lasttrddate.nunique().gt(1).any():
        raise ValueError("Instrument IDs with multiple expiries require a shorter request range")
    definitions = definitions.drop_duplicates("instrument_id", keep="last")
    required_stats = {"instrument_id", "ts_ref", "ts_event", "stat_type", "stat_flags", "price", "update_action"}
    statistics = statistics.reset_index() if "ts_event" not in statistics else statistics.copy()
    if not required_stats.issubset(statistics):
        raise ValueError("Settlement statistics are missing required fields")
    statistics = statistics[statistics.stat_type.eq(3)].copy()
    statistics["date_"] = pd.to_datetime(statistics.ts_ref, utc=True).dt.tz_localize(None).dt.normalize()
    statistics = statistics[statistics.date_.between(pd.Timestamp(start_date), pd.Timestamp(end_date))]
    statistics["ts_event"] = pd.to_datetime(statistics.ts_event, utc=True)
    # Select the latest message first, so deletions cannot resurrect old values.
    statistics = statistics.sort_values("ts_event", kind="stable").drop_duplicates(["date_", "instrument_id"], keep="last")
    flags = statistics.stat_flags.astype("int64")
    statistics = statistics[(flags & 3).eq(3) & (flags & 8).eq(0) & statistics.update_action.eq(1)]
    rows = statistics.merge(definitions[["instrument_id", "raw_symbol", "lasttrddate"]], on="instrument_id", validate="many_to_one")
    rows = rows[rows.lasttrddate > rows.date_].copy()
    if rows.empty:
        return pd.DataFrame(columns=columns)
    selected = []
    for day, group in rows.groupby("date_", sort=True):
        expected = definitions[definitions.lasttrddate > day].sort_values("lasttrddate").head(n)
        if len(expected) != n or not set(expected.instrument_id).issubset(group.instrument_id):
            raise ValueError(f"Incomplete final WTI settlement curve for {day.date()}")
        group = group[group.instrument_id.isin(expected.instrument_id)].copy()
        values = group.price.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Log-price models require positive finite settlements")
        selected.append(group)
    rows = pd.concat(selected, ignore_index=True)
    rows["futcode"] = "CL-" + rows.lasttrddate.dt.strftime("%Y-%m-%d")
    rows = rows.rename(columns={"price": "settlement", "raw_symbol": "dsmnem"})
    return prepare_panel(rows[columns], n=n)


class DatabentoFeed:
    def __init__(self, client):
        self.client = client

    def available_end(self):
        info = self.client.metadata.get_dataset_range(dataset=DATASET)
        # Schema-level availability can lag the aggregate dataset range.
        schemas = info.get("schema", {})
        ends = [info["end"]]
        for schema in ("definition", "statistics"):
            if schema in schemas:
                ends.append(schemas[schema]["end"])
        return min(pd.Timestamp(value) for value in ends)

    def requests(self, start, end):
        common = dict(dataset=DATASET, symbols=[SYMBOL], stype_in="parent", start=start, end=end)
        return [common | {"schema": schema} for schema in ("definition", "statistics")]

    def estimate(self, start, end):
        costs = [float(self.client.metadata.get_cost(**request)) for request in self.requests(start, end)]
        if any(not math.isfinite(cost) or cost < 0 for cost in costs):
            raise ValueError("Provider returned an invalid cost estimate")
        return sum(costs)

    def fetch(self, start, end, *, start_date, end_date, max_cost_usd, n=6):
        budget = float(max_cost_usd)
        if not math.isfinite(budget) or budget < 0:
            raise ValueError("Set a finite, nonnegative per-run data budget")
        cost = self.estimate(start, end)
        if cost > budget:
            raise RuntimeError(f"Estimated data cost ${cost:.4f} exceeds the ${budget:.4f} per-run limit; no price data requested")
        frames = [self.client.timeseries.get_range(**request).to_df(
            pretty_ts=True, price_type="float", map_symbols=False) for request in self.requests(start, end)]
        return normalize_settlements(*frames, start_date=start_date, end_date=end_date, n=n), cost
