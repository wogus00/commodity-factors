from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pandas as pd
import pytest

from commodity_factors.analysis import compare_models
from commodity_factors.blob_store import BlobStore, LiveRepository, pack_bundle, unpack_bundle
from commodity_factors.daily_update import append_daily_data
from commodity_factors.demo import make_demo_panel
from commodity_factors.live_feed import DatabentoFeed, DATASET_ID, normalize_settlements
from commodity_factors.live_update import expected_observation, request_window, update_from_feed
from commodity_factors.settlement_calendar import SettlementCalendar


@pytest.fixture
def calendar():
    return SettlementCalendar(json.loads((Path(__file__).parents[1] / "configs/settlement_calendar.json").read_text()))


@pytest.fixture
def bundle(small_panel, model_params):
    return compare_models(small_panel, *model_params, 0.004, horizons=(1, 3), dataset_id=DATASET_ID)


def records():
    definitions = pd.DataFrame([
        {"instrument_id": i, "instrument_class": "F", "raw_symbol": f"CL{month}26", "expiration": f"2026-{i:02}-20T19:30:00Z"}
        for i, month in enumerate("FGHJKMN", start=1)
    ] + [{"instrument_id": 99, "instrument_class": "S", "raw_symbol": "CLF26-CLG26", "expiration": "2026-01-20T19:30:00Z"}])
    stats = pd.DataFrame([
        {"instrument_id": i, "ts_ref": "2026-01-02T00:00:00Z", "ts_event": "2026-01-03T00:30:00Z",
         "stat_type": 3, "stat_flags": 3, "price": 70.0 + i, "update_action": 1} for i in range(1, 8)
    ])
    return definitions, stats


def test_final_prices_use_reference_date_and_nearest_complete_curve():
    definitions, stats = records()
    preliminary = stats.iloc[:1].copy()
    preliminary["price"] = 999
    preliminary["stat_flags"] = 2
    preliminary["ts_event"] = "2026-01-02T20:00:00Z"
    normalized = normalize_settlements(definitions, pd.concat([preliminary, stats]), start_date="2026-01-02", end_date="2026-01-03")
    assert normalized.date_.unique().tolist() == [pd.Timestamp("2026-01-02")]
    assert len(normalized) == 6
    assert normalized.iloc[0].settlement == 71
    assert normalized.iloc[0].futcode == "CL-2026-01-20"
    for invalid in (stats[stats.instrument_id != 1], stats.assign(stat_flags=[11, 3, 3, 3, 3, 3, 3]),
                    stats.assign(update_action=[2, 1, 1, 1, 1, 1, 1])):
        with pytest.raises(ValueError, match="Incomplete"):
            normalize_settlements(definitions, invalid, start_date="2026-01-02", end_date="2026-01-03")
    with pytest.raises(ValueError, match="positive"):
        normalize_settlements(definitions, stats.assign(price=[-1, 72, 73, 74, 75, 76, 77]), start_date="2026-01-02", end_date="2026-01-03")


def test_budget_checks_both_requests_before_any_paid_call():
    calls = []
    def cost(**request):
        calls.append(request["schema"])
        return 0.6
    def no_prices(**kwargs):
        raise AssertionError("No timeseries call may occur over budget")
    feed = DatabentoFeed(SimpleNamespace(metadata=SimpleNamespace(get_cost=cost), timeseries=SimpleNamespace(get_range=no_prices)))
    with pytest.raises(RuntimeError, match="no price data requested"):
        feed.fetch("2026-01-01", "2026-01-04", start_date="2026-01-01", end_date="2026-01-03", max_cost_usd=1)
    assert calls == ["definition", "statistics"]


def test_adapter_uses_current_sdk_float_prices():
    pytest.importorskip("databento")
    from databento.common.dbnstore import DBNStore
    import inspect
    parameters = inspect.signature(DBNStore.to_df).parameters
    assert {"price_type", "pretty_ts", "map_symbols"}.issubset(parameters)
    definitions, stats = records()
    iterator = iter([definitions, stats])
    class Store:
        def to_df(self, *, price_type, pretty_ts, map_symbols):
            assert (price_type, pretty_ts, map_symbols) == ("float", True, False)
            return next(iterator)
    client = SimpleNamespace(metadata=SimpleNamespace(get_cost=lambda **_: 0), timeseries=SimpleNamespace(get_range=lambda **_: Store()))
    panel, cost = DatabentoFeed(client).fetch("2026-01-01", "2026-01-04", start_date="2026-01-01", end_date="2026-01-03", max_cost_usd=0)
    assert len(panel) == 6 and cost == 0


def test_calendar_skips_settlement_holidays_and_labels_projections(calendar):
    assert calendar.next_session("2026-07-02") == ("2026-07-06", "CME NYMEX settlement calendar")
    assert calendar.next_session("2026-04-02")[0] == "2026-04-06"
    assert calendar.next_session("2026-10-09")[0] == "2026-10-12"  # Columbus Day is a settlement session.
    assert calendar.next_session("2026-12-31")[0] == "2027-01-04"
    assert calendar.next_session("2026-12-31")[1].startswith("Projected")
    assert expected_observation(calendar, datetime(2026, 7, 5, 1, tzinfo=timezone.utc), 0) == "2026-07-02"


def test_catchup_preserves_issued_forecasts_and_rejects_gaps(bundle, calendar):
    bundle = append_daily_data(bundle, bundle.panel, next_session="2025-01-15")
    issued = deepcopy(bundle.metadata["daily_forecast"]["issued_forecasts"])
    raw = make_demo_panel(n_ranks=3).query("'2025-01-15' <= date_ <= '2025-01-16'")
    now = datetime(2025, 1, 17, 1, tzinfo=timezone.utc)
    updated = update_from_feed(bundle, raw, calendar=calendar, now=now, expected_date="2025-01-16")
    assert updated.metadata["parameters"] == bundle.metadata["parameters"]
    assert updated.metadata["daily_forecast"]["issued_forecasts"][:len(issued)] == issued
    assert {r["origin_date"] for r in updated.metadata["daily_forecast"]["issued_forecasts"]} == {"2025-01-14", "2025-01-16"}
    assert updated.metadata["daily_forecast"]["next_session"] == "2025-01-17"
    with pytest.raises(RuntimeError, match="Missing settlement"):
        update_from_feed(bundle, raw[raw.date_ == pd.Timestamp("2025-01-16")], calendar=calendar, now=now, expected_date="2025-01-16")
    with pytest.raises(RuntimeError, match="behind schedule"):
        update_from_feed(bundle, pd.DataFrame(), calendar=calendar, now=now, expected_date="2025-01-16")


def test_no_forecast_is_recorded_after_target_settlement(bundle, calendar):
    with pytest.raises(RuntimeError, match="already passed"):
        update_from_feed(bundle, pd.DataFrame(), calendar=calendar, now=datetime(2025, 1, 15, 20, tzinfo=timezone.utc), expected_date="2025-01-14")


def test_request_window_catches_up_and_obeys_publication_delay(bundle):
    start, end = request_window(bundle, now=datetime(2025, 1, 20, tzinfo=timezone.utc), available_end="2025-01-21T00:00:00Z", publication_delay_hours=24)
    assert start == pd.Timestamp("2025-01-11", tz="UTC")
    assert end == pd.Timestamp("2025-01-19", tz="UTC")


class FakeBlob:
    class BlobNotFoundError(Exception):
        pass
    def __init__(self, access):
        self.access, self.objects, self.fail_pointer = access, {}, False
    def get(self, path, **kwargs):
        if path not in self.objects:
            raise self.BlobNotFoundError()
        return SimpleNamespace(content=self.objects[path], status_code=200)
    def put(self, path, content, **kwargs):
        if self.fail_pointer and path.endswith("latest.json"):
            raise RuntimeError("Simulated upload failure")
        assert kwargs["access"] == self.access
        url = f"https://test.{self.access}.blob.vercel-storage.com/{path}"
        self.objects[path] = self.objects[url] = content
        return SimpleNamespace(url=url)
    def list_objects(self, **kwargs):
        return SimpleNamespace(blobs=[], has_more=False)


def test_blob_archive_and_failed_publish_retain_last_public_version(bundle):
    pd.testing.assert_frame_equal(unpack_bundle(pack_bundle(bundle)).panel, bundle.panel)
    invalid = BytesIO()
    with ZipFile(invalid, "w") as archive:
        archive.writestr("../secret", "invalid")
    with pytest.raises(ValueError, match="Invalid"):
        unpack_bundle(invalid.getvalue())
    private, public = FakeBlob("private"), FakeBlob("public")
    repository = LiveRepository(BlobStore("private-token", "private", sdk=private), BlobStore("public-token", "public", sdk=public))
    assert repository.load() is None
    with pytest.raises(ValueError, match="display rights"):
        repository.publish(bundle, display_approved=False)
    assert not public.objects and not private.objects
    bundle = append_daily_data(bundle, bundle.panel, next_session="2025-01-15")
    manifest_url = repository.publish(bundle, display_approved=True)
    original = public.objects[manifest_url]
    public.fail_pointer = True
    with pytest.raises(RuntimeError, match="upload failure"):
        repository.publish(bundle, display_approved=True)
    assert public.objects[manifest_url] == original
    assert repository.load().metadata == bundle.metadata
    public.fail_pointer = False
    assert repository.publish(bundle, display_approved=True) == manifest_url


def test_wrong_provider_cannot_be_published(bundle):
    bundle.metadata["dataset_id"] = "wti"
    repository = LiveRepository(BlobStore("p", "private", sdk=FakeBlob("private")), BlobStore("q", "public", sdk=FakeBlob("public")))
    with pytest.raises(ValueError, match="Databento"):
        repository.publish(bundle, display_approved=True)
