from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from commodity_factors.analysis import compare_models, load_bundle, save_bundle
from commodity_factors.daily import contract_key, daily_snapshot
from commodity_factors.daily_update import append_daily_data
from commodity_factors.demo import DEMO_PARAMS, make_demo_panel


@pytest.fixture(scope="module")
def daily_bundle():
    panel = make_demo_panel(n_ranks=3)
    one = {"mu": 0.02, "sigma": 0.3, "delta": -0.08, "sigma_epsilon": 0.01}
    return compare_models(panel, one, DEMO_PARAMS, 0.004, horizons=(1, 5))


def test_prior_matches_saved_predictions_and_reports_dollar_errors(daily_bundle):
    snapshot = daily_snapshot(daily_bundle, "2025-10-07", rank=1)
    assert snapshot["prior_origin"] == "2025-10-06"
    assert snapshot["target_date"] == "2025-10-08"
    target = pd.Timestamp("2025-10-07")
    saved = daily_bundle.predictions.query("date_ == @target and maturity_rank == 1 and horizon == 1")
    for row in snapshot["models"]:
        expected = saved[saved.model == row["model"]].iloc[0]
        assert row["prior_log"] == pytest.approx(expected.predicted_log, abs=1e-12)
        assert row["error_price"] == pytest.approx(snapshot["observed_price"] - np.exp(expected.predicted_log))
        assert row["error_percent"] == pytest.approx(row["error_price"] / snapshot["observed_price"] * 100)
    assert snapshot["prior_recorded_at"] is None


def test_daily_forecasts_ignore_future_prices_and_current_price_does_not_change_prior(daily_bundle):
    before = daily_snapshot(daily_bundle, "2025-10-07")
    modified = daily_bundle.panel.copy()
    modified.loc[modified.date_ > pd.Timestamp("2025-10-07"), "settlement"] *= 2
    modified["log_settlement"] = np.log(modified.settlement)
    p = daily_bundle.metadata["parameters"]
    altered = compare_models(modified, p["one_factor"], p["two_factor"], 0.004, horizons=(1, 5))
    assert daily_snapshot(altered, "2025-10-07") == before
    modified.loc[modified.date_ == pd.Timestamp("2025-10-07"), "settlement"] *= 1.1
    modified["log_settlement"] = np.log(modified.settlement)
    altered = compare_models(modified, p["one_factor"], p["two_factor"], 0.004, horizons=(1, 5))
    after = daily_snapshot(altered, "2025-10-07")
    assert before["state"] != after["state"]
    for old, new in zip(before["models"], after["models"]):
        assert old["prior_log"] == new["prior_log"]
        assert old["next_log"] != new["next_log"]


def test_contract_identity_survives_roll_and_missing_prior_quote_is_unavailable(daily_bundle):
    snapshot = daily_snapshot(daily_bundle, "2025-01-20")
    origin, target = pd.Timestamp("2025-01-17"), pd.Timestamp("2025-01-20")
    previous = daily_bundle.panel.query("date_ == @origin and maturity_rank == 2").iloc[0]
    assert snapshot["futcode"] == contract_key(previous.futcode)
    assert snapshot["models"][0]["prior_log"] == previous.log_settlement
    rank_wise = daily_bundle.predictions.query("date_ == @target and maturity_rank == 1 and horizon == 1 and model == 'random_walk'").iloc[0]
    assert snapshot["models"][0]["prior_log"] != rank_wise.predicted_log
    new_contract = daily_snapshot(daily_bundle, "2025-01-20", rank=3)
    assert new_contract["models"][0]["prior_price"] is None
    assert all(row["prior_price"] is not None for row in new_contract["models"][1:])


def test_expiry_calibration_and_calendar_boundaries(daily_bundle):
    latest = daily_snapshot(daily_bundle)
    assert latest["calendar_basis"].startswith("Provisional weekday")
    feed = deepcopy(daily_bundle)
    feed.metadata["daily_forecast"] = {"next_session": "2026-01-02"}
    assert daily_snapshot(feed)["target_date"] == "2026-01-02"
    assert daily_snapshot(feed)["calendar_basis"] == "Feed settlement calendar"
    friday = daily_snapshot(feed, "2025-10-10")
    assert friday["target_date"] == "2025-10-13"
    expired = daily_snapshot(feed, target_date=latest["expiry"])
    assert expired["expired_before_target"]
    assert all(row["next_price"] is None for row in expired["models"])
    with pytest.raises(ValueError, match="calibration"):
        daily_snapshot(feed, "2023-06-01")
    with pytest.raises(ValueError, match="after"):
        daily_snapshot(feed, target_date=latest["as_of"])
    with pytest.raises(ValueError, match="present"):
        daily_snapshot(feed, "2025-10-11")


def test_daily_update_preserves_parameters_history_and_issued_forecasts(small_panel, model_params, tmp_path, monkeypatch):
    import commodity_factors.analysis as analysis

    def no_fitting(*args, **kwargs):
        raise AssertionError("The daily update must not recalibrate")

    monkeypatch.setattr(analysis, "estimate_one_factor", no_fitting)
    monkeypatch.setattr(analysis, "estimate_real_parameters", no_fitting)
    original = compare_models(small_panel, *model_params, 0.004, horizons=(1, 3))
    issued = append_daily_data(original, small_panel, next_session="2025-01-15")
    archive = deepcopy(issued.metadata["daily_forecast"]["issued_forecasts"])
    assert len(archive) == 9
    repeated = append_daily_data(issued, small_panel, next_session="2025-01-15")
    assert repeated.metadata["daily_forecast"]["issued_forecasts"] == archive
    target = pd.Timestamp("2025-01-15")
    incoming = make_demo_panel(n_ranks=3).query("date_ == @target")
    updated = append_daily_data(issued, incoming, next_session="2025-01-16")
    assert updated.metadata["parameters"] == original.metadata["parameters"]
    pd.testing.assert_frame_equal(updated.states.iloc[:-1].reset_index(drop=True), original.states)
    prior_predictions = updated.predictions[updated.predictions.date_ <= small_panel.date_.max()].reset_index(drop=True)
    pd.testing.assert_frame_equal(prior_predictions, original.predictions)
    assert updated.metadata["daily_forecast"]["issued_forecasts"][:9] == archive
    snapshot = daily_snapshot(updated)
    assert snapshot["prior_recorded_at"] == archive[0]["issued_at_utc"]
    assert snapshot["prior_origin"] == "2025-01-14"
    for row in snapshot["models"]:
        record = next(r for r in archive if r["model"] == row["model"] and r["futcode"] == snapshot["futcode"])
        assert row["prior_log"] == record["predicted_log"]
    save_bundle(updated, tmp_path)
    assert daily_snapshot(load_bundle(tmp_path)) == snapshot
    revised = incoming.copy()
    revised["settlement"] *= 1.01
    with pytest.raises(ValueError, match="revise"):
        append_daily_data(updated, revised, next_session="2025-01-16")
    revised_records = deepcopy(issued)
    revised_records.metadata["daily_forecast"]["issued_forecasts"][0]["predicted_log"] += 0.01
    with pytest.raises(ValueError, match="cannot be changed"):
        append_daily_data(revised_records, small_panel, next_session="2025-01-15")
    with pytest.raises(ValueError, match="follow"):
        append_daily_data(updated, incoming, next_session="2025-01-15")


def test_numeric_contract_identifiers_are_stable_across_feed_serialization(small_panel, model_params):
    panel = small_panel.copy()
    panel["futcode"] = panel.futcode.astype(float)
    bundle = compare_models(panel, *model_params, 0.004, horizons=(1,))
    updated = append_daily_data(bundle, small_panel, next_session="2025-01-15")
    snapshot = daily_snapshot(updated)
    assert "." not in snapshot["futcode"]
    assert all("." not in row["futcode"] for row in updated.metadata["daily_forecast"]["issued_forecasts"])
