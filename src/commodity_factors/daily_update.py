"""Append completed daily curves without refitting or revising old forecasts."""

from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from commodity_factors.analysis import compare_models
from commodity_factors.daily import contract_key, forecast_contract
from commodity_factors.data import prepare_panel


def append_daily_data(bundle, raw, *, next_session=None):
    """Update an analysis from the feed's complete contract settlement curves.

    Overlapping data must match the saved history. Repeated input is safe.
    Provider adapters supply normalized columns and a settlement session date;
    this function never performs MLE or guesses an exchange holiday calendar.
    """
    n = int(bundle.metadata["quality"]["contracts_per_day"])
    incoming = prepare_panel(raw, n=n)
    latest = pd.Timestamp(bundle.panel.date_.max())
    older = incoming[incoming.date_ <= latest]
    if not older.empty:
        keys = ["date_", "maturity_rank"]
        saved = bundle.panel.set_index(keys)
        overlap = older.set_index(keys)
        if not overlap.index.isin(saved.index).all():
            raise ValueError("Daily updates may not insert dates into saved history")
        existing = saved.loc[overlap.index]
        if (not np.allclose(existing.settlement, overlap.settlement, rtol=0, atol=1e-10)
                or not np.array_equal(existing.futcode.map(contract_key), overlap.futcode.map(contract_key))
                or not np.array_equal(existing.lasttrddate, overlap.lasttrddate)):
            raise ValueError("Daily updates may not revise saved historical observations")
    additions = incoming[incoming.date_ > latest]
    if not additions.empty:
        meta = bundle.metadata
        parameters = meta["parameters"]
        bundle = compare_models(
            pd.concat([bundle.panel, additions], ignore_index=True),
            parameters["one_factor"], parameters["two_factor"], parameters["two_factor"]["sigma_epsilon"],
            validation_start=meta["validation_start"], test_start=meta["test_start"],
            horizons=meta["horizons"], dataset_id=meta["dataset_id"], dataset_label=meta["dataset_label"],
            source=meta["source"], calibration=meta["calibration"],
        )
        bundle.metadata["daily_forecast"] = deepcopy(meta.get("daily_forecast", {}))
    else:
        bundle = deepcopy(bundle)
    daily = bundle.metadata.setdefault("daily_forecast", {})
    daily["last_update_utc"] = datetime.now(timezone.utc).isoformat()
    latest = pd.Timestamp(bundle.panel.date_.max())
    if next_session is not None:
        target = pd.Timestamp(next_session)
        if target <= latest:
            raise ValueError("The next settlement session must follow the latest observation")
        daily["next_session"] = str(target.date())
        issued = daily.setdefault("issued_forecasts", [])
        keyed = {(r["origin_date"], r["target_date"], r["futcode"], r["model"]): r for r in issued}
        state = bundle.states[bundle.states.date_ == latest].iloc[0]
        for _, quote in bundle.panel[bundle.panel.date_ == latest].iterrows():
            prices = forecast_contract(state, quote, target, bundle.metadata["parameters"])
            if prices is None:
                continue
            for model, prediction in prices.items():
                key = (str(latest.date()), str(target.date()), contract_key(quote.futcode), model)
                if key in keyed:
                    if not np.isclose(keyed[key]["predicted_log"], prediction, rtol=0, atol=1e-10):
                        raise ValueError("An issued forecast cannot be changed")
                    continue
                record = dict(zip(("origin_date", "target_date", "futcode", "model"), key))
                record.update(predicted_log=prediction, issued_at_utc=daily["last_update_utc"])
                issued.append(record)
    elif daily.get("next_session", "") <= str(latest.date()):
        daily.pop("next_session", None)
    return bundle
