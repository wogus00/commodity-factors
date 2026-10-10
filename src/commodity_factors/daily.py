"""Daily forecast snapshots using fixed parameters and causal filtered states."""

from numbers import Real

import numpy as np
import pandas as pd

from commodity_factors.evaluation import forecast_h_steps


def contract_key(value):
    """Keep numeric identifiers stable across CSV, Parquet and JSON exports."""
    if isinstance(value, Real) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value)


def forecast_contract(state, quote, target_date, parameters):
    """Forecast the held contract; never read prices after the origin."""
    origin = pd.Timestamp(quote["date_"])
    target = pd.Timestamp(target_date)
    if target <= origin:
        raise ValueError("The forecast target must be after the observation date")
    expiry = pd.Timestamp(quote["lasttrddate"])
    if target >= expiry:
        return None
    elapsed = (target - origin).days / 365.25
    tau = (expiry - target).days / 365.25
    one = parameters["one_factor"]
    return {
        "random_walk": float(quote["log_settlement"]),
        "one_factor": float(state["one_factor_state"] + one["mu"] * elapsed
                            + (one["delta"] + 0.5 * one["sigma"] ** 2) * tau),
        "two_factor": float(forecast_h_steps(
            [state["chi"], state["xi"]], [tau], parameters["two_factor"], elapsed)[0]),
    }


def daily_snapshot(bundle, as_of=None, rank=1, target_date=None):
    """Observed settlement, prior forecast/error, current state, next forecast.

    Unlike the rank-wise backtest, this view holds the selected contract fixed
    across the three dates. Missing prior contract quotes remain unavailable.
    The next recorded session is preferred; beyond the data a weekday is only
    a provisional target unless the feed supplies its next settlement session.
    """
    dates = pd.DatetimeIndex(sorted(bundle.panel.date_.unique()))
    as_of = pd.Timestamp(as_of if as_of is not None else dates[-1])
    if as_of not in dates:
        raise ValueError("Choose an observation date present in the dataset")
    if as_of < pd.Timestamp(bundle.metadata["train_end"]):
        raise ValueError("Daily forecasts start when calibration is complete")
    curve = bundle.panel[bundle.panel.date_ == as_of]
    selection = curve[curve.maturity_rank == rank]
    if selection.empty:
        raise ValueError("Choose an available maturity rank")
    quote = selection.iloc[0]
    states = bundle.states.set_index("date_")
    state = states.loc[as_of]
    index = dates.get_loc(as_of)
    previous_date = dates[index - 1] if index else None
    prior = None
    if previous_date is not None and previous_date >= pd.Timestamp(bundle.metadata["train_end"]):
        previous = bundle.panel[(bundle.panel.date_ == previous_date)
                                & (bundle.panel.futcode.map(contract_key) == contract_key(quote.futcode))]
        if not previous.empty:
            prior = forecast_contract(states.loc[previous_date], previous.iloc[0], as_of,
                                      bundle.metadata["parameters"])
        else:
            # The structural models can price a newly included contract, but
            # a same-contract random walk needs a previous quote for it.
            one = bundle.metadata["parameters"]["one_factor"]
            p = states.loc[previous_date]
            gap = (as_of - previous_date).days / 365.25
            prior = {
                "random_walk": None,
                "one_factor": float(p.one_factor_state + one["mu"] * gap
                                    + (one["delta"] + 0.5 * one["sigma"] ** 2) * quote.tau),
                "two_factor": float(forecast_h_steps([p.chi, p.xi], [quote.tau],
                                                     bundle.metadata["parameters"]["two_factor"], gap)[0]),
            }
    feed_target = bundle.metadata.get("daily_forecast", {}).get("next_session")
    issued = bundle.metadata.get("daily_forecast", {}).get("issued_forecasts", [])
    records = [r for r in issued if r["target_date"] == str(as_of.date())
               and r["futcode"] == contract_key(quote.futcode) and r["origin_date"] < str(as_of.date())]
    recorded_at = None
    if records:
        origin = max(r["origin_date"] for r in records)
        recorded = {r["model"]: r for r in records if r["origin_date"] == origin}
        if set(recorded) == {"random_walk", "one_factor", "two_factor"}:
            prior = {model: r["predicted_log"] for model, r in recorded.items()}
            previous_date = pd.Timestamp(origin)
            recorded_at = recorded["two_factor"]["issued_at_utc"]
    if target_date is not None:
        target, basis = pd.Timestamp(target_date), "Selected target date"
    elif index + 1 < len(dates):
        target, basis = dates[index + 1], "Next recorded session"
    elif feed_target:
        target = pd.Timestamp(feed_target)
        basis = bundle.metadata.get("daily_forecast", {}).get("calendar_basis", "Feed settlement calendar")
    else:
        target, basis = as_of + pd.offsets.BDay(1), "Provisional weekday; exchange holidays not checked"
    future = forecast_contract(state, quote, target, bundle.metadata["parameters"])
    observed = float(quote.settlement)
    label = quote.get("dsmnem")
    if pd.isna(label):
        label = contract_key(quote.futcode)
    models = []
    for model in ("random_walk", "one_factor", "two_factor"):
        prior_log = prior.get(model) if prior else None
        prior_price = float(np.exp(prior_log)) if prior_log is not None else None
        future_log = future[model] if future else None
        models.append({
            "model": model, "prior_log": prior_log, "prior_price": prior_price,
            "error_price": observed - prior_price if prior_price is not None else None,
            "error_percent": (observed - prior_price) / observed * 100 if prior_price is not None else None,
            "next_log": future_log,
            "next_price": float(np.exp(future_log)) if future_log is not None else None,
        })
    return {
        "as_of": str(as_of.date()), "prior_origin": str(previous_date.date()) if prior else None,
        "prior_recorded_at": recorded_at,
        "target_date": str(target.date()), "calendar_basis": basis,
        "futcode": contract_key(quote.futcode), "contract_label": str(label),
        "expiry": str(pd.Timestamp(quote.lasttrddate).date()), "rank": int(rank),
        "observed_price": observed, "observed_log": float(quote.log_settlement),
        "state": {key: float(state[key]) for key in ("one_factor_state", "chi", "xi")},
        "models": models, "expired_before_target": future is None,
    }
