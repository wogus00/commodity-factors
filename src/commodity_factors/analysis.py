"""Training-only model comparison and portable dashboard result bundles."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from commodity_factors.data import prepare_kalman_data, validate_panel
from commodity_factors.estimation import estimate_real_parameters
from commodity_factors.evaluation import forecast_h_steps
from commodity_factors.kalman import kalman_filter
from commodity_factors.one_factor import estimate_one_factor, one_factor_kalman_filter
from commodity_factors.schwartz_smith import futures_log_price


MODEL_LABELS = {"random_walk": "Random walk", "one_factor": "One factor", "two_factor": "Two factors"}
TABLES = ("panel", "predictions", "reconstructions", "states", "metrics")
SCHEMA_VERSION = 1


@dataclass
class AnalysisBundle:
    panel: pd.DataFrame
    predictions: pd.DataFrame
    reconstructions: pd.DataFrame
    states: pd.DataFrame
    metrics: pd.DataFrame
    metadata: dict


def split_labels(dates, validation_start, test_start):
    dates = pd.DatetimeIndex(dates)
    if pd.Timestamp(validation_start) >= pd.Timestamp(test_start):
        raise ValueError("Validation must start before the test period")
    return np.where(dates < pd.Timestamp(validation_start), "train",
                    np.where(dates < pd.Timestamp(test_start), "validation", "test"))


def summarize_predictions(predictions):
    """Compute log- and price-level errors; rank 0 represents pooled observations."""
    rows = []
    group_keys = ["split", "horizon", "model"]
    for key, group in predictions.groupby(group_keys, sort=True):
        for rank, observations in [(0, group), *group.groupby("maturity_rank", sort=True)]:
            errors = observations["error_log"].to_numpy(dtype=float)
            price_errors = (np.exp(observations["observed_log"])
                            - np.exp(observations["predicted_log"]))
            rows.append(dict(zip(group_keys, key)) | {
                "maturity_rank": int(rank),
                "rmse_log": float(np.sqrt(np.mean(errors ** 2))),
                "mae_log": float(np.mean(np.abs(errors))),
                "rmse_price": float(np.sqrt(np.mean(price_errors ** 2))),
                "mae_price": float(np.mean(np.abs(price_errors))),
                "n": len(observations), "n_dates": observations["date_"].nunique(),
            })
    return pd.DataFrame(rows)


def _validate_predictions(predictions):
    keys = ["model", "horizon", "date_", "maturity_rank"]
    if predictions.empty or predictions.duplicated(keys).any():
        raise ValueError("Forecasts must be nonempty and uniquely keyed")
    if not np.isfinite(predictions[["observed_log", "predicted_log", "error_log"]]).all().all():
        raise ValueError("Forecasts must be finite")
    if not (predictions["origin_date"] < predictions["date_"]).all():
        raise ValueError("Forecast origins must precede target dates")
    coverage = predictions.groupby(["horizon", "date_", "maturity_rank"])["model"].nunique()
    if not (coverage == len(MODEL_LABELS)).all():
        raise ValueError("All models must be scored on the same sample")


def compare_models(panel, one_params, two_params, sigma_epsilon, *,
                   validation_start="2024-01-01", test_start="2025-01-01",
                   horizons=(1, 5, 20), dataset_id="wti", dataset_label="WTI crude oil",
                   source="WRDS / LSEG Datastream", calibration=None):
    """Forecast fixed models from each origin's filtered state.

    Random walk repeats the origin's rank-wise log curve. Structural forecasts
    use target maturities from contract metadata. Contract rolls are part of
    this rank-based task; this is not a same-contract random-walk benchmark.
    """
    panel = panel.copy()
    panel["date_"] = pd.to_datetime(panel["date_"])
    panel = panel.sort_values(["date_", "maturity_rank"]).reset_index(drop=True)
    for column in ("settlement", "log_settlement", "tau"):
        panel[column] = panel[column].astype(float)
    n_ranks = panel["maturity_rank"].nunique()
    quality = validate_panel(panel, n=n_ranks)
    dates, Y, tau, dt = prepare_kalman_data(panel)
    splits = split_labels(dates, validation_start, test_start)
    n_train = int(np.sum(splits == "train"))
    if n_train < 2 or not np.any(splits == "validation") or not np.any(splits == "test"):
        raise ValueError("The panel must cover training, validation, and test periods")
    if not horizons or any(not isinstance(h, (int, np.integer)) or h < 1 for h in horizons):
        raise ValueError("Horizons must be positive integers")
    horizons = sorted(set(int(h) for h in horizons))
    if max(horizons) > len(dates) - n_train:
        raise ValueError("Horizons exceed the available holdout history")

    one = one_factor_kalman_filter(Y, tau, dt, one_params)
    two = kalman_filter(Y, tau, two_params, dt=dt, sigma_epsilon=sigma_epsilon)
    observations = panel.set_index(["date_", "maturity_rank"])
    frames = []
    for h in horizons:
        # Fitted parameters are only available at the last training date.
        origins = np.arange(n_train - 1, len(dates) - h)
        targets = origins + h
        elapsed = (dates[targets] - dates[origins]).days.to_numpy() / 365.25
        predicted = {
            "random_walk": Y[origins],
            "one_factor": (
                one["filtered_states"][origins, None] + one_params["mu"] * elapsed[:, None]
                + (one_params["delta"] + 0.5 * one_params["sigma"] ** 2) * tau[targets]
            ),
            "two_factor": np.stack([
                forecast_h_steps(two["filtered_states"][origin], tau[target], two_params, gap)
                for origin, target, gap in zip(origins, targets, elapsed)
            ]),
        }
        index = pd.MultiIndex.from_product([dates[targets], range(1, n_ranks + 1)],
                                          names=["date_", "maturity_rank"])
        base = observations.reindex(index)[["futcode", "tau", "log_settlement"]].reset_index()
        base = base.rename(columns={"log_settlement": "observed_log"})
        base["origin_date"] = np.repeat(dates[origins].to_numpy(), n_ranks)
        base["split"] = np.repeat(splits[targets], n_ranks)
        base["horizon"] = h
        base["elapsed_years"] = np.repeat(elapsed, n_ranks)
        for model, values in predicted.items():
            frame = base.copy()
            frame["model"] = model
            frame["predicted_log"] = values.ravel()
            frame["error_log"] = frame["observed_log"] - frame["predicted_log"]
            frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    _validate_predictions(predictions)

    fitted_two = np.stack([futures_log_price(x, maturity, two_params)
                           for x, maturity in zip(two["filtered_states"], tau)])
    fitted_one = (one["filtered_states"][:, None]
                  + (one_params["delta"] + 0.5 * one_params["sigma"] ** 2) * tau)
    reconstruction_frames = []
    for model, values in (("one_factor", fitted_one), ("two_factor", fitted_two)):
        frame = panel[["date_", "maturity_rank", "tau", "log_settlement"]].copy()
        frame = frame.rename(columns={"log_settlement": "observed_log"})
        frame["model"] = model
        frame["fitted_log"] = values.ravel()
        frame["split"] = np.repeat(splits, n_ranks)
        reconstruction_frames.append(frame)
    states = pd.DataFrame({
        "date_": dates, "split": splits,
        "chi": two["filtered_states"][:, 0], "xi": two["filtered_states"][:, 1],
        "one_factor_state": one["filtered_states"],
        "one_factor_predictive_ll": one["log_likelihoods"],
        "two_factor_predictive_ll": two["log_likelihoods"],
    })
    quality = {key: float(value) if isinstance(value, np.floating) else int(value)
               for key, value in quality.items()}
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id, "dataset_label": dataset_label, "source": source,
        "price_unit": "USD per barrel", "target": "maturity_rank",
        "validation_start": str(pd.Timestamp(validation_start).date()),
        "test_start": str(pd.Timestamp(test_start).date()),
        "train_end": str(dates[n_train - 1].date()), "horizons": horizons,
        "quality": quality, "models": MODEL_LABELS,
        "parameters": {
            "one_factor": {k: float(v) for k, v in one_params.items()},
            "two_factor": {k: float(v) for k, v in two_params.items()}
                          | {"sigma_epsilon": float(sigma_epsilon)},
        },
        "calibration": calibration or {},
        "assumptions": [
            "One factor uses a Brownian log-price state with physical drift, not an OU state.",
            "Random walk repeats the origin's maturity-rank curve; ranks can roll to new contracts.",
            "Maturities use last trading date and calendar years of 365.25 days.",
            "Forecast horizons count observations in the supplied trading-date panel.",
            "States use observations through the origin; parameters stay fixed after training.",
            "Exponentiated log forecasts are conditional medians, not price-level conditional means.",
            "Filtered factors are model estimates. Same-day reconstruction is separate from prediction.",
            "Initial states use the first training curve and a fixed prior covariance; initial-day predictions are not scored.",
        ],
    }
    return AnalysisBundle(panel, predictions, pd.concat(reconstruction_frames, ignore_index=True),
                          states, summarize_predictions(predictions), metadata)


def _fit_diagnostics(fit):
    result = fit["optimizer_result"]
    return {"success": bool(fit["success"]), "message": str(fit["message"]),
            "training_log_likelihood": float(fit["log_likelihood"]),
            "iterations": int(result.nit), "evaluations": int(result.nfev)}


def run_analysis(panel, *, validation_start="2024-01-01", test_start="2025-01-01",
                 horizons=(1, 5, 20), dataset_id="wti", dataset_label="WTI crude oil",
                 source="WRDS / LSEG Datastream", maxiter=300, progress=None):
    """Fit on training data only; require successful fits before saving results."""
    dates, Y, tau, dt = prepare_kalman_data(panel)
    splits = split_labels(dates, validation_start, test_start)
    n_train = int(np.sum(splits == "train"))
    if n_train < 2 or not np.any(splits == "validation") or not np.any(splits == "test"):
        raise ValueError("The panel must cover training, validation, and test periods")
    report = progress or (lambda message: None)
    report(f"Fitting one factor on {n_train} training dates...")
    one = estimate_one_factor(Y[:n_train], tau[:n_train], dt[:n_train - 1], maxiter=maxiter)
    if not one["success"]:
        raise RuntimeError(f"One-factor fit did not converge: {one['message']}")
    report("Fitting two factors with lambda_chi fixed at zero...")
    fits = []
    for kappa in (1.5, 0.3):
        fit = estimate_real_parameters(
            Y[:n_train], tau[:n_train], dt[:n_train - 1],
            initial_guess=[kappa, 0.3, 0.2, 0.0, 0.02, 0.0, -0.05, 0.01],
            fixed_lambda_chi=0.0, maxiter=maxiter,
        )
        fits.append(fit)
        report(f"Two-factor start kappa={kappa}: {fit['message']}")
    successful = [fit for fit in fits if fit["success"] and np.isfinite(fit["log_likelihood"])]
    if not successful:
        raise RuntimeError("Two-factor fits did not converge: "
                           + "; ".join(str(fit["message"]) for fit in fits))
    two = max(successful, key=lambda fit: fit["log_likelihood"])
    report("Building aligned forecasts and evaluation tables...")
    return compare_models(
        panel, one["params"], two["params"], two["sigma_epsilon"],
        validation_start=validation_start, test_start=test_start, horizons=horizons,
        dataset_id=dataset_id, dataset_label=dataset_label, source=source,
        calibration={"method": "Training-only maximum likelihood",
                     "one_factor": _fit_diagnostics(one), "two_factor": _fit_diagnostics(two),
                     "two_factor_starts": [_fit_diagnostics(fit) for fit in fits],
                     "lambda_chi_fixed": 0.0},
    )


def save_bundle(bundle, directory):
    """Save Parquet tables and JSON metadata without serializing optimizer objects."""
    _validate_predictions(bundle.predictions)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        getattr(bundle, name).to_parquet(directory / f"{name}.parquet", index=False)
    (directory / "metadata.json").write_text(json.dumps(bundle.metadata, indent=2, allow_nan=False) + "\n")
    return directory


def load_bundle(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported analysis bundle schema")
    tables = {name: pd.read_parquet(directory / f"{name}.parquet") for name in TABLES}
    _validate_predictions(tables["predictions"])
    return AnalysisBundle(**tables, metadata=metadata)
