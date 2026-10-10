
import numpy as np
import pandas as pd

from commodity_factors.estimation import estimate_real_parameters
from commodity_factors.kalman import kalman_filter


def evaluate_out_of_sample(
    Y,
    tau,
    dt,
    dates,
    test_start="2025-01-01",
):
    """
    Train Schwartz-Smith on dates before test_start,
    then evaluate one-step-ahead predictions afterward.
    """

    Y = np.asarray(Y, dtype=float)
    tau = np.asarray(tau, dtype=float)
    dt = np.asarray(dt, dtype=float)
    dates = pd.DatetimeIndex(dates)

    # 1. Split data chronologically
    n_train = int((dates < pd.Timestamp(test_start)).sum())

    if not 1 < n_train < len(Y):
        raise ValueError("Train and test periods must be nonempty")

    Y_train = Y[:n_train]
    tau_train = tau[:n_train]
    dt_train = dt[:n_train - 1]

    # 2. Estimate parameters using training data ONLY
    fit = estimate_real_parameters(
        Y=Y_train,
        tau=tau_train,
        dt=dt_train,
    )

    if not fit["success"]:
        raise RuntimeError(
            f"MLE did not converge: {fit['message']}"
        )

    estimated_params = fit["params"]
    sigma_epsilon = fit["sigma_epsilon"]

    # 3. Run Kalman filter forward with FIXED parameters
    # Running from the beginning carries the training
    # state into the test period.
    results = kalman_filter(
        log_prices=Y,
        tau=tau,
        params=estimated_params,
        dt=dt,
        sigma_epsilon=sigma_epsilon,
    )

    # 4. Extract one-step-ahead test errors
    # Innovations are computed BEFORE each day's update.
    test_errors = results["innovations"][n_train:]

    # 5. Evaluate predictions
    rmse = np.sqrt(np.mean(test_errors**2, axis=0))
    mae = np.mean(np.abs(test_errors), axis=0)

    return {
        "fit": fit,
        "params": estimated_params,
        "kalman_results": results,
        "train_end": dates[n_train - 1],
        "test_dates": dates[n_train:],
        "test_errors": test_errors,
        "rmse": rmse,
        "mae": mae,
        "overall_rmse": np.sqrt(np.mean(test_errors**2)),
    }



from commodity_factors.schwartz_smith import futures_log_price


def forecast_h_steps(
    filtered_state,
    target_tau,
    params,
    elapsed_years,
):
    """
    Forecast log futures prices h trading days ahead
    without using intermediate observations.
    """

    chi, xi = filtered_state

    # Forecast the hidden factors
    chi_forecast = (
        np.exp(-params["kappa"] * elapsed_years) * chi
    )

    xi_forecast = (
        xi + params["mu_xi"] * elapsed_years
    )

    predicted_state = np.array([
        chi_forecast,
        xi_forecast,
    ])

    # Convert predicted factors into futures prices
    return np.asarray(
        futures_log_price(
            predicted_state,
            target_tau,
            params,
        ),
        dtype=float,
    )
