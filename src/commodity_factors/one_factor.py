
import numpy as np
from scipy.optimize import minimize


def one_factor_kalman_filter(
    Y,
    tau,
    dt,
    params,
):
    """
    One-factor Gaussian state-space model.

    State:
        x[t] = x[t-1] + mu*dt + process_noise

    Observation:
        Y[t,j] = x[t] + (delta + 0.5*sigma**2)*tau[t,j]
                 + measurement_noise
    """

    Y = np.asarray(Y, dtype=float)
    tau = np.asarray(tau, dtype=float)
    dt = np.asarray(dt, dtype=float)

    n_days, n_contracts = Y.shape

    mu = params["mu"]
    sigma = params["sigma"]
    delta = params["delta"]
    sigma_epsilon = params["sigma_epsilon"]
    if n_days == 0 or not np.isfinite(Y).all():
        raise ValueError("Y must be nonempty and finite")
    if tau.shape != Y.shape or not np.isfinite(tau).all() or np.any(tau <= 0):
        raise ValueError("tau must contain positive finite maturities matching Y")
    if dt.shape != (n_days - 1,) or not np.isfinite(dt).all() or np.any(dt <= 0):
        raise ValueError("dt must contain positive gaps between consecutive dates")
    if sigma <= 0 or sigma_epsilon <= 0:
        raise ValueError("Noise standard deviations must be positive")

    R = sigma_epsilon**2 * np.eye(n_contracts)
    H = np.ones(n_contracts)

    filtered_states = np.zeros(n_days)
    innovations = np.zeros_like(Y)
    predicted_prices = np.zeros_like(Y)
    log_likelihoods = np.zeros(n_days)

    # Initial state and uncertainty
    a0 = (delta + 0.5 * sigma**2) * tau[0]
    x = np.mean(Y[0] - a0)
    P = 0.1

    log_likelihood = 0.0

    for t in range(n_days):

        # Prediction step
        if t > 0:
            x_pred = x + mu * dt[t - 1]
            P_pred = P + sigma**2 * dt[t - 1]
        else:
            x_pred = x
            P_pred = P

        # Observation equation
        a = (delta + 0.5 * sigma**2) * tau[t]

        y_pred = H * x_pred + a
        innovation = Y[t] - y_pred

        S = P_pred * np.outer(H, H) + R

        # Kalman gain
        K = np.linalg.solve(
            S, P_pred * H
        )

        # Update state
        x = x_pred + K @ innovation

        # Joseph-form covariance update
        M = 1.0 - K @ H
        P = M**2 * P_pred + K @ R @ K
        P = max(float(P), 0.0)

        # Log-likelihood
        sign, logdet = np.linalg.slogdet(S)

        if sign <= 0:
            raise ValueError("Invalid innovation covariance")

        log_likelihoods[t] = -0.5 * (
            n_contracts * np.log(2 * np.pi)
            + logdet
            + innovation @ np.linalg.solve(S, innovation)
        )
        log_likelihood += log_likelihoods[t]

        filtered_states[t] = x
        innovations[t] = innovation
        predicted_prices[t] = y_pred

    return {
        "filtered_states": filtered_states,
        "innovations": innovations,
        "predicted_prices": predicted_prices,
        "log_likelihood": log_likelihood,
        "log_likelihoods": log_likelihoods,
    }


def estimate_one_factor(Y, tau, dt, maxiter=300):
    """
    Estimate one-factor model parameters by MLE.
    """

    def objective(theta):
        mu, sigma, delta, sigma_epsilon = theta

        params = {
            "mu": mu,
            "sigma": sigma,
            "delta": delta,
            "sigma_epsilon": sigma_epsilon,
        }

        result = one_factor_kalman_filter(
            Y, tau, dt, params
        )

        return -result["log_likelihood"]

    initial_guess = [
        0.02,   # mu
        0.30,   # sigma
        -0.05,  # delta
        0.01,   # sigma_epsilon
    ]

    bounds = [
        (-1.0, 1.0),
        (0.01, 2.0),
        (-2.0, 2.0),
        (0.001, 0.1),
    ]

    result = minimize(
        objective,
        x0=initial_guess,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter},
    )

    names = [
        "mu",
        "sigma",
        "delta",
        "sigma_epsilon",
    ]

    params = dict(zip(names, result.x))

    return {
        "params": params,
        "log_likelihood": -result.fun,
        "success": result.success,
        "message": result.message,
        "optimizer_result": result,
    }
