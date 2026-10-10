
import numpy as np
from scipy.optimize import minimize

from commodity_factors.kalman import kalman_filter


PARAMETER_NAMES = ["kappa", 
                   "sigma_chi", 
                   "sigma_xi", 
                   "rho"]


def negative_log_likelihood(theta, 
                            log_prices,
                            tau,
                            fixed_params,
                            dt=1 / 252,
                            sigma_epsilon=0.01):
    """
    Calculate negative log-likelihood for candidate parameters.
    """

    params = fixed_params.copy()

    for name, value in zip(PARAMETER_NAMES, theta):
        params[name] = float(value)

    results = kalman_filter(
        log_prices=log_prices,
        tau=tau,
        params=params,
        dt=dt,
        sigma_epsilon=sigma_epsilon,
    )

    return -results["log_likelihood"]


def estimate_parameters(
    log_prices,
    tau,
    fixed_params,
    initial_guess=None,
    dt=1 / 252,
    sigma_epsilon=0.01,
):
    """
    Estimate four Schwartz-Smith parameters by MLE.
    """

    if initial_guess is None:
        initial_guess = [0.8, 0.2, 0.15, 0.0]

    # Constraints on parameter values
    bounds = [
        (0.01, 10.0),   # kappa > 0
        (0.001, 2.0),   # sigma_chi > 0
        (0.001, 2.0),   # sigma_xi > 0
        (-0.99, 0.99),  # -1 < rho < 1
    ]

    result = minimize(
        fun=negative_log_likelihood,
        x0=initial_guess,
        args=(
            log_prices,
            tau,
            fixed_params,
            dt,
            sigma_epsilon,
        ),
        method="L-BFGS-B",
        bounds=bounds,
    )

    estimated_params = fixed_params.copy()

    for name, value in zip(PARAMETER_NAMES, result.x):
        estimated_params[name] = float(value)

    return {
        "params": estimated_params,
        "log_likelihood": -result.fun,
        "success": result.success,
        "message": result.message,
        "optimizer_result": result,
    }




# Real-data parameter estimation

REAL_PARAMETER_NAMES = [
    "kappa",
    "sigma_chi",
    "sigma_xi",
    "rho",
    "mu_xi",
    "lambda_chi",
    "delta",
    "sigma_epsilon",
]


def unpack_real_parameters(theta):
    """
    Convert optimizer parameters into Schwartz-Smith parameters.
    """
    (
        kappa,
        sigma_chi,
        sigma_xi,
        rho,
        mu_xi,
        lambda_chi,
        delta,
        sigma_epsilon,
    ) = theta

    params = {
        "kappa": float(kappa),
        "sigma_chi": float(sigma_chi),
        "sigma_xi": float(sigma_xi),
        "rho": float(rho),
        "mu_xi": float(mu_xi),
        "lambda_chi": float(lambda_chi),
        "lambda_xi": float(mu_xi - delta),
    }

    return params, float(sigma_epsilon)


def negative_log_likelihood_real(theta, Y, tau, dt):
    """
    Negative Kalman log-likelihood for real WTI data.
    """
    params, sigma_epsilon = unpack_real_parameters(theta)

    results = kalman_filter(
        log_prices=Y,
        tau=tau,
        params=params,
        dt=dt,
        sigma_epsilon=sigma_epsilon,
    )

    return -results["log_likelihood"]


def estimate_real_parameters(Y, tau, dt, initial_guess=None, *,
                             fixed_lambda_chi=None, maxiter=300):
    """
    Estimate Schwartz-Smith parameters using real futures prices.

    Set ``fixed_lambda_chi=0`` for the restricted dashboard calibration.
    Leaving it as None preserves the existing notebook parameterization.
    """

    if initial_guess is None:
        initial_guess = [
            1.50,  # kappa
            0.30,  # sigma_chi
            0.20,  # sigma_xi
            0.30,  # rho
            0.02,  # mu_xi
            0.01,  # lambda_chi
            0.01,  # delta
            0.01,  # sigma_epsilon
        ]

    bounds = [
        (0.05, 10.0),   # kappa
        (0.01, 2.0),    # sigma_chi
        (0.01, 2.0),    # sigma_xi
        (-0.95, 0.95),  # rho
        (-1.0, 1.0),    # mu_xi
        (-2.0, 2.0),    # lambda_chi
        (-2.0, 2.0),    # delta
        (0.001, 0.1),   # sigma_epsilon
    ]
    if fixed_lambda_chi is not None:
        initial_guess = list(initial_guess)
        initial_guess[5] = float(fixed_lambda_chi)
        bounds[5] = (float(fixed_lambda_chi), float(fixed_lambda_chi))

    result = minimize(
        fun=negative_log_likelihood_real,
        x0=initial_guess,
        args=(Y, tau, dt),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter},
    )

    params, sigma_epsilon = unpack_real_parameters(result.x)

    return {
        "params": params,
        "sigma_epsilon": sigma_epsilon,
        "delta": params["mu_xi"] - params["lambda_xi"],
        "log_likelihood": -result.fun,
        "success": result.success,
        "message": result.message,
        "optimizer_result": result,
    }
