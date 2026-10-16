
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
                            sigma_epsilon=0.01, x0=None, P0=None):
    """
    Calculate negative log-likelihood for candidate parameters.
    """

    if np.asarray(theta).shape != (4,):
        raise ValueError("theta must contain four parameters")
    if not np.isfinite(np.asarray(log_prices, dtype=float)).any():
        raise ValueError("MLE requires observed prices")
    params = fixed_params.copy()

    for name, value in zip(PARAMETER_NAMES, theta):
        params[name] = float(value)

    results = kalman_filter(
        log_prices=log_prices,
        tau=tau,
        params=params,
        dt=dt,
        sigma_epsilon=sigma_epsilon, x0=x0, P0=P0, diagnostics=False,
    )

    return -results["log_likelihood"]


def estimate_parameters(
    log_prices,
    tau,
    fixed_params,
    initial_guess=None,
    dt=1 / 252,
    sigma_epsilon=0.01,
    *, x0=None, P0=None, maxiter=300,
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

    _validate_start(initial_guess, bounds)
    result = minimize(
        fun=negative_log_likelihood,
        x0=initial_guess,
        args=(
            log_prices,
            tau,
            fixed_params,
            dt,
            sigma_epsilon, x0, P0,
        ),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter},
    )

    estimated_params = fixed_params.copy()

    for name, value in zip(PARAMETER_NAMES, result.x):
        estimated_params[name] = float(value)

    return {
        "params": estimated_params,
        "log_likelihood": -result.fun,
        "success": bool(result.success and np.isfinite(result.fun)),
        "diagnostics": _optimizer_diagnostics(result, bounds, PARAMETER_NAMES),
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


def negative_log_likelihood_real(theta, Y, tau, dt, x0=None, P0=None):
    """
    Negative Kalman log-likelihood for real WTI data.
    """
    if not np.isfinite(np.asarray(Y, dtype=float)).any():
        raise ValueError("MLE requires observed prices")
    params, sigma_epsilon = unpack_real_parameters(theta)

    results = kalman_filter(
        log_prices=Y,
        tau=tau,
        params=params,
        dt=dt,
        sigma_epsilon=sigma_epsilon, x0=x0, P0=P0, diagnostics=False,
    )

    return -results["log_likelihood"]


def estimate_real_parameters(Y, tau, dt, initial_guess=None, *,
                             fixed_lambda_chi=None, maxiter=300, x0=None, P0=None):
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
    if np.asarray(initial_guess).shape != (len(bounds),):
        raise ValueError("initial_guess must contain eight parameters")
    if fixed_lambda_chi is not None:
        initial_guess = list(initial_guess)
        initial_guess[5] = float(fixed_lambda_chi)
        bounds[5] = (float(fixed_lambda_chi), float(fixed_lambda_chi))

    _validate_start(initial_guess, bounds)
    result = minimize(
        fun=negative_log_likelihood_real,
        x0=initial_guess,
        args=(Y, tau, dt, x0, P0),
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
        "success": bool(result.success and np.isfinite(result.fun)),
        "diagnostics": _optimizer_diagnostics(result, bounds, REAL_PARAMETER_NAMES),
        "message": result.message,
        "optimizer_result": result,
    }


def _validate_start(initial_guess, bounds):
    guess = np.asarray(initial_guess, dtype=float)
    limits = np.asarray(bounds)
    if (guess.shape != (len(bounds),) or not np.isfinite(guess).all()
            or not np.isfinite(limits).all()
            or np.any(guess < limits[:, 0]) or np.any(guess > limits[:, 1])):
        raise ValueError("initial_guess must be finite, match the parameters and lie within bounds")


def _optimizer_diagnostics(result, bounds, names):
    """Report constrained stationarity and active bounds, not just solver success."""
    projected = np.asarray(result.jac, dtype=float).copy()
    active, fixed = [], []
    for i, (name, (lower, upper)) in enumerate(zip(names, bounds)):
        if lower == upper:
            fixed.append(name)
            projected[i] = 0.0
            continue
        at_lower = np.isclose(result.x[i], lower, rtol=0, atol=1e-6)
        at_upper = np.isclose(result.x[i], upper, rtol=0, atol=1e-6)
        if at_lower or at_upper:
            active.append(name)
        if (at_lower and projected[i] > 0) or (at_upper and projected[i] < 0):
            projected[i] = 0.0
    return {"iterations": int(result.nit), "function_evaluations": int(result.nfev),
            "projected_gradient_inf_norm": float(np.max(np.abs(projected))),
            "active_bounds": active, "fixed_parameters": fixed,
            "finite_objective": bool(np.isfinite(result.fun))}
