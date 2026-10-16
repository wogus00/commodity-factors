"""Schwartz–Smith: physical dchi=-kappa*chi*dt+sigma_chi*dW,
dxi=mu_xi*dt+sigma_xi*dB, corr(dW,dB)=rho.
Risk-neutral drifts subtract lambda_chi and lambda_xi (drift units).
"""
import numpy as np


def validate_parameters(params):
    names = ("kappa", "sigma_chi", "sigma_xi", "rho", "mu_xi", "lambda_chi", "lambda_xi")
    if not all(np.isfinite(params[name]) for name in names):
        raise ValueError("Model parameters must be finite")
    if params["kappa"] <= 0 or min(params["sigma_chi"], params["sigma_xi"]) < 0:
        raise ValueError("kappa must be positive and volatilities nonnegative")
    if abs(params["rho"]) > 1:
        raise ValueError("rho must be between -1 and 1")


def _decay_integral(kappa, time):
    """Integral of exp(-kappa*s), stable near zero, including its limit."""
    time = np.asarray(time, dtype=float)
    if not np.isfinite(kappa) or kappa < 0 or not np.isfinite(time).all() or np.any(time < 0):
        raise ValueError("kappa and time must be finite and nonnegative")
    return time if kappa == 0 else -np.expm1(-kappa * time) / kappa


def transition_matrix(kappa, dt):
    """Exact physical transition for state order [chi, xi]."""
    return np.diag([np.exp(-kappa * dt), 1.0])


def transition_mean(mu_xi, dt):
    return np.array([0.0, mu_xi * dt])


def process_covariance(kappa, sigma_chi, sigma_xi, rho, dt):
    """Exact integrated covariance, including the Brownian kappa=0 limit."""
    if not np.isfinite([sigma_chi, sigma_xi, rho]).all() or min(sigma_chi, sigma_xi) < 0 or abs(rho) > 1:
        raise ValueError("Invalid diffusion parameters")
    q11 = sigma_chi**2 * _decay_integral(2 * kappa, dt)
    q12 = rho * sigma_chi * sigma_xi * _decay_integral(kappa, dt)
    return np.array([[q11, q12], [q12, sigma_xi**2 * dt]])


def observation_matrix(kappa, tau):
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    return np.column_stack([np.exp(-kappa * tau), np.ones(len(tau))])


def pricing_adjustment(tau, params):
    """A(tau) in log F = exp(-kappa*tau)*chi + xi + A(tau).

    mu_xi is physical; mu_xi-lambda_xi is the risk-neutral long-run drift.
    The half-variance term prices the expected spot level, not expected log spot.
    """
    tau = np.asarray(tau, dtype=float)
    kappa = params["kappa"]
    b = _decay_integral(kappa, tau)
    variance = (params["sigma_chi"]**2 * _decay_integral(2 * kappa, tau)
                + params["sigma_xi"]**2 * tau
                + 2 * params["rho"] * params["sigma_chi"] * params["sigma_xi"] * b)
    return (-params["lambda_chi"] * b
            + (params["mu_xi"] - params["lambda_xi"]) * tau + 0.5 * variance)


def futures_log_price(states, tau, params):
    return observation_matrix(params['kappa'], tau) @ states + pricing_adjustment(tau, params)
