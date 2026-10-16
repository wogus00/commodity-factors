import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from commodity_factors.schwartz_smith import (transition_matrix,
                                              transition_mean,
                                              process_covariance,
                                              validate_parameters,
                                              pricing_adjustment)


# kalman.py

def kalman_predict(x, P, F, c, Q):
    x_pred = F @ x + c
    P_pred = F @ P @ F.T + Q
    return x_pred, (P_pred + P_pred.T) / 2

def kalman_update(x_pred, P_pred, y, H, a, R):
    # Prediction error
    innovation = y - (H @ x_pred + a)

    # Innovation covariance
    S = H @ P_pred @ H.T + R

    # Kalman gain
    K = cho_solve(cho_factor(S, lower=True), H @ P_pred).T

    # Updated factors
    x = x_pred + K @ innovation

    # Updated uncertainty (Joseph form)
    I = np.eye(len(x))
    M = I - K @ H
    P = M @ P_pred @ M.T + K @ R @ K.T
    P = (P + P.T) / 2

    return x, P, innovation, S


def kalman_log_likelihood(innovation, S):
    m = len(innovation)

    factor = cho_factor(S, lower=True)
    logdet = 2 * np.log(np.diag(factor[0])).sum()
    ell = -0.5 * (m * np.log(2 * np.pi) + logdet
                  + innovation @ cho_solve(factor, innovation))

    
    return ell



def kalman_filter(
    log_prices,
    tau,
    params,
    dt=1 / 252,
    sigma_epsilon=0.01,
    x0=None,
    P0=None,
    diagnostics=True,
):
    """
    Estimate Schwartz-Smith hidden factors using the Kalman filter.

    Parameters
    ----------
    log_prices : array (n_days, n_contracts)
        Observed log futures prices; NaN entries are skipped.
    tau : array (n_contracts,) or (n_days, n_contracts)
        Time to maturity in years.
    params : dict
        Schwartz-Smith model parameters.
    dt : float
        Time step in years.
    sigma_epsilon : float
        Standard deviation of measurement noise.
    x0 : array (2,), optional
        Prior mean before the first observation (default [0, 0]).
    P0 : array (2, 2), optional
        Prior covariance; default stationary chi variance and broad xi variance
        1e4. This is a proper finite prior, not an exact diffuse likelihood.
    diagnostics : bool
        Return marginal and Cholesky-whitened innovations, NIS and conditioning.
    """

    log_prices = np.asarray(log_prices, dtype=float)
    tau = np.asarray(tau, dtype=float)

    validate_parameters(params)
    if log_prices.ndim != 2 or 0 in log_prices.shape or np.isinf(log_prices).any():
        raise ValueError("log_prices must be a nonempty matrix with finite values or NaN")
    n_days, n_contracts = log_prices.shape
    observed = np.isfinite(log_prices)
    if not np.isfinite(sigma_epsilon) or sigma_epsilon <= 0:
        raise ValueError("sigma_epsilon must be positive")





    dt = np.asarray(dt, dtype=float)

    # Convert scalar dt to an array
    if dt.ndim == 0 and (not np.isfinite(dt) or dt <= 0):
        raise ValueError("All time intervals must be positive")
    if dt.ndim == 0:
        dt = np.full(n_days - 1, dt)

    if dt.shape != (n_days - 1,):
        raise ValueError("dt must be a scalar or have length n_days - 1")

    if not np.isfinite(dt).all() or np.any(dt <= 0):
        raise ValueError("All time intervals must be positive")




    # Allow fixed maturities or time-varying maturities
    if tau.shape == (n_contracts,):
        tau = np.broadcast_to(tau, (n_days, n_contracts))
    if (tau.shape != log_prices.shape or np.isinf(tau).any()
            or np.any(np.isfinite(tau) & (tau < 0))
            or np.any(observed & ~np.isfinite(tau))):
        raise ValueError("tau must contain nonnegative finite maturities at observed prices")
    known_tau = np.isfinite(tau)
    tau = np.where(known_tau, tau, 0.0)

    R = sigma_epsilon**2 * np.eye(n_contracts)

    # Initial factor estimates
    if x0 is None:
        x = np.zeros(2)
    else:
        x = np.asarray(x0, dtype=float).copy()

    # Initial estimation uncertainty
    if P0 is None:
        P = np.diag([params["sigma_chi"]**2 / (2 * params["kappa"]), 1e4])
    else:
        P = np.asarray(P0, dtype=float).copy()

    if x.shape != (2,) or not np.isfinite(x).all():
        raise ValueError("x0 must be a finite vector of length 2")
    if (P.shape != (2, 2) or not np.isfinite(P).all()
            or not np.allclose(P, P.T, rtol=0, atol=1e-12)
            or np.linalg.eigvalsh(P).min() < 0):
        raise ValueError("P0 must be finite, symmetric and positive semidefinite")

    # Store results
    predicted_states = np.zeros((n_days, 2))
    filtered_states = np.zeros((n_days, 2))
    innovations = np.full((n_days, n_contracts), np.nan)
    if diagnostics:
        standardized = np.full_like(innovations, np.nan)
        whitened = np.full_like(innovations, np.nan)
        innovation_variances = np.full_like(innovations, np.nan)
        nis = np.full(n_days, np.nan)
        condition = np.full(n_days, np.nan)
    predicted_prices = np.zeros_like(log_prices)
    filtered_covariances = np.zeros((n_days, 2, 2))
    log_likelihoods = np.zeros(n_days)

    # Price adjustments are independent of latent states; compute once per fit.
    adjustments = pricing_adjustment(tau, params)
    loadings = np.exp(-params["kappa"] * tau)
    transitions = {}
    for delta in np.unique(dt):
        transitions[float(delta)] = (
            transition_matrix(params["kappa"], delta),
            transition_mean(params["mu_xi"], delta),
            process_covariance(params["kappa"], params["sigma_chi"],
                               params["sigma_xi"], params["rho"], delta),
        )

    log_likelihood = 0.0

    for t in range(n_days):

        # 2. Predict today's factors
        # At t=0, use the initial state distribution directly
        if t == 0:
            x_pred = x.copy()
            P_pred = P.copy()

        else:
            # Time elapsed since the previous trading day
            delta_t = dt[t - 1]

            F, c, Q = transitions[float(delta_t)]

            x_pred, P_pred = kalman_predict(
                x, P, F, c, Q
            )
        # 3. Calculate H and pricing adjustment
        H = np.column_stack([loadings[t], np.ones(n_contracts)])
        a = adjustments[t]
        predicted_prices[t] = H @ x_pred + a

        predicted_prices[t, ~known_tau[t]] = np.nan
        mask = observed[t]
        if mask.any():
            try:
                x, P, innovation, S = kalman_update(
                    x_pred, P_pred, log_prices[t, mask], H[mask], a[mask],
                    R[np.ix_(mask, mask)],
                )
                log_likelihoods[t] = kalman_log_likelihood(innovation, S)
            except np.linalg.LinAlgError as exc:
                raise np.linalg.LinAlgError(f"Innovation covariance failed at row {t}") from exc
            innovations[t, mask] = innovation
            if diagnostics:
                z = solve_triangular(np.linalg.cholesky(S), innovation, lower=True)
                innovation_variances[t, mask] = np.diag(S)
                standardized[t, mask] = innovation / np.sqrt(np.diag(S))
                whitened[t, mask] = z
                nis[t] = z @ z
                condition[t] = np.linalg.cond(S)
        else:
            x, P = x_pred, P_pred
        if not np.isfinite(x).all() or not np.isfinite(P).all() or not np.isfinite(log_likelihoods[t]):
            raise FloatingPointError(f"Nonfinite Kalman result at row {t}")
        log_likelihood += log_likelihoods[t]

        # 6. Store results
        predicted_states[t] = x_pred
        filtered_states[t] = x
        filtered_covariances[t] = P

    result = {
        "predicted_states": predicted_states,
        "filtered_states": filtered_states,
        "innovations": innovations,
        "predicted_prices": predicted_prices,
        "filtered_covariances": filtered_covariances,
        "log_likelihoods": log_likelihoods,
        "log_likelihood": log_likelihood,
    }

    if diagnostics:
        result.update(standardized_innovations=standardized, whitened_innovations=whitened,
                      innovation_variances=innovation_variances, normalized_innovation_squared=nis,
                      innovation_condition_numbers=condition, observation_counts=observed.sum(axis=1))
    return result
