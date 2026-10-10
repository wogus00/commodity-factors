import numpy as np
from scipy.linalg import cho_factor, cho_solve

from commodity_factors.schwartz_smith import (transition_matrix,
                                              transition_mean,
                                              process_covariance,
                                              observation_matrix,
                                              pricing_adjustment)


# kalman.py

def kalman_predict(x, P, F, c, Q):
    x_pred = F @ x + c
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred

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
):
    """
    Estimate Schwartz-Smith hidden factors using the Kalman filter.

    Parameters
    ----------
    log_prices : array (n_days, n_contracts)
        Observed log futures prices.
    tau : array (n_contracts,) or (n_days, n_contracts)
        Time to maturity in years.
    params : dict
        Schwartz-Smith model parameters.
    dt : float
        Time step in years.
    sigma_epsilon : float
        Standard deviation of measurement noise.
    x0 : array (2,), optional
        Initial state estimate.
    P0 : array (2, 2), optional
        Initial state covariance.
    """

    log_prices = np.asarray(log_prices, dtype=float)
    tau = np.asarray(tau, dtype=float)

    n_days, n_contracts = log_prices.shape
    if n_days == 0 or n_contracts == 0 or not np.isfinite(log_prices).all():
        raise ValueError("log_prices must be a nonempty finite matrix")
    if not np.isfinite(sigma_epsilon) or sigma_epsilon <= 0:
        raise ValueError("sigma_epsilon must be positive")





    dt = np.asarray(dt, dtype=float)

    # Convert scalar dt to an array
    if dt.ndim == 0:
        dt = np.full(n_days - 1, dt)

    if dt.shape != (n_days - 1,):
        raise ValueError("dt must be a scalar or have length n_days - 1")

    if not np.isfinite(dt).all() or np.any(dt <= 0):
        raise ValueError("All time intervals must be positive")




    # Allow fixed maturities or time-varying maturities
    if tau.ndim == 1:
        tau = np.broadcast_to(tau, (n_days, n_contracts))
    if tau.shape != log_prices.shape or not np.isfinite(tau).all() or np.any(tau <= 0):
        raise ValueError("tau must contain positive finite maturities matching prices")

    R = sigma_epsilon**2 * np.eye(n_contracts)

    # Initial factor estimates
    if x0 is None:
        x = np.array([0.0, np.mean(log_prices[0])])
    else:
        x = np.asarray(x0, dtype=float).copy()

    # Initial estimation uncertainty
    if P0 is None:
        P = np.diag([0.1, 0.1])
    else:
        P = np.asarray(P0, dtype=float).copy()

    # Store results
    predicted_states = np.zeros((n_days, 2))
    filtered_states = np.zeros((n_days, 2))
    innovations = np.zeros((n_days, n_contracts))
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

        # 4. Update using today's observed prices
        x, P, innovation, S = kalman_update(
            x_pred,
            P_pred,
            log_prices[t],
            H,
            a,
            R,
        )

        # 5. Calculate log-likelihood
        log_likelihoods[t] = kalman_log_likelihood(
            innovation,
            S,
        )
        log_likelihood += log_likelihoods[t]

        # 6. Store results
        predicted_states[t] = x_pred
        filtered_states[t] = x
        innovations[t] = innovation
        filtered_covariances[t] = P

    return {
        "predicted_states": predicted_states,
        "filtered_states": filtered_states,
        "innovations": innovations,
        "predicted_prices": predicted_prices,
        "filtered_covariances": filtered_covariances,
        "log_likelihoods": log_likelihoods,
        "log_likelihood": log_likelihood,
    }
