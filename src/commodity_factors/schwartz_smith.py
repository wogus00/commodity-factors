import numpy as np

def transition_matrix(kappa, dt):
    """Return the exact state transition matrix.
    F(delta T) = exp(-kappa * delta T) for the Schwartz-Smith model."""
    mat = np.array([
        [np.exp(-kappa * dt), 0],
        [0, 1]
    ])
    return mat



def transition_mean(mu_xi, dt):
    """Return the state transition intercept.
    c(delta T) = mu_xi * dt
    """
    mat = np.array([
        0,
        mu_xi * dt
    ])
    return mat


def process_covariance(kappa, sigma_chi, sigma_xi, rho, dt):
    """Return the exact process noise covariance."""

    q11 = (sigma_chi**2 / (2*kappa)) * (1-np.exp(-2*kappa * dt))
    q12 = (rho*sigma_chi*sigma_xi / kappa) * (1-np.exp(-kappa * dt))
    q21 = (rho*sigma_chi*sigma_xi / kappa) * (1-np.exp(-kappa * dt))
    q22 = sigma_xi**2 *dt
    mat = np.array([
        [q11, q12],
        [q21, q22]
    ])
    return mat



def observation_matrix(kappa, tau):
    """Return the maturity-dependent observation matrix."""

    tau = np.asarray(tau)
    mat = np.column_stack([
        np.exp(-kappa * tau), np.ones(len(tau))
    ])
    return mat



def pricing_adjustment(tau, params):
    """Calculate the risk-neutral futures pricing adjustment A(tau)."""
    tau = np.asarray(tau, dtype=float)

    kappa = params["kappa"]
    sigma_chi = params["sigma_chi"]
    sigma_xi = params["sigma_xi"]
    rho = params["rho"]
    mu_xi = params["mu_xi"]
    lambda_chi = params["lambda_chi"]
    lambda_xi = params["lambda_xi"]

    decay = np.exp(-kappa * tau)

    mean_adjustment = (
        -lambda_chi / kappa * (1 - decay)
        + (mu_xi - lambda_xi) * tau
    )

    variance = (
        sigma_chi**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * tau))
        + sigma_xi**2 * tau
        + 2 * rho * sigma_chi * sigma_xi / kappa * (1 - decay)
    )

    return mean_adjustment + 0.5 * variance

def futures_log_price(states, tau, params):
    """Compute theoretical log futures prices."""
    H = observation_matrix(params['kappa'], tau)
    A = pricing_adjustment(tau, params)

    return H @ states + A
