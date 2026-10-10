import numpy as np
from commodity_factors.schwartz_smith import (
    transition_matrix,
    transition_mean,
    process_covariance,
    futures_log_price
)

def simulate_states(n_steps, params, dt=1/252, x0=None, seed=42):
    """
    Simulate the short-term and long-term latent factors.

    Returns
    -------
    states : ndarray, shape (n_steps, 2)
        Simulated [chi_t, xi_t] at each time step.
    """

    rng = np.random.default_rng(seed)

    if x0 is None:
        x0 = np.array([0.0, 4.3])

    states = np.zeros((n_steps, 2))
    states[0] = x0

    F = transition_matrix(params["kappa"], dt)
    c = transition_mean(params["mu_xi"], dt)
    Q = process_covariance(params['kappa'],
                           params['sigma_chi'],
                           params['sigma_xi'],
                           params['rho'], 
                           dt)
    for t in range(1, n_steps):
        noise = rng.multivariate_normal(mean=np.zeros(2), cov=Q)
        states[t] = F@states[t-1] + c + noise


    return states


def simulate_futures(states, tau, params, sigma_epsilon=0.01, seed=42):
    """
    Generate synthetic log futures prices from latent states.

    Parameters
    ----------
    states : ndarray, shape (n_steps, 2)
    tau : ndarray, shape (6,) or (n_steps, 6)
    params : dict
    sigma_epsilon : float

    Returns
    -------
    log_prices : ndarray, shape (n_steps, 6)
    """

    rng = np.random.default_rng(seed)

    n_steps = states.shape[0]
    tau = np.asarray(tau)

    if tau.ndim == 1:
        tau = np.broadcast_to(tau, (n_steps, len(tau)))

    prices = np.array([
        futures_log_price(states[t], tau[t], params)
        for t in range(n_steps)])

    # Add measurement noise
    noise = rng.normal(
        loc=0,
        scale=sigma_epsilon,
        size=prices.shape
    )

    log_prices = prices + noise

    return log_prices

def simulate_dataset(n_steps, 
                     tau, 
                     params,
                     dt=1/252,
                     sigma_epsilon=0.01,
                     seed=42,):
    """
    Simulate latent states and observed futures prices.

    Returns
    -------
    states : ndarray, shape (n_steps, 2)
    log_prices : ndarray, shape (n_steps, 6)
    """

    states = simulate_states(
        n_steps=n_steps,
        params=params,
        dt=dt,
        seed=seed)

    log_prices = simulate_futures(
        states=states,
        tau=tau,
        params=params,
        sigma_epsilon=sigma_epsilon,
        seed=seed + 1)

    return states, log_prices