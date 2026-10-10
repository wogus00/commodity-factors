"""Seeded synthetic curves with real calendar gaps and illustrative rolling contracts."""

import numpy as np
import pandas as pd

from commodity_factors.data import prepare_panel
from commodity_factors.schwartz_smith import process_covariance, transition_matrix, transition_mean
from commodity_factors.simulation import simulate_futures


DEMO_PARAMS = {"kappa": 1.5, "sigma_chi": 0.30, "sigma_xi": 0.18, "rho": 0.25,
               "mu_xi": 0.02, "lambda_chi": 0.0, "lambda_xi": 0.08}


def make_demo_panel(seed=42, n_ranks=6):
    """Generate 2023–2025 business-day curves, not historical WTI quotes.

    Cutoffs are illustrative monthly dates, not NYMEX expiry rules. State and
    measurement noise use independent random streams.
    """
    if n_ranks < 1:
        raise ValueError("n_ranks must be positive")
    dates = pd.bdate_range("2023-01-03", "2025-12-31")
    cutoffs = pd.date_range("2023-01-01", periods=36 + n_ranks + 1, freq="MS") + pd.Timedelta(days=19)
    ids = np.array([np.flatnonzero(cutoffs > date)[:n_ranks] for date in dates])
    selected_cutoffs = cutoffs.to_numpy()[ids]
    tau = (selected_cutoffs - dates.to_numpy()[:, None]) / np.timedelta64(1, "D") / 365.25
    dt = np.diff(dates.to_numpy()) / np.timedelta64(1, "D") / 365.25
    states = np.zeros((len(dates), 2))
    states[0] = [0.0, np.log(75.0)]
    rng = np.random.default_rng(seed)
    p = DEMO_PARAMS
    for t, gap in enumerate(dt, start=1):
        covariance = process_covariance(p["kappa"], p["sigma_chi"], p["sigma_xi"], p["rho"], gap)
        states[t] = (transition_matrix(p["kappa"], gap) @ states[t - 1]
                     + transition_mean(p["mu_xi"], gap)
                     + rng.multivariate_normal(np.zeros(2), covariance))
    Y = simulate_futures(states, tau, p, sigma_epsilon=0.004, seed=seed + 1)
    raw = pd.DataFrame({"date_": np.repeat(dates.to_numpy(), n_ranks), "futcode": ids.ravel(),
                        "lasttrddate": selected_cutoffs.ravel(), "settlement": np.exp(Y).ravel(),
                        "dsmnem": [f"DEMO-{pd.Timestamp(d):%Y-%m}" for d in selected_cutoffs.ravel()]})
    return prepare_panel(raw, n=n_ranks)
