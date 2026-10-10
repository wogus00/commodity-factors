import numpy as np
import pytest

from commodity_factors.schwartz_smith import process_covariance, pricing_adjustment, transition_matrix
from commodity_factors.simulation import simulate_dataset


@pytest.mark.parametrize("gap", [1 / 365.25, 3 / 365.25, 1.0])
def test_exact_covariance_and_correlated_shocks(model_params, gap):
    _, p = model_params
    Q = process_covariance(p["kappa"], p["sigma_chi"], p["sigma_xi"], p["rho"], gap)
    np.testing.assert_allclose(Q, Q.T)
    assert np.linalg.eigvalsh(Q).min() > 0
    assert Q[0, 1] != 0
    np.testing.assert_allclose(Q[1, 1], p["sigma_xi"] ** 2 * gap)
    np.testing.assert_allclose(transition_matrix(p["kappa"], gap),
                               np.diag([np.exp(-p["kappa"] * gap), 1]))


def test_pricing_contains_convexity_adjustment(model_params):
    _, p = model_params
    tau = np.array([0.0, 0.5, 1.0])
    Q = process_covariance(p["kappa"], p["sigma_chi"], p["sigma_xi"], p["rho"], 0.5)
    expected = ((p["mu_xi"] - p["lambda_xi"]) * 0.5
                - p["lambda_chi"] / p["kappa"] * (1 - np.exp(-p["kappa"] * 0.5))
                + 0.5 * np.ones(2) @ Q @ np.ones(2))
    adjustment = pricing_adjustment(tau, p)
    assert adjustment[0] == 0
    np.testing.assert_allclose(adjustment[1], expected)


def test_seeded_simulation_is_reproducible(model_params):
    _, params = model_params
    a = simulate_dataset(50, np.array([0.1, 0.5]), params, seed=12)
    b = simulate_dataset(50, np.array([0.1, 0.5]), params, seed=12)
    for actual, expected in zip(a, b):
        np.testing.assert_array_equal(actual, expected)
