import numpy as np
import pytest
from scipy.stats import multivariate_normal

from commodity_factors.data import prepare_kalman_data
from commodity_factors.kalman import kalman_filter, kalman_log_likelihood, kalman_update


def test_likelihood_matches_multivariate_normal():
    S = np.array([[2.0, 0.3], [0.3, 1.0]])
    innovation = np.array([0.2, -0.4])
    np.testing.assert_allclose(kalman_log_likelihood(innovation, S),
                               multivariate_normal.logpdf(innovation, cov=S))


def test_update_matches_scalar_gaussian_posterior():
    x, P, innovation, S = kalman_update(np.array([2.0]), np.array([[4.0]]),
                                       np.array([3.0]), np.array([[1.0]]), np.zeros(1), np.array([[1.0]]))
    np.testing.assert_allclose(x, [2.8])
    np.testing.assert_allclose(P, [[0.8]])
    np.testing.assert_allclose(innovation, [1.0])
    np.testing.assert_allclose(S, [[5.0]])


def test_irregular_time_filter_and_covariances(small_panel, model_params):
    _, Y, tau, dt = prepare_kalman_data(small_panel)
    result = kalman_filter(Y, tau, model_params[1], dt=dt)
    np.testing.assert_allclose(Y - result["predicted_prices"], result["innovations"])
    assert np.isfinite(result["log_likelihood"])
    assert np.linalg.eigvalsh(result["filtered_covariances"]).min() >= -1e-12
    with pytest.raises(ValueError, match="positive"):
        kalman_filter(Y, tau, model_params[1], dt=-1)


def test_non_positive_definite_innovation_covariance_fails():
    with pytest.raises(np.linalg.LinAlgError):
        kalman_log_likelihood(np.zeros(2), np.array([[1.0, 2.0], [2.0, 1.0]]))
