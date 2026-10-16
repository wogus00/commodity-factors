"""Independent mathematical checks and seeded calibration regression tests."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import multivariate_normal

from commodity_factors.estimation import estimate_parameters, estimate_real_parameters, unpack_real_parameters
from commodity_factors.kalman import kalman_filter
from commodity_factors.schwartz_smith import (
    process_covariance, transition_matrix, pricing_adjustment, futures_log_price,
)
from commodity_factors.simulation import simulate_dataset


@pytest.mark.parametrize("kappa", [0.0, 1e-16, 0.8, 10.0])
def test_covariance_matches_integrated_diffusion(kappa):
    dt, sc, sx, rho = 0.37, 0.3, 0.2, -0.7
    expected = [[sc**2 * quad(lambda s: np.exp(-2*kappa*s), 0, dt)[0],
                 rho*sc*sx * quad(lambda s: np.exp(-kappa*s), 0, dt)[0]],
                [rho*sc*sx * quad(lambda s: np.exp(-kappa*s), 0, dt)[0], sx**2*dt]]
    np.testing.assert_allclose(process_covariance(kappa, sc, sx, rho, dt), expected, rtol=1e-14)


def test_exact_transition_semigroup(model_params):
    p = model_params[1]
    q = lambda t: process_covariance(p['kappa'], p['sigma_chi'], p['sigma_xi'], p['rho'], t)
    F = transition_matrix(p['kappa'], 0.7)
    np.testing.assert_allclose(q(0.9), F @ q(0.2) @ F.T + q(0.7))


def test_risk_neutral_pricing_and_physical_drift(model_params):
    p = model_params[1]
    tau = np.array([0, 0.2, 2.0])
    x = np.array([0.1, 4.0])
    changed = dict(p, mu_xi=p['mu_xi']+0.2, lambda_xi=p['lambda_xi']+0.2)
    np.testing.assert_allclose(futures_log_price(x, tau, p), futures_log_price(x, tau, changed))
    assert futures_log_price(x, tau, p)[0] == x.sum()
    args = dict(log_prices=np.full((2, 3), np.nan), tau=tau, dt=0.3, x0=x)
    a = kalman_filter(params=p, **args)
    b = kalman_filter(params=changed, **args)
    np.testing.assert_allclose(b['predicted_states'][1] - a['predicted_states'][1], [0, 0.06])
    mapped, noise = unpack_real_parameters([1, .3, .2, .1, .08, .02, -.03, .01])
    assert mapped['mu_xi'] - mapped['lambda_xi'] == pytest.approx(-.03)
    assert noise == .01


def test_likelihood_matches_full_joint_gaussian_with_missing_and_rolls(model_params):
    p = model_params[1]
    tau = np.array([[.1, .5], [.08, .48], [.3, .7]])  # roll on last date
    gaps = [.02, .2]
    x0 = np.array([.1, 4.1])
    P0 = np.array([[.04, .01], [.01, .03]])
    means, covs = [x0], [[P0]]
    for t, dt in enumerate(gaps, 1):
        F = transition_matrix(p['kappa'], dt)
        means.append(F @ means[-1] + [0, p['mu_xi']*dt])
        cross = [F @ covs[t-1][s] for s in range(t)]
        for s in range(t):
            covs[s].append(cross[s].T)
        Q = process_covariance(p['kappa'], p['sigma_chi'], p['sigma_xi'], p['rho'], dt)
        covs.append(cross + [F @ covs[t-1][t-1] @ F.T + Q])
    H = [np.column_stack([np.exp(-p['kappa']*row), np.ones(2)]) for row in tau]
    mean = np.concatenate([H[t] @ means[t] + pricing_adjustment(tau[t], p) for t in range(3)])
    cov = np.block([[H[t] @ covs[t][s] @ H[s].T for s in range(3)] for t in range(3)])
    cov += .01**2 * np.eye(6)
    Y = (mean + [.01, -.02, .03, -.01, .02, .04]).reshape(3, 2)
    Y[1, 0] = np.nan
    observed = np.isfinite(Y.ravel())
    result = kalman_filter(Y, tau, p, dt=gaps, x0=x0, P0=P0)
    expected = multivariate_normal.logpdf(Y.ravel()[observed], mean=mean[observed], cov=cov[np.ix_(observed, observed)])
    assert result['log_likelihood'] == pytest.approx(expected, abs=1e-10)
    np.testing.assert_allclose(np.nansum(result['whitened_innovations']**2, axis=1), result['normalized_innovation_squared'])
    reordered = kalman_filter(Y[:, ::-1], tau[:, ::-1], p, dt=gaps, x0=x0, P0=P0)
    np.testing.assert_allclose(result['filtered_states'], reordered['filtered_states'])
    assert result['log_likelihood'] == pytest.approx(reordered['log_likelihood'])


def test_default_prior_does_not_use_first_observation(model_params):
    p = model_params[1]
    a = kalman_filter([[4., 4.1]], [.1, 1.], p)
    b = kalman_filter([[5., 5.1]], [.1, 1.], p)
    np.testing.assert_array_equal(a['predicted_states'], b['predicted_states'])
    np.testing.assert_array_equal(a['predicted_prices'], b['predicted_prices'])


def test_missing_rows_propagate_and_unknown_maturities_stay_missing(model_params):
    p = model_params[1]
    Y = np.array([[4., np.nan], [np.nan, np.nan], [4.1, 4.2]])
    tau = np.array([[.1, np.nan], [.1, .5], [.1, .5]])
    r = kalman_filter(Y, tau, p, dt=[.1, .2])
    assert r['log_likelihoods'][1] == 0
    assert np.isnan(r['innovations'][1]).all()
    assert np.isnan(r['predicted_prices'][0, 1])
    np.testing.assert_array_equal(r['filtered_states'][1], r['predicted_states'][1])
    np.testing.assert_allclose(r['predicted_states'][1], transition_matrix(p['kappa'], .1) @ r['filtered_states'][0] + [0, p['mu_xi']*.1])
    assert r['observation_counts'].tolist() == [1, 0, 2]


@pytest.mark.parametrize('kwargs', [dict(x0=[0]), dict(P0=[[1, 2], [2, 1]]),
                                  dict(P0=[[1, .1], [0, 1]]), dict(dt=0)])
def test_invalid_prior_or_time_rejected(model_params, kwargs):
    with pytest.raises(ValueError):
        kalman_filter([[4., 4.1]], [.1, 1.], model_params[1], **kwargs)


@pytest.mark.parametrize('change', [dict(kappa=0), dict(rho=1.1), dict(sigma_chi=-.1), dict(mu_xi=np.nan)])
def test_invalid_model_rejected(model_params, change):
    with pytest.raises(ValueError):
        kalman_filter([[4., 4.1]], [.1, 1.], dict(model_params[1], **change))


def test_synthetic_likelihood_diagnostics_and_parameter_recovery():
    p = dict(kappa=1.2, sigma_chi=.3, sigma_xi=.2, rho=-.3,
             mu_xi=.02, lambda_chi=.01, lambda_xi=.03)
    tau = np.array([.05, .2, .5, 1., 2., 4.])
    # Long time span and broad maturity coverage identify kappa and diffusion.
    states, Y = simulate_dataset(1000, tau, p, dt=.025, sigma_epsilon=.005, seed=812)
    prior = dict(x0=states[0], P0=np.zeros((2, 2)))
    truth = kalman_filter(Y, tau, p, dt=.025, sigma_epsilon=.005, **prior)
    wrong = kalman_filter(Y, tau, dict(p, sigma_chi=.6, sigma_xi=.4), dt=.025, sigma_epsilon=.005, **prior)
    assert truth['log_likelihood'] > wrong['log_likelihood']
    z = truth['whitened_innovations'][10:]
    assert abs(z.mean()) < .05
    assert abs(np.mean(z**2) - 1) < .07
    assert abs(np.corrcoef(z[:-1].ravel(), z[1:].ravel())[0, 1]) < .05
    assert np.linalg.eigvalsh(truth['filtered_covariances']).min() >= -1e-12
    fit = estimate_parameters(Y, tau, p, initial_guess=[.8, .25, .15, 0.], dt=.025,
                              sigma_epsilon=.005, **prior)
    assert fit['success'], fit['message']
    assert fit['log_likelihood'] >= truth['log_likelihood'] - 1e-6
    for name in ['kappa', 'sigma_chi', 'sigma_xi']:
        assert fit['params'][name] == pytest.approx(p[name], rel=.2)
    assert fit['params']['rho'] == pytest.approx(p['rho'], abs=.15)
    assert fit['diagnostics']['active_bounds'] == []


def test_optimizer_failure_and_initial_guess_validation(model_params):
    p = model_params[1]
    _, Y = simulate_dataset(20, np.array([.1, .5, 2.]), p)
    with pytest.raises(ValueError, match='initial_guess'):
        estimate_parameters(Y, [.1, .5, 2.], p, initial_guess=[-1, .2, .1, 0])
    fit = estimate_real_parameters(Y, [.1, .5, 2.], 1/252, fixed_lambda_chi=0, maxiter=0)
    assert not fit['success']
    assert fit['diagnostics']['fixed_parameters'] == ['lambda_chi']
    assert fit['diagnostics']['finite_objective']


def test_near_collinear_loadings_with_small_noise_remain_stable(model_params):
    p = dict(model_params[1], rho=.999999)
    tau = np.array([.5, .5 + 1e-10, .5 + 2e-10])
    _, Y = simulate_dataset(100, tau, p, sigma_epsilon=1e-5)
    r = kalman_filter(Y, tau, p, sigma_epsilon=1e-5, P0=np.diag([.1, 1.]))
    assert np.isfinite(r['log_likelihood'])
    assert np.isfinite(r['whitened_innovations']).all()
    assert np.linalg.eigvalsh(r['filtered_covariances']).min() >= -1e-12
    assert r['innovation_condition_numbers'].max() > 1e8


def test_calendar_maturities_and_rank_roll():
    import pandas as pd
    from commodity_factors.data import prepare_panel, prepare_kalman_data

    rows = []
    for date in ['2024-02-28', '2024-03-01']:
        for contract, expiry in [('A', '2024-02-29'), ('B', '2024-03-29'), ('C', '2024-04-30')]:
            rows.append(dict(date_=date, futcode=contract, lasttrddate=expiry, settlement=80.))
    panel = prepare_panel(pd.DataFrame(rows), n=2)
    _, _, tau, dt = prepare_kalman_data(panel)
    np.testing.assert_allclose(tau * 365.25, [[1, 30], [28, 60]])
    np.testing.assert_allclose(dt * 365.25, [2])
    assert panel.loc[panel.maturity_rank == 1, 'futcode'].tolist() == ['A', 'B']


def test_mle_rejects_no_observations(model_params):
    with pytest.raises(ValueError, match='observed prices'):
        estimate_parameters(np.full((2, 2), np.nan), [.1, .5], model_params[1])
    with pytest.raises(ValueError, match='observed prices'):
        estimate_real_parameters(np.full((2, 2), np.nan), [.1, .5], .01)
