"""Rolling PCA alignment, leakage and common-support robustness checks."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.linalg import subspace_angles

from commodity_factors.pca_baseline import (
    CurvePCA, align_pca_loadings, constant_maturity_curves, rolling_pca_analysis,
    run_pca_robustness, save_pca_robustness, subperiod_pca_analysis,
)
from commodity_factors.plotting import plot_pca_robustness


@pytest.fixture
def curves():
    rng = np.random.default_rng(103)
    dates = pd.bdate_range('2021-12-01', periods=75)
    basis, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    changes = rng.normal(size=(len(dates), 5)) * [.03, .01, .004, .0003, .0001]
    return pd.DataFrame(4 + np.cumsum(changes @ basis.T, axis=0), index=dates,
                        columns=np.arange(30, 151, 30)/365.25)


@pytest.fixture
def panel(curves):
    return pd.DataFrame({'date_': np.repeat(curves.index, 5), 'futcode': np.tile(np.arange(5), len(curves)),
        'lasttrddate': np.repeat(curves.index, 5) + pd.to_timedelta(np.tile(np.arange(30, 151, 30), len(curves)), unit='D'),
        'settlement': np.exp(curves.to_numpy().ravel())})


def test_alignment_undoes_signs_and_component_swaps():
    basis, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(8, 3)))
    candidate = basis[:, [2, 0, 1]] * [-3., 2., -4.]
    aligned, order, signs = align_pca_loadings(basis, candidate)
    np.testing.assert_allclose(aligned, basis, atol=1e-14)
    assert order.tolist() == [1, 2, 0]
    assert signs.tolist() == [1, -1, -1]


def test_near_degenerate_rotation_preserves_subspace_not_directions():
    basis = np.eye(5)[:, :3]
    angle = np.pi/4
    rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    aligned, _, _ = align_pca_loadings(basis, basis @ rotation)
    assert abs(aligned[:, 1] @ basis[:, 1]) == pytest.approx(np.sqrt(.5))
    assert np.degrees(subspace_angles(basis, aligned)).max() < 1e-10


def test_rolling_prior_only_and_manual_heldout_reconstruction(curves):
    result = rolling_pca_analysis(curves, windows=(20,), step=1)
    assert (result['fits'].train_end < result['fits'].date_).all()
    date = curves.index[20]
    train = curves.diff().iloc[:20].dropna()
    for standardize, method in [(False, 'covariance'), (True, 'correlation')]:
        model = CurvePCA(4, standardize).fit(train)
        score = model.transform(curves.diff().iloc[[20]])
        for k in (1, 2, 3):
            truncated = score.copy()
            truncated[:, k:] = 0
            expected = curves.iloc[19].to_numpy() + model.inverse_transform(truncated)[0]
            actual = result['errors'].query('date_ == @date and method == @method and k == @k').sort_values('tau')
            np.testing.assert_allclose(actual.fitted_log, expected)
    for _, group in result['errors'].query('method == "covariance"').groupby('date_'):
        squared = group.groupby('k').error_log.apply(lambda x: np.sum(x*x)).to_numpy()
        assert np.all(np.diff(squared) <= 1e-15)


def test_future_and_current_observations_cannot_change_prior_fit(curves):
    baseline = rolling_pca_analysis(curves, windows=(20,), step=2)
    changed = curves.copy()
    cutoff = curves.index[40]
    changed.loc[cutoff:] += np.arange(5) * .2
    altered = rolling_pca_analysis(changed, windows=(20,), step=2)
    for table in ('spectrum', 'loadings', 'fits'):
        pd.testing.assert_frame_equal(baseline[table].query('date_ <= @cutoff').reset_index(drop=True),
                                      altered[table].query('date_ <= @cutoff').reset_index(drop=True))
    earlier = baseline['errors'].query('date_ < @cutoff').reset_index(drop=True)
    pd.testing.assert_frame_equal(earlier, altered['errors'].query('date_ < @cutoff').reset_index(drop=True))
    assert not np.allclose(baseline['errors'].query('date_ == @cutoff').fitted_log,
                           altered['errors'].query('date_ == @cutoff').fitted_log)


def test_window_counts_supplied_dates_not_only_complete_dates(curves):
    changed = curves.copy()
    changed.iloc[5, 0] = np.nan
    result = rolling_pca_analysis(changed, windows=(20,), step=20, min_coverage=.8)
    first = result['fits'].iloc[0]
    assert first.n_train == 17  # first difference, missing date and next difference
    assert first.window_start == curves.index[0]
    assert first.train_end == curves.index[19]
    strict = rolling_pca_analysis(changed, windows=(20,), step=20, min_coverage=1.)
    assert strict['fits'].iloc[0].status == 'insufficient_coverage'


def test_loading_instability_detects_changed_subspace():
    rng = np.random.default_rng(4)
    scores = rng.normal(size=(120, 3)) * [.03, .01, .004]
    directions = np.eye(6)[:, :3]
    shifted = np.eye(6)[:, [0, 1, 3]]
    values = np.concatenate([scores[:60] @ directions.T, scores[60:] @ shifted.T])
    curves = pd.DataFrame(values, index=pd.bdate_range('2021-01-01', periods=120), columns=np.arange(1, 7)/12)
    result = rolling_pca_analysis(curves, windows=(30,), step=10, transformation='log_prices')
    fits = result['fits'].query('method == "covariance"')
    assert fits.iloc[0].anchor_subspace_angle_deg < 1e-8
    assert fits.iloc[-1].anchor_subspace_angle_deg > 89


def test_yearly_comparisons_are_descriptive_and_reference_aligned(curves):
    result = subperiod_pca_analysis(curves)
    assert set(result.period) == {'2021', '2022'}
    first = result.query('period == "2021"')
    np.testing.assert_allclose(first.anchor_cosine, 1, atol=1e-14)
    assert (result.n_train >= 5).all()


def test_pchip_sensitivity_and_no_extrapolation(panel):
    grid = np.array([15, 45, 75, 105, 135, 180])/365.25
    linear = constant_maturity_curves(panel, grid)
    pchip = constant_maturity_curves(panel, grid, interpolation='pchip')
    assert pchip.iloc[:, [0, -1]].isna().all().all()
    assert not np.allclose(linear.iloc[:, 1:-1], pchip.iloc[:, 1:-1])
    changed = panel.copy()
    cutoff = changed.date_.unique()[30]
    changed.loc[changed.date_ >= cutoff, 'settlement'] *= 2
    altered = constant_maturity_curves(changed, grid, interpolation='pchip')
    pd.testing.assert_frame_equal(pchip.loc[pchip.index < cutoff], altered.loc[altered.index < cutoff])


def test_common_support_sensitivity_and_saved_plots(panel, curves, tmp_path):
    result = run_pca_robustness(panel, maturities=curves.columns, windows=(12, 24), step=12,
        sensitivity_window=24, missing_fraction=.05, min_coverage=.5)
    common = result['sensitivity_common_errors']
    assert not common.empty
    comparisons = result['sensitivity_loading_comparisons']
    np.testing.assert_allclose(comparisons.query('scenario == "baseline"').baseline_cosine, 1., atol=1e-12)
    assert common.groupby(['date_', 'tau', 'method', 'k']).scenario.nunique().eq(6).all()
    np.testing.assert_allclose(common.error_log, common.baseline_log - common.fitted_log)
    # Both window metrics use the exact same held-out observations.
    metrics = result['common_window_metrics'].query('period == "all" and tau == 0')
    assert metrics.groupby(['method', 'k']).n_dates.nunique().eq(1).all()
    destination = save_pca_robustness(result, tmp_path)
    pd.testing.assert_frame_equal(pd.read_parquet(destination/'rolling_spectrum.parquet'), result['rolling_spectrum'])
    figures = plot_pca_robustness(result, window=24)
    assert len(figures) == 5
    for name, fig in figures.items():
        fig.savefig(tmp_path/f'{name}.png')
        plt.close(fig)
    repeated = run_pca_robustness(panel.sample(frac=1, random_state=19), maturities=curves.columns,
        windows=(12, 24), step=12, sensitivity_window=24, missing_fraction=.05, min_coverage=.5)
    pd.testing.assert_frame_equal(result['sensitivity_errors'], repeated['sensitivity_errors'])


def test_robustness_output_protects_existing_static_results(tmp_path):
    (tmp_path/'metadata.json').write_text('{"method": "PCA"}')
    with pytest.raises(ValueError, match='separate'):
        save_pca_robustness({'metadata': {}}, tmp_path)


@pytest.mark.parametrize('kwargs', [dict(windows=(3,)), dict(step=0), dict(min_coverage=0)])
def test_invalid_rolling_configuration_rejected(curves, kwargs):
    with pytest.raises(ValueError):
        rolling_pca_analysis(curves, **kwargs)


def test_rank_deficient_windows_are_reported_not_interpreted(curves):
    values = curves.copy()
    values.iloc[:, :] = np.arange(len(curves))[:, None]
    result = rolling_pca_analysis(values, windows=(20,), transformation='log_prices')
    assert result['fits'].status.eq('rank_below_three').all()
    assert result['errors'].empty


def test_refit_schedule_is_shared_across_different_windows(curves):
    result = rolling_pca_analysis(curves, windows=(12, 20), step=7)
    errors = result['errors']
    shorter = set(errors.query('window == 12').date_)
    longer = set(errors.query('window == 20').date_)
    assert longer and longer.issubset(shorter)
    assert errors.date_.map(curves.index.get_loc).mod(7).eq(0).all()


def test_unscored_missing_scenario_is_explicit(panel, curves):
    result = run_pca_robustness(panel, maturities=curves.columns, windows=(24,), step=24,
                               sensitivity_window=24, missing_fraction=.9)
    assert result['metadata']['unscored_scenarios'] == ['missing_quotes']
    assert 'missing_quotes' not in result['metadata']['common_support_scenarios']
    assert not result['sensitivity_metrics'].empty
    assert result['sensitivity_metrics'].scenario.nunique() == 5
