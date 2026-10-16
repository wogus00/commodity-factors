"""PCA subspace recovery, maturity alignment and train-only preprocessing."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from commodity_factors.pca_baseline import (
    CurvePCA, constant_maturity_curves, loading_interpretations,
    run_pca_analysis, save_pca_analysis,
)
from commodity_factors.plotting import plot_pca_analysis


@pytest.fixture
def factor_curves():
    rng = np.random.default_rng(17)
    tau = np.arange(1, 7) / 12
    x = tau - tau.mean()
    basis, _ = np.linalg.qr(np.column_stack([np.ones(6), x, x*x]))
    # Deliberately put curvature first, level second and slope third.
    loadings = basis[:, [2, 0, 1]]
    raw = rng.normal(size=(180, 3))
    scores, _ = np.linalg.qr(raw - raw.mean(axis=0))
    scores *= [1.2, .6, .2]
    values = 4. + .1 * tau + scores @ loadings.T
    return pd.DataFrame(values, columns=tau), loadings


@pytest.fixture
def historical_panel():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range('2023-01-02', periods=180)
    days = np.arange(30, 181, 30)
    x = days / 365.25
    q, _ = np.linalg.qr(np.column_stack([np.ones(6), x, x*x]))
    factors = rng.normal(size=(len(dates), 3)) * [.03, .015, .005]
    values = 4. + factors @ q.T
    return pd.DataFrame({'date_': np.repeat(dates, 6),
        'lasttrddate': np.repeat(dates, 6) + pd.to_timedelta(np.tile(days, len(dates)), unit='D'),
        'futcode': np.tile(np.arange(6), len(dates)), 'settlement': np.exp(values.ravel())})


def run(panel, **kwargs):
    return run_pca_analysis(panel, maturities=np.arange(30, 181, 30)/365.25,
                            validation_start='2023-05-01', test_start='2023-07-01', **kwargs)


def test_known_subspace_recovery_and_nonstandard_economic_order(factor_curves):
    curves, basis = factor_curves
    model = CurvePCA().fit(curves)
    components = model.pca_.components_
    np.testing.assert_allclose(components.T @ components, basis @ basis.T, atol=1e-13)
    np.testing.assert_allclose(np.abs(components @ basis), np.eye(3), atol=1e-13)
    np.testing.assert_allclose(model.pca_.explained_variance_ratio_, np.array([1.2, .6, .2])**2 / 1.84)
    np.testing.assert_allclose(model.inverse_transform(model.transform(curves)), curves, atol=1e-13)
    interpretation = loading_interpretations(pd.DataFrame(components.T, index=curves.columns))
    assert interpretation.abs().idxmax(axis=1).tolist() == ['curvature', 'level', 'slope']


@pytest.mark.parametrize('standardize', [False, True])
def test_full_rank_reconstruction_and_standardization(standardize):
    rng = np.random.default_rng(6)
    curves = pd.DataFrame(rng.normal(size=(40, 5)) * [1, 10, .01, 2, .2] + 4.)
    model = CurvePCA(5, standardize=standardize).fit(curves)
    np.testing.assert_allclose(model.inverse_transform(model.transform(curves)), curves, atol=1e-12)
    np.testing.assert_allclose(model.scale_, curves.std(ddof=1) if standardize else np.ones(5))


def test_reconstruction_error_decreases_and_matches_discarded_variance():
    rng = np.random.default_rng(7)
    curves = pd.DataFrame(rng.normal(size=(100, 6)))
    full = CurvePCA(6).fit(curves)
    errors = []
    for k in range(1, 7):
        model = CurvePCA(k).fit(curves)
        residual = curves.to_numpy() - model.inverse_transform(model.transform(curves))
        errors.append(np.sum(residual**2))
        assert errors[-1] == pytest.approx(99 * full.pca_.explained_variance_[k:].sum(), abs=1e-10)
    assert np.all(np.diff(errors) < 0)


def test_rank_deficient_and_ill_conditioned_inputs():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(80, 3)) * [1, 1e-7, 1e-12]
    curves = pd.DataFrame(np.column_stack([values, values[:, 0], np.ones(80)]))
    model = CurvePCA(3, standardize=True).fit(curves)
    assert np.isfinite(model.pca_.explained_variance_ratio_).all()
    assert np.isfinite(model.transform(curves)).all()
    np.testing.assert_allclose(model.inverse_transform(model.transform(curves)), curves, atol=1e-12)
    with pytest.raises(ValueError, match='variation'):
        CurvePCA().fit(pd.DataFrame(np.ones((10, 5))))


def test_column_order_and_incomplete_inputs_rejected(factor_curves):
    curves, _ = factor_curves
    model = CurvePCA().fit(curves)
    with pytest.raises(ValueError, match='order'):
        model.transform(curves.iloc[:, ::-1])
    curves.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        model.transform(curves)


def test_fixed_maturity_interpolation_removes_rank_roll_jump():
    # An unchanged linear log curve: nearby rank 1 jumps when A expires.
    rows = []
    for date in pd.to_datetime(['2024-01-01', '2024-01-11']):
        for code, expiry in zip('ABCD', pd.to_datetime(['2024-01-10', '2024-02-10', '2024-03-10', '2024-04-10'])):
            tau = (expiry-date).days/365.25
            if tau > 0:
                rows.append(dict(date_=date, lasttrddate=expiry, futcode=code,
                                 settlement=np.exp(4. + .5*tau)))
    panel = pd.DataFrame(rows)
    grid = np.array([40, 50, 60])/365.25
    aligned = constant_maturity_curves(panel, grid)
    np.testing.assert_allclose(aligned.iloc[0], aligned.iloc[1], atol=1e-14)
    assert panel.groupby('date_').first().settlement.nunique() == 2


def test_missing_contract_gap_no_extrapolation_or_time_fill(historical_panel):
    dates = historical_panel.date_.unique()
    panel = historical_panel.copy()
    # Remove one interior contract (60-day bracket permitted), then three (120-day gap refused).
    panel = panel[~((panel.date_ == dates[1]) & panel.futcode.isin([2]))]
    panel = panel[~((panel.date_ == dates[2]) & panel.futcode.isin([1, 2, 3]))]
    panel.loc[panel.date_ == dates[3], 'settlement'] = np.nan
    grid = np.array([15, 60, 90, 120, 210])/365.25
    curves = constant_maturity_curves(panel, grid)
    assert curves.iloc[:, [0, -1]].isna().all().all()
    assert np.isfinite(curves.iloc[1, 2])
    assert curves.iloc[2, 1:4].isna().all()
    assert curves.iloc[3].isna().all()
    changed = panel.copy()
    changed.loc[changed.date_ > dates[3], 'settlement'] *= 2
    pd.testing.assert_frame_equal(curves.iloc[:4], constant_maturity_curves(changed, grid).iloc[:4])


@pytest.mark.parametrize('transformation', ['log_prices', 'log_changes'])
@pytest.mark.parametrize('standardize', [False, True])
def test_holdout_cannot_change_training_or_earlier_scores(historical_panel, transformation, standardize):
    result = run(historical_panel, transformation=transformation, standardize=standardize)
    changed = historical_panel.copy()
    changed.loc[changed.date_ >= '2023-07-01', 'settlement'] *= np.tile([1., 1.1, 1.2, 1.3, 1.4, 1.5],
                                                                 (changed.date_ >= '2023-07-01').sum()//6)
    perturbed = run(changed, transformation=transformation, standardize=standardize)
    np.testing.assert_array_equal(result.model.mean_, perturbed.model.mean_)
    np.testing.assert_array_equal(result.model.scale_, perturbed.model.scale_)
    pd.testing.assert_frame_equal(result.loadings, perturbed.loadings)
    pd.testing.assert_frame_equal(result.explained_variance, perturbed.explained_variance)
    pd.testing.assert_frame_equal(result.scores.loc[:'2023-06-30'], perturbed.scores.loc[:'2023-06-30'])
    assert not np.allclose(result.scores.loc['2023-07-01':], perturbed.scores.loc['2023-07-01':])


def test_missing_date_breaks_changes_and_long_gap_excluded(historical_panel):
    panel = historical_panel.copy()
    date = panel.date_.unique()[20]
    panel.loc[panel.date_ == date, 'settlement'] = np.nan
    result = run(panel)
    assert result.features.loc[date:].iloc[:2].isna().all().all()
    panel = panel[~panel.date_.between('2023-02-01', '2023-02-15')]
    result = run(panel)
    assert result.features.loc['2023-02-16'].isna().all()
    assert result.coverage.loc['2023-02-16', 'elapsed_days'] > 7


def test_change_reconstruction_metrics_serialization_and_plot(historical_panel, tmp_path):
    result = run(historical_panel)
    complete = result.coverage.feature_complete
    reconstructed_changes = result.model.inverse_transform(result.scores.loc[complete].to_numpy())
    fitted = result.reconstructions.pivot(index='date_', columns='tau', values='fitted_log')
    np.testing.assert_allclose(fitted.loc[complete], result.curves.shift().loc[complete] + reconstructed_changes)
    test_rows = result.reconstructions.query("split == 'test'")
    expected = np.sqrt(np.nanmean(test_rows.error_log**2))
    assert result.metrics.query("split == 'test' and tau == 0").rmse_log.iloc[0] == pytest.approx(expected)
    destination = save_pca_analysis(result, tmp_path)
    pd.testing.assert_frame_equal(pd.read_parquet(destination/'scores.parquet'), result.scores)
    saved = np.load(destination/'model.npz')
    projected = ((result.features.loc[complete].to_numpy()-saved['mean'])/saved['scale']-saved['pca_mean']) @ saved['components'].T
    np.testing.assert_allclose(projected, result.scores.loc[complete], atol=1e-14)
    fig = plot_pca_analysis(result)
    assert len(fig.axes) == 4
    fig.savefig(tmp_path/'plot.png')
    plt.close(fig)
    assert (tmp_path/'plot.png').stat().st_size > 1000


def test_ambiguous_maturities_rejected(historical_panel):
    extra = historical_panel.iloc[[0]].copy()
    extra.futcode = 99
    with pytest.raises(ValueError, match='duplicate maturities'):
        constant_maturity_curves(pd.concat([historical_panel, extra]))


@pytest.mark.parametrize('grid', [[.1, .1, .2], [.2, .1, .3], [-.1, .1, .2], [.1, np.nan, .3]])
def test_invalid_maturity_grid_rejected(historical_panel, grid):
    with pytest.raises(ValueError, match='maturities'):
        constant_maturity_curves(historical_panel, grid)


def test_pca_cannot_overwrite_forecasting_bundle(historical_panel, tmp_path):
    (tmp_path/'metadata.json').write_text('{"schema_version": 1}')
    with pytest.raises(ValueError, match='separate'):
        save_pca_analysis(run(historical_panel), tmp_path)
    assert (tmp_path/'metadata.json').read_text() == '{"schema_version": 1}'
