"""Training-only PCA of constant-maturity curves, separate from forecasting."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.interpolate import PchipInterpolator
from scipy.optimize import linear_sum_assignment
from scipy.linalg import subspace_angles

from commodity_factors.analysis import split_labels
from commodity_factors.data import clean_futures_data, calculate_maturity


DEFAULT_MATURITIES = np.arange(1, 6) / 12


def constant_maturity_curves(panel, maturities=DEFAULT_MATURITIES, *, max_gap_years=62 / 365.25, interpolation="linear"):
    """Interpolate same-day log prices only inside observed maturity brackets.

    No extrapolation or time filling. Gaps wider than max_gap_years are left
    missing. Contract identifiers are never treated as fixed maturity features.
    """
    if interpolation not in ("linear", "pchip"):
        raise ValueError("interpolation must be linear or pchip")
    grid = np.asarray(maturities, dtype=float)
    if (grid.ndim != 1 or len(grid) < 3 or not np.isfinite(grid).all()
            or np.any(grid <= 0) or np.any(np.diff(grid) <= 0)):
        raise ValueError("maturities must be at least three increasing positive finite year fractions")
    if not np.isfinite(max_gap_years) or max_gap_years <= 0:
        raise ValueError("max_gap_years must be positive and finite")
    dates = pd.DatetimeIndex(pd.to_datetime(panel['date_'], errors='coerce').dropna().unique()).sort_values()
    if len(dates) == 0:
        raise ValueError("No valid observation dates")
    data = calculate_maturity(clean_futures_data(panel))
    curves = pd.DataFrame(np.nan, index=dates, columns=grid)
    curves.index.name = 'date_'
    curves.columns.name = 'tau'
    for date, rows in data.groupby('date_'):
        rows = rows.sort_values('tau')
        tau = rows.tau.to_numpy(dtype=float)
        values = rows.log_settlement.to_numpy(dtype=float)
        if len(np.unique(tau)) != len(tau):
            raise ValueError(f"Ambiguous duplicate maturities on {date}")
        interpolator = PchipInterpolator(tau, values, extrapolate=False) if interpolation == "pchip" and len(tau) > 1 else None
        for target in grid:
            exact = np.flatnonzero(np.isclose(tau, target, atol=1e-12, rtol=0))
            if len(exact):
                curves.loc[date, target] = values[exact[0]]
                continue
            right = np.searchsorted(tau, target)
            if 0 < right < len(tau) and tau[right] - tau[right-1] <= max_gap_years + 1e-12:
                curves.loc[date, target] = (float(interpolator(target)) if interpolator is not None else
                    np.interp(target, tau[right-1:right+1], values[right-1:right+1]))
    return curves


class CurvePCA:
    """Full-SVD PCA with a frozen training mean, optional scale and maturity order."""

    def __init__(self, n_components=3, standardize=False):
        self.n_components = n_components
        self.standardize = standardize

    def fit(self, curves):
        values = self._values(curves, fitting=True)
        if (not isinstance(self.n_components, (int, np.integer))
                or not 1 <= self.n_components <= min(len(curves)-1, curves.shape[1])):
            raise ValueError("n_components exceeds the centered training sample dimensions")
        self.maturities_ = curves.columns.copy()
        self.mean_ = values.mean(axis=0)
        centered = values - self.mean_
        if np.linalg.norm(centered) <= np.finfo(float).eps * max(1., np.linalg.norm(values)):
            raise ValueError("PCA requires nonzero training variation")
        self.scale_ = values.std(axis=0, ddof=1) if self.standardize else np.ones(values.shape[1])
        self.scale_[self.scale_ <= np.finfo(float).eps * np.maximum(1., np.abs(self.mean_))] = 1.
        self.pca_ = PCA(n_components=self.n_components, svd_solver='full', whiten=False)
        self.pca_.fit(centered / self.scale_)
        # Fix arbitrary signs for reproducible displays without imposing economic labels.
        for component in self.pca_.components_:
            if component[np.argmax(np.abs(component))] < 0:
                component *= -1
        return self

    def _values(self, curves, fitting=False):
        if not isinstance(curves, pd.DataFrame) or curves.empty:
            raise ValueError("Expected a nonempty maturity-column DataFrame")
        if curves.columns.has_duplicates or (not fitting and not curves.columns.equals(self.maturities_)):
            raise ValueError("Maturity columns must match the fitted order")
        values = curves.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("PCA inputs must be finite; exclude incomplete curves explicitly")
        return values

    def transform(self, curves):
        values = self._values(curves)
        return self.pca_.transform((values - self.mean_) / self.scale_)

    def inverse_transform(self, scores):
        return self.pca_.inverse_transform(scores) * self.scale_ + self.mean_


def loading_interpretations(loadings):
    """Signed cosine alignment with orthogonal constant/linear/quadratic shapes.

    These are empirical shape comparisons, not assigned component identities.
    Columns are physical-unit loading vectors; rows are maturity year fractions.
    """
    tau = loadings.index.to_numpy(dtype=float)
    x = (tau - tau.mean()) / np.ptp(tau)
    templates, _ = np.linalg.qr(np.column_stack([np.ones(len(tau)), x, x*x]))
    for j, reference in enumerate([np.ones(len(tau)), x, x*x]):
        if templates[:, j] @ reference < 0:
            templates[:, j] *= -1
    vectors = loadings.to_numpy()
    cosines = vectors.T @ templates / np.linalg.norm(vectors, axis=0)[:, None]
    return pd.DataFrame(cosines, index=loadings.columns, columns=['level', 'slope', 'curvature'])


@dataclass
class PCAAnalysis:
    model: CurvePCA
    curves: pd.DataFrame
    features: pd.DataFrame
    scores: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance: pd.DataFrame
    interpretations: pd.DataFrame
    reconstructions: pd.DataFrame
    metrics: pd.DataFrame
    coverage: pd.DataFrame
    metadata: dict


def run_pca_analysis(panel, *, maturities=DEFAULT_MATURITIES, transformation='log_changes',
                     n_components=3, standardize=False, validation_start='2024-01-01',
                     test_start='2025-01-01', max_gap_years=62/365.25, max_change_days=7):
    """Fit only pre-validation data; holdout scores use frozen loadings.

    Reconstructions use the current curve, so they are not forecasts. Change
    reconstruction adds the reconstructed increment to the previous observed
    curve. Gaps above max_change_days are excluded rather than annualized.
    """
    curves = constant_maturity_curves(panel, maturities, max_gap_years=max_gap_years)
    splits = pd.Series(split_labels(curves.index, validation_start, test_start), index=curves.index)
    features, gaps = _curve_features(curves, transformation, max_change_days)
    complete = features.notna().all(axis=1)
    train = complete & splits.eq('train')
    model = CurvePCA(n_components, standardize).fit(features.loc[train])
    names = [f'PC{i+1}' for i in range(n_components)]
    scores = pd.DataFrame(np.nan, index=curves.index, columns=names)
    scores.loc[complete] = model.transform(features.loc[complete])
    reconstructed = pd.DataFrame(np.nan, index=curves.index, columns=curves.columns)
    reconstructed.loc[complete] = model.inverse_transform(scores.loc[complete].to_numpy())
    fitted_log = reconstructed if transformation == 'log_prices' else curves.shift() + reconstructed
    loadings = pd.DataFrame(model.pca_.components_.T * model.scale_[:, None], index=curves.columns, columns=names)
    spectrum = pd.DataFrame({'component': names, 'variance': model.pca_.explained_variance_,
                             'ratio': model.pca_.explained_variance_ratio_,
                             'cumulative_ratio': model.pca_.explained_variance_ratio_.cumsum()})
    frames = []
    for maturity in curves.columns:
        frames.append(pd.DataFrame({'date_': curves.index, 'tau': maturity, 'split': splits.to_numpy(),
            'observed_log': curves[maturity].to_numpy(), 'fitted_log': fitted_log[maturity].to_numpy(),
            'observed_feature': features[maturity].to_numpy(),
            'fitted_feature': reconstructed[maturity].to_numpy()}))
    reconstructions = pd.concat(frames, ignore_index=True)
    reconstructions['error_log'] = reconstructions.observed_log - reconstructions.fitted_log
    metric_rows = []
    scored = reconstructions.dropna(subset=['fitted_log'])
    for split, group in scored.groupby('split'):
        for maturity, subset in [(0., group), *group.groupby('tau')]:
            error = subset.error_log.to_numpy()
            metric_rows.append(dict(split=split, tau=maturity, rmse_log=float(np.sqrt(np.mean(error**2))),
                                    mae_log=float(np.mean(np.abs(error))), n=len(subset),
                                    n_dates=subset.date_.nunique()))
    coverage = pd.DataFrame({'split': splits, 'available_maturities': curves.notna().sum(axis=1),
                             'curve_complete': curves.notna().all(axis=1), 'feature_complete': complete,
                             'elapsed_days': gaps})
    metadata = dict(method='PCA', transformation=transformation, standardize=bool(standardize),
                    n_components=int(n_components), maturities_years=curves.columns.tolist(),
                    validation_start=str(pd.Timestamp(validation_start).date()),
                    test_start=str(pd.Timestamp(test_start).date()),
                    train_end=str(features.index[train][-1].date()), n_training_rows=int(train.sum()),
                    max_interpolation_gap_years=float(max_gap_years), max_change_days=float(max_change_days),
                    fitted_rank=int(np.linalg.matrix_rank((features.loc[train].to_numpy()-model.mean_) / model.scale_)),
                    coverage_by_split={s: {'dates': int((splits == s).sum()),
                                          'scored_dates': int((complete & splits.eq(s)).sum())}
                                       for s in ('train', 'validation', 'test')},
                    target='constant_maturity', reconstruction_is_forecast=False)
    return PCAAnalysis(model, curves, features, scores, loadings, spectrum,
                       loading_interpretations(loadings), reconstructions,
                       pd.DataFrame(metric_rows), coverage, metadata)


def save_pca_analysis(result, directory):
    """Save PCA-only tables and numeric fitted parameters without pickle."""
    directory = Path(directory)
    metadata_path = directory / 'metadata.json'
    if metadata_path.exists() and json.loads(metadata_path.read_text()).get('method') != 'PCA':
        raise ValueError('PCA output must be separate from existing non-PCA results')
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('curves', 'features', 'scores', 'loadings', 'explained_variance',
                 'interpretations', 'reconstructions', 'metrics', 'coverage'):
        getattr(result, name).to_parquet(directory / f'{name}.parquet')
    model = result.model
    np.savez(directory / 'model.npz', maturities=model.maturities_.to_numpy(dtype=float),
             mean=model.mean_, scale=model.scale_, pca_mean=model.pca_.mean_,
             components=model.pca_.components_, explained_variance=model.pca_.explained_variance_,
             explained_variance_ratio=model.pca_.explained_variance_ratio_)
    (directory / 'metadata.json').write_text(json.dumps(result.metadata, indent=2, allow_nan=False) + '\n')
    return directory


def _curve_features(curves, transformation, max_change_days):
    if transformation not in ('log_prices', 'log_changes'):
        raise ValueError("transformation must be log_prices or log_changes")
    if not np.isfinite(max_change_days) or max_change_days <= 0:
        raise ValueError("max_change_days must be positive and finite")
    gaps = curves.index.to_series().diff().dt.total_seconds() / 86400
    features = curves.copy() if transformation == 'log_prices' else curves.diff()
    if transformation == 'log_changes':
        features.loc[gaps > max_change_days] = np.nan
    return features, gaps


def align_pca_loadings(reference, candidate):
    """Match unit loading columns by maximum absolute cosine, then align signs.

    Returns aligned directions, source column indices and signed multipliers.
    This is diagnostic only: never reorder variance-ranked reconstruction PCs.
    """
    reference, candidate = np.asarray(reference, float), np.asarray(candidate, float)
    if (reference.ndim != 2 or reference.shape != candidate.shape
            or not np.isfinite(reference).all() or not np.isfinite(candidate).all()
            or np.any(np.linalg.norm(reference, axis=0) == 0)
            or np.any(np.linalg.norm(candidate, axis=0) == 0)):
        raise ValueError('Loadings must be matching finite matrices with nonzero columns')
    reference = reference / np.linalg.norm(reference, axis=0)
    candidate = candidate / np.linalg.norm(candidate, axis=0)
    cosine = reference.T @ candidate
    _, order = linear_sum_assignment(-np.abs(cosine))
    signs = np.where(cosine[np.arange(len(order)), order] < 0, -1., 1.)
    return candidate[:, order] * signs, order, signs


def _physical_directions(model):
    directions = model.pca_.components_[:3].T * model.scale_[:, None]
    return directions / np.linalg.norm(directions, axis=0)


def reconstruction_metrics(errors, keys=('window', 'method')):
    """RMSE in original log units by calendar year/maturity, plus pooled rows."""
    rows = []
    if errors.empty:
        return pd.DataFrame(columns=[*keys, 'period', 'tau', 'rmse_log', 'n', 'n_dates'])
    for key, group in errors.groupby(list(keys)):
        if not isinstance(key, tuple):
            key = (key,)
        for period, period_group in [('all', group), *group.groupby(group.date_.dt.year.astype(str))]:
            for tau, subset in [(0., period_group), *period_group.groupby('tau')]:
                rows.append(dict(zip(keys, key)) | dict(period=period, tau=float(tau),
                    rmse_log=float(np.sqrt(np.mean(subset.error_log**2))),
                    n=len(subset), n_dates=subset.date_.nunique()))
    return pd.DataFrame(rows)


def rolling_pca_analysis(curves, *, windows=(126, 252, 504), step=1,
                         transformation='log_changes', max_change_days=7, min_coverage=.8):
    """Refit before each sampled date using the preceding W supplied trading rows.

    Missing rows reduce the fit sample, not the span. K=1,2,3 reconstructions use
    held-out current curves and the same prior-only model. Both PCA scalings run.
    """
    windows = tuple(windows)
    if (not windows or any(not isinstance(w, (int, np.integer)) or w < 5 for w in windows)
            or len(set(windows)) != len(windows) or not isinstance(step, (int, np.integer)) or step < 1
            or not 0 < min_coverage <= 1):
        raise ValueError('Use unique windows >=5, positive integer step and coverage in (0,1]')
    if (not isinstance(curves.index, pd.DatetimeIndex) or not curves.index.is_unique
            or not curves.index.is_monotonic_increasing or curves.shape[1] < 3
            or np.isinf(curves.to_numpy(dtype=float)).any()):
        raise ValueError('Curves require unique increasing dates, >=3 maturities and finite values or NaN')
    features, _ = _curve_features(curves, transformation, max_change_days)
    previous_curves = curves.shift()
    fit_rows, spectra, loadings, errors = [], [], [], []
    for window in windows:
        for standardize in (False, True):
            method = 'correlation' if standardize else 'covariance'
            anchor = previous = None
            for t in range(window + (-window % step), len(curves), step):
                date = curves.index[t]
                train = features.iloc[t-window:t].dropna()
                row = dict(date_=date, window=window, method=method,
                           window_start=curves.index[t-window], train_end=curves.index[t-1],
                           n_train=len(train), coverage=len(train)/window)
                if len(train) < max(5, int(np.ceil(window * min_coverage))):
                    fit_rows.append(row | dict(status='insufficient_coverage', scored=False))
                    continue
                if np.linalg.matrix_rank(train.to_numpy()-train.to_numpy().mean(axis=0)) < 3:
                    fit_rows.append(row | dict(status='rank_below_three', scored=False))
                    continue
                model = CurvePCA(min(4, curves.shape[1]), standardize).fit(train)
                directions = _physical_directions(model)
                if previous is None:
                    anchor = previous = directions.copy()
                aligned, order, signs = align_pca_loadings(previous, directions)
                angle = float(np.degrees(subspace_angles(anchor, directions)).max())
                eligible = bool(features.iloc[t].notna().all())
                fit_rows.append(row | dict(status='ok', scored=eligible, anchor_subspace_angle_deg=angle))
                variance = model.pca_.explained_variance_
                for i in range(3):
                    source = int(order[i])
                    gap = float((variance[source]-variance[source+1])/variance[source]) if source+1 < len(variance) else np.nan
                    spectra.append(dict(date_=date, window=window, method=method, track=i+1,
                        source_rank=source+1, variance_ratio=float(model.pca_.explained_variance_ratio_[source]),
                        relative_eigengap=gap, sign=float(signs[i]),
                        previous_cosine=float(aligned[:, i] @ previous[:, i]),
                        anchor_cosine=float(aligned[:, i] @ anchor[:, i])))
                    for j, tau in enumerate(curves.columns):
                        loadings.append(dict(date_=date, window=window, method=method, track=i+1,
                                             source_rank=source+1, tau=float(tau), loading=float(aligned[j, i])))
                previous = aligned
                if not eligible:
                    continue
                scores = model.transform(features.iloc[[t]])
                for k in (1, 2, 3):
                    truncated = scores.copy()
                    truncated[:, k:] = 0
                    fitted = model.inverse_transform(truncated)[0]
                    if transformation == 'log_changes':
                        fitted = previous_curves.iloc[t].to_numpy() + fitted
                    for j, tau in enumerate(curves.columns):
                        observed = float(curves.iloc[t, j])
                        errors.append(dict(date_=date, window=window, method=method, k=k, tau=float(tau),
                            observed_log=observed, fitted_log=float(fitted[j]), error_log=observed-float(fitted[j])))
    errors = pd.DataFrame(errors)
    return dict(fits=pd.DataFrame(fit_rows), spectrum=pd.DataFrame(spectra), loadings=pd.DataFrame(loadings),
                errors=errors, metrics=reconstruction_metrics(errors, keys=('window', 'method', 'k')))


def subperiod_pca_analysis(curves, *, transformation='log_changes', max_change_days=7):
    """Descriptive calendar-year fits, including 2022; not held-out estimates."""
    features, _ = _curve_features(curves, transformation, max_change_days)
    rows = []
    for standardize in (False, True):
        anchor = None
        for year, sample in features.groupby(features.index.year):
            sample = sample.dropna()
            if len(sample) < 5 or np.linalg.matrix_rank(sample.to_numpy()-sample.to_numpy().mean(axis=0)) < 3:
                continue
            model = CurvePCA(min(4, curves.shape[1]), standardize).fit(sample)
            directions = _physical_directions(model)
            if anchor is None:
                anchor = directions.copy()
            aligned, order, _ = align_pca_loadings(anchor, directions)
            angle = float(np.degrees(subspace_angles(anchor, directions)).max())
            for i in range(3):
                for j, tau in enumerate(curves.columns):
                    rows.append(dict(period=str(year), method='correlation' if standardize else 'covariance',
                        track=i+1, source_rank=int(order[i])+1, tau=float(tau), loading=float(aligned[j, i]),
                        anchor_cosine=float(aligned[:, i] @ anchor[:, i]), anchor_subspace_angle_deg=angle,
                        variance_ratio=float(model.pca_.explained_variance_ratio_[order[i]]), n_train=len(sample),
                        feature_rms_sd=float(np.sqrt(sample.var(ddof=1).mean()))))
    return pd.DataFrame(rows)


def run_pca_robustness(panel, *, maturities=DEFAULT_MATURITIES, windows=(126, 252, 504), step=1,
                       transformation='log_changes', max_gap_years=62/365.25,
                       max_change_days=7, min_coverage=.8, sensitivity=True,
                       sensitivity_window=252, missing_fraction=.1, seed=1729):
    """Rolling, yearly and common-support robustness comparisons.

    Sensitivities use one selected window, the same refit schedule, and baseline
    observed log curves as the comparison target on shared dates/maturities.
    """
    if not 0 <= missing_fraction < 1:
        raise ValueError('missing_fraction must lie in [0,1)')
    curves = constant_maturity_curves(panel, maturities, max_gap_years=max_gap_years)
    options = dict(step=step, transformation=transformation, max_change_days=max_change_days,
                   min_coverage=min_coverage)
    baseline = rolling_pca_analysis(curves, windows=windows, **options)
    if baseline['errors'].empty:
        raise ValueError('No eligible held-out dates; reduce windows or review coverage')
    result = {f'rolling_{key}': value for key, value in baseline.items()}
    result['subperiod_loadings'] = subperiod_pca_analysis(curves, transformation=transformation,
                                                        max_change_days=max_change_days)
    result['curves'] = curves
    scenarios = {}
    if sensitivity:
        # Stable ordering makes the seeded quote-loss experiment invariant to row ordering.
        missing = panel.sort_values(['date_', 'futcode']).copy()
        rng = np.random.default_rng(seed)
        missing.loc[rng.random(len(missing)) < missing_fraction, 'settlement'] = np.nan
        grid = curves.columns.to_numpy(dtype=float)
        shorter = grid[:-1] if len(grid) > 3 else grid
        dense = np.sort(np.concatenate([grid, (grid[:-1]+grid[1:])/2]))
        variants = dict(baseline=curves,
            strict_brackets=constant_maturity_curves(panel, grid, max_gap_years=min(max_gap_years, 35/365.25)),
            pchip=constant_maturity_curves(panel, grid, max_gap_years=max_gap_years, interpolation='pchip'),
            shorter_grid=constant_maturity_curves(panel, shorter, max_gap_years=max_gap_years),
            denser_grid=constant_maturity_curves(panel, dense, max_gap_years=max_gap_years),
            missing_quotes=constant_maturity_curves(missing, grid, max_gap_years=max_gap_years))
        fits, error_frames, sensitivity_spectra, sensitivity_loadings = [], [], [], []
        for name, aligned_curves in variants.items():
            if name == 'baseline' and sensitivity_window in windows:
                run = {key: value[value.window == sensitivity_window] if 'window' in value else value
                       for key, value in baseline.items()}
            else:
                run = rolling_pca_analysis(aligned_curves, windows=(sensitivity_window,), **options)
            fits.append(run['fits'].assign(scenario=name))
            sensitivity_spectra.append(run['spectrum'].assign(scenario=name))
            sensitivity_loadings.append(run['loadings'].assign(scenario=name))
            error_frames.append(run['errors'].assign(scenario=name))
            scenarios[name] = dict(maturities_years=aligned_curves.columns.tolist(),
                complete_curves=int(aligned_curves.notna().all(axis=1).sum()), supplied_dates=len(aligned_curves),
                scored_dates=int(run['errors'].date_.nunique()) if not run['errors'].empty else 0)
        all_errors = pd.concat(error_frames, ignore_index=True)
        result['sensitivity_fits'] = pd.concat(fits, ignore_index=True)
        result['sensitivity_errors'] = all_errors
        result['sensitivity_spectrum'] = pd.concat(sensitivity_spectra, ignore_index=True)
        result['sensitivity_loadings'] = pd.concat(sensitivity_loadings, ignore_index=True)
        result['sensitivity_loading_comparisons'] = _sensitivity_loading_comparisons(result['sensitivity_loadings'])
        # Compare identical targets, not just each scenario's own smoothed observations.
        support = all_errors.groupby(['date_', 'tau', 'method', 'k']).scenario.nunique()
        available_scenarios = [name for name, details in scenarios.items() if details['scored_dates'] > 0]
        shared = support[support == len(available_scenarios)].reset_index()[['date_', 'tau', 'method', 'k']]
        common = all_errors.merge(shared, on=['date_', 'tau', 'method', 'k'])
        targets = curves.rename_axis(index='date_', columns='tau').stack().rename('baseline_log').reset_index()
        common = common.merge(targets, on=['date_', 'tau'])
        common['error_log'] = common.baseline_log - common.fitted_log
        result['sensitivity_common_errors'] = common
        result['sensitivity_metrics'] = reconstruction_metrics(common, keys=('scenario', 'method', 'k'))
    # Window comparisons must also hold date support constant.
    support = baseline['errors'].groupby(['date_', 'tau', 'method', 'k']).window.nunique()
    shared = support[support == len(windows)].reset_index()[['date_', 'tau', 'method', 'k']]
    common = baseline['errors'].merge(shared, on=['date_', 'tau', 'method', 'k'])
    result['common_window_metrics'] = reconstruction_metrics(common, keys=('window', 'method', 'k'))
    result['metadata'] = dict(method='PCA robustness', windows=list(windows), step=int(step),
        transformation=transformation, min_coverage=float(min_coverage),
        max_interpolation_gap_years=float(max_gap_years), max_change_days=float(max_change_days),
        sensitivity_window=int(sensitivity_window), missing_fraction=float(missing_fraction), seed=int(seed),
        scenarios=scenarios, maturity_years=curves.columns.tolist(),
        common_support_scenarios=[name for name, details in scenarios.items() if details['scored_dates'] > 0],
        unscored_scenarios=[name for name, details in scenarios.items() if details['scored_dates'] == 0],
        alignment='Sequential signed maximum-cosine assignment of physical-unit directions',
        evaluation='Prior-only fit; held-out same-day reconstruction, not forecasting')
    return result


def save_pca_robustness(result, directory):
    directory = Path(directory)
    metadata_path = directory / 'metadata.json'
    if metadata_path.exists() and json.loads(metadata_path.read_text()).get('method') != 'PCA robustness':
        raise ValueError('Robustness output must be separate from existing results')
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in result.items():
        if isinstance(table, pd.DataFrame):
            table.to_parquet(directory / f'{name}.parquet')
    metadata_path.write_text(json.dumps(result['metadata'], indent=2, allow_nan=False) + '\n')
    return directory


def _sensitivity_loading_comparisons(loadings):
    """Compare prior-fitted directions on each scenario's shared baseline knots."""
    rows = []
    baseline = loadings[loadings.scenario == 'baseline']
    references = {(date, method): group.pivot(index='tau', columns='track', values='loading')
                  for (date, method), group in baseline.groupby(['date_', 'method'])}
    for (scenario, date, method), group in loadings.groupby(['scenario', 'date_', 'method']):
        if (date, method) not in references:
            continue
        reference = references[date, method]
        candidate = group.pivot(index='tau', columns='track', values='loading')
        common = reference.index.intersection(candidate.index)
        reference = reference.loc[common].to_numpy()
        candidate = candidate.loc[common].to_numpy()
        if len(common) < 3 or min(np.linalg.matrix_rank(reference), np.linalg.matrix_rank(candidate)) < 3:
            continue
        aligned, _, _ = align_pca_loadings(reference, candidate)
        reference /= np.linalg.norm(reference, axis=0)
        angle = float(np.degrees(subspace_angles(reference, candidate)).max())
        for i in range(3):
            rows.append(dict(scenario=scenario, date_=date, method=method, track=i+1,
                baseline_cosine=float(reference[:, i] @ aligned[:, i]), subspace_angle_deg=angle,
                common_maturities=len(common)))
    return pd.DataFrame(rows)
