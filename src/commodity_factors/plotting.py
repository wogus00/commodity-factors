"""Static PCA diagnostics, separate from the forecasting dashboard."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_pca_analysis(result, date=None):
    """Return a four-panel figure; default curve is the last scored observation."""
    scored = result.reconstructions.dropna(subset=['fitted_log'])
    date = scored.date_.max() if date is None else pd.Timestamp(date)
    curve = scored[scored.date_ == date].sort_values('tau')
    if curve.empty:
        raise ValueError('No reconstructed curve at the requested date')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    spectrum = result.explained_variance
    ax = axes[0, 0]
    ax.bar(spectrum.component, spectrum.ratio * 100, label='Individual')
    ax.plot(spectrum.component, spectrum.cumulative_ratio * 100, 'o-', color='black', label='Cumulative')
    ax.set(title='Training explained variance', ylabel='Variance (%)', ylim=(0, 105))
    for i, ratio in enumerate(spectrum.ratio):
        ax.text(i, ratio * 100 + 1, f'{ratio * 100:.4f}%', ha='center', fontsize=9)
    ax.legend()
    ax = axes[0, 1]
    for name in result.loadings:
        ax.plot(result.loadings.index * 12, result.loadings[name], 'o-', label=name)
    ax.axhline(0, color='grey', linewidth=.7)
    ax.set(title='Component loadings (original feature units)', xlabel='Maturity (months)', ylabel='Loading')
    ax.legend()
    ax = axes[1, 0]
    for name, variance in zip(result.scores, spectrum.variance):
        if variance > np.finfo(float).eps * spectrum.variance.max():
            ax.plot(result.scores.index, result.scores[name] / np.sqrt(variance),
                    label=name, linewidth=.7, alpha=.75)
    for boundary in ('validation_start', 'test_start'):
        ax.axvline(pd.Timestamp(result.metadata[boundary]), color='grey', linestyle='--', linewidth=1)
    ax.set(title='Scores / training SD (display only)', ylabel='Standard deviations')
    ax.tick_params(axis='x', rotation=20)
    ax.legend()
    ax = axes[1, 1]
    ax.plot(curve.tau * 12, np.exp(curve.observed_log), 'o-', label='Observed (interpolated)')
    ax.plot(curve.tau * 12, np.exp(curve.fitted_log), 'x--', label='Reconstructed')
    ax.set(title=f'Curve reconstruction: {date:%Y-%m-%d}', xlabel='Maturity (months)', ylabel='Futures price')
    ax.legend()
    fig.suptitle(f"PCA · {result.metadata['transformation']} · same-day curve reconstruction", fontsize=14)
    return fig


def plot_pca_robustness(result, window=252):
    """Return five report figures; loading heatmaps use covariance PCA."""
    spectrum = result['rolling_spectrum']
    if window not in spectrum.window.unique():
        window = int(spectrum.window.iloc[0])
    figures = {}
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True, sharex=True)
    for rank, ax in enumerate(axes, 1):
        for (w, method), rows in spectrum[spectrum.source_rank == rank].groupby(['window', 'method']):
            ax.plot(rows.date_, rows.variance_ratio * 100, label=f'{w}d {method}', linewidth=1,
                    linestyle='--' if method == 'correlation' else '-')
        ax.set(ylabel=f'PC{rank} variance (%)')
    axes[0].legend(ncol=3, fontsize=8)
    fig.suptitle('Prior-window explained variance · components ordered by variance')
    figures['rolling_variance'] = fig

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    selected = result['rolling_loadings'].query('window == @window and method == "covariance"')
    for track in (1, 2, 3):
        values = selected[selected.track == track].pivot(index='tau', columns='date_', values='loading')
        ax = axes[track-1, 0]
        mesh = ax.pcolormesh(values.columns, values.index*12, values.to_numpy(), shading='nearest', cmap='RdBu_r')
        fig.colorbar(mesh, ax=ax, label='Unit loading')
        ax.set(title=f'Aligned track {track}', ylabel='Maturity (months)')
        ax = axes[track-1, 1]
        rows = spectrum.query('window == @window and method == "covariance" and track == @track')
        ax.plot(rows.date_, np.degrees(np.arccos(np.clip(rows.anchor_cosine, -1, 1))),
                label='Angle to first window')
        ax.plot(rows.date_, np.degrees(np.arccos(np.clip(rows.previous_cosine, -1, 1))),
                label='Angle to previous window', alpha=.7)
        ax.set(ylabel='Loading direction angle (degrees)', ylim=(0, None))
        ax.legend(fontsize=8)
    fig.suptitle(f'Loading direction stability · {window} supplied trading dates · covariance PCA')
    figures['loading_stability'] = fig

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    metrics = result['rolling_metrics']
    for method, ax in zip(('covariance', 'correlation'), axes):
        rows = metrics.query('window == @window and method == @method and tau == 0 and period != "all"')
        for period, group in rows.groupby('period'):
            ax.plot(group.k, group.rmse_log, 'o-', label=period)
        ax.set(title=method, xlabel='Retained components K', ylabel='Held-out log RMSE', xticks=[1, 2, 3], yscale='log')
        ax.legend()
    fig.suptitle(f'Prior-only reconstruction · {window}-date window · calendar-year evaluation')
    figures['reconstruction_by_k'] = fig

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    subperiods = result['subperiod_loadings']
    for i, method in enumerate(('covariance', 'correlation')):
        for j, track in enumerate((1, 2, 3)):
            ax = axes[i, j]
            for period, rows in subperiods.query('method == @method and track == @track').groupby('period'):
                ax.plot(rows.tau*12, rows.loading, 'o-', label=period)
            ax.set(title=f'{method} · aligned track {track}', xlabel='Maturity (months)', ylabel='Unit loading')
            ax.legend(fontsize=8)
    fig.suptitle('Descriptive annual loadings · aligned to earliest year · not held-out fits')
    figures['subperiod_loadings'] = fig

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for i, method in enumerate(('covariance', 'correlation')):
        for j, k in enumerate((1, 2, 3)):
            ax = axes[i, j]
            rows = metrics.query('window == @window and method == @method and k == @k and tau > 0 and period != "all"')
            for period, group in rows.groupby('period'):
                ax.plot(group.tau*12, group.rmse_log, 'o-', label=period)
            ax.set(title=f'{method} · K={k}', xlabel='Maturity (months)', ylabel='Held-out log RMSE', yscale='log')
            ax.legend(fontsize=8)
    fig.suptitle(f'Reconstruction error by maturity · {window}-date prior window')
    figures['reconstruction_by_maturity'] = fig
    return figures
