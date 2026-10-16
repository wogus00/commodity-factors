#!/usr/bin/env python3
"""Export saved factor analyses only; never fit models or load forecast tables."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def table(frame):
    # Python JSON uses round-trip float representation; gzip provides compression.
    def scalar(value):
        if pd.isna(value):
            return None
        if isinstance(value, (pd.Timestamp, np.datetime64)):
            return pd.Timestamp(value).isoformat()
        return value.item() if isinstance(value, np.generic) else value
    return {'columns': list(frame.columns),
            'data': [[scalar(value) for value in row] for row in frame.to_numpy(dtype=object)]}


def export_modeling_bundle(bundle_dir, output, pca_dir=None, robustness_dir=None, relaxed_dir=None):
    directory = Path(bundle_dir)
    original = json.loads((directory/'metadata.json').read_text())
    output = Path(output).resolve()
    if original['dataset_id'] != 'synthetic' and output.is_relative_to(ROOT/'web'):
        raise ValueError('Keep empirical exports outside web/. Import the saved JSON locally.')
    sources = {}

    def read(path):
        path = Path(path)
        sources[str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_parquet(path)

    panel = read(directory/'panel.parquet')
    states = read(directory/'states.parquet')[['date_', 'split', 'chi', 'xi', 'two_factor_predictive_ll']]
    reconstruction = read(directory/'reconstructions.parquet').query("model == 'two_factor'").drop(columns='model')
    reconstruction['residual_log'] = reconstruction.observed_log - reconstruction.fitted_log
    parameters = original['parameters']['two_factor']
    loadings = panel[['date_', 'maturity_rank', 'tau']].copy()
    loadings['short_loading'] = np.exp(-parameters['kappa']*loadings.tau)
    loadings['long_loading'] = 1.
    tables = dict(panel=table(panel[['date_', 'futcode', 'lasttrddate', 'maturity_rank', 'tau', 'settlement', 'log_settlement']]),
                  states=table(states), ss_reconstructions=table(reconstruction), ss_loadings=table(loadings))
    metadata = {key: original[key] for key in ('dataset_id', 'dataset_label', 'source', 'price_unit', 'train_end')}
    metadata.update(parameters=parameters, state_estimate='filtered', calibration=original.get('calibration', {}).get('two_factor', {}),
        smoothed_states_available=False, standardized_innovations_available=False,
        state_provenance='Saved filtered states; export does not refilter or recalibrate. Saved calibration may predate the model audit.',
        comparison_note='No matched-target Schwartz–Smith versus PCA comparison has been computed. Structural curves use contract maturities; PCA uses a constant-maturity grid.')
    if pca_dir:
        pca = Path(pca_dir)
        meta = json.loads((pca/'metadata.json').read_text())
        pca_curves = read(pca/'curves.parquet')
        # Reject a mismatched dataset rather than silently combining unrelated files.
        dates = set(pd.to_datetime(panel.date_))
        if not set(pd.to_datetime(pca_curves.index)).issubset(dates):
            raise ValueError('PCA dates do not match the structural bundle')
        metadata['pca'] = meta
        for name in ('explained_variance', 'interpretations', 'scores', 'loadings', 'reconstructions', 'metrics', 'coverage'):
            frame = read(pca/f'{name}.parquet')
            if name in ('interpretations', 'scores', 'loadings', 'coverage'):
                frame = frame.rename_axis({'interpretations':'component', 'scores':'date_', 'loadings':'tau', 'coverage':'date_'}[name]).reset_index()
            tables[f'pca_{name}'] = table(frame)
    if robustness_dir:
        if not pca_dir:
            raise ValueError('A PCA bundle is required with robustness results')
        robust = Path(robustness_dir)
        metadata['robustness'] = json.loads((robust/'metadata.json').read_text())
        for name in ('rolling_spectrum', 'rolling_fits', 'rolling_loadings', 'rolling_metrics',
                     'subperiod_loadings', 'common_window_metrics', 'sensitivity_metrics', 'sensitivity_fits'):
            frame = read(robust/f'{name}.parquet')
            tables[name] = table(frame)
        if relaxed_dir:
            relaxed = Path(relaxed_dir)
            metadata['relaxed_sensitivity'] = json.loads((relaxed/'metadata.json').read_text())
            tables['relaxed_sensitivity_metrics'] = table(read(relaxed/'sensitivity_metrics.parquet'))
    # Forecasts are separate from same-day model reconstructions and are read
    # only from the already-saved Python evaluation bundle.
    if (directory/'predictions.parquet').exists() and (directory/'metrics.parquet').exists():
        predictions = read(directory/'predictions.parquet')
        metrics = read(directory/'metrics.parquet')
        tables['forecast_predictions'] = table(predictions[['date_', 'origin_date', 'horizon', 'split', 'model', 'maturity_rank', 'tau', 'observed_log', 'predicted_log', 'error_log']])
        tables['forecast_metrics'] = table(metrics[['split', 'horizon', 'model', 'maturity_rank', 'rmse_log', 'mae_log', 'rmse_price', 'mae_price', 'n', 'n_dates']])
        metadata['forecast'] = {
            'horizons': sorted(int(value) for value in predictions.horizon.unique()),
            'training_end': original.get('train_end'),
            'validation_start': original.get('validation_start'),
            'test_start': original.get('test_start'),
            'models': sorted(str(value) for value in predictions.model.unique()),
            'provenance': 'Saved forecasts and evaluation metrics; exporter does not fit models.',
        }
    metadata['source_sha256'] = sources
    payload = dict(format='crude-oil-factor-modeling', version=1, metadata=metadata, tables=tables)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False)+'\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT/'data/synthetic/demo')
    parser.add_argument('--output', type=Path, default=ROOT/'web/data/modeling-demo.json')
    parser.add_argument('--pca', type=Path)
    parser.add_argument('--robustness', type=Path)
    parser.add_argument('--relaxed-sensitivity', type=Path)
    args = parser.parse_args()
    print(export_modeling_bundle(args.bundle, args.output, args.pca, args.robustness, args.relaxed_sensitivity))


if __name__ == '__main__':
    main()
