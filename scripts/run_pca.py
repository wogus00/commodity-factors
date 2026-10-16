#!/usr/bin/env python3
"""Fit and evaluate PCA on local historical curves; save separate diagnostics."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from commodity_factors.data import load_raw_data
from commodity_factors.pca_baseline import run_pca_analysis, save_pca_analysis, run_pca_robustness, save_pca_robustness
from commodity_factors.plotting import plot_pca_analysis, plot_pca_robustness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, default=Path('data/processed/wti_panel_2021_2025.parquet'))
    parser.add_argument('--output', type=Path, default=Path('results/pca/wti'))
    parser.add_argument('--maturities-months', nargs='+', type=float, default=[1, 2, 3, 4, 5])
    parser.add_argument('--transformation', choices=['log_changes', 'log_prices'], default='log_changes')
    parser.add_argument('--components', type=int, default=3)
    parser.add_argument('--standardize', action='store_true')
    parser.add_argument('--validation-start', default='2024-01-01')
    parser.add_argument('--test-start', default='2025-01-01')
    parser.add_argument('--max-interpolation-gap-days', type=float, default=62)
    parser.add_argument('--max-change-days', type=float, default=7)
    parser.add_argument('--robustness', action='store_true', help='Write a separate rolling/robustness report')
    parser.add_argument('--windows', nargs='+', type=int, default=[126, 252, 504])
    parser.add_argument('--step', type=int, default=1, help='Refit every N supplied trading dates')
    parser.add_argument('--min-coverage', type=float, default=.8)
    parser.add_argument('--sensitivity-window', type=int, default=252)
    parser.add_argument('--missing-fraction', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--skip-sensitivity', action='store_true')
    args = parser.parse_args()
    if args.robustness:
        result = run_pca_robustness(load_raw_data(args.panel),
            maturities=np.array(args.maturities_months)/12, windows=args.windows, step=args.step,
            transformation=args.transformation, max_gap_years=args.max_interpolation_gap_days/365.25,
            max_change_days=args.max_change_days, min_coverage=args.min_coverage,
            sensitivity=not args.skip_sensitivity, sensitivity_window=args.sensitivity_window,
            missing_fraction=args.missing_fraction, seed=args.seed)
        result['metadata']['source_path'] = str(args.panel)
        destination = save_pca_robustness(result, args.output / 'robustness')
        for name, fig in plot_pca_robustness(result, window=args.sensitivity_window).items():
            fig.savefig(destination / f'{name}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        print(result['rolling_metrics'].query('tau == 0').to_string(index=False))
        if 'sensitivity_metrics' in result:
            print('\nSensitivity on common dates/maturities against baseline observations:')
            print(result['sensitivity_metrics'].query('tau == 0 and period == "all"').to_string(index=False))
            print('Unscored scenarios:', result['metadata']['unscored_scenarios'])
            print('Scored dates by scenario:', {name: details['scored_dates']
                  for name, details in result['metadata']['scenarios'].items()})
        print(f'Saved robustness analysis to {destination.resolve()}')
        return
    result = run_pca_analysis(load_raw_data(args.panel), maturities=np.array(args.maturities_months)/12,
        transformation=args.transformation, n_components=args.components, standardize=args.standardize,
        validation_start=args.validation_start, test_start=args.test_start,
        max_gap_years=args.max_interpolation_gap_days/365.25, max_change_days=args.max_change_days)
    result.metadata['source_path'] = str(args.panel)
    destination = save_pca_analysis(result, args.output)
    fig = plot_pca_analysis(result)
    fig.savefig(destination / 'diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(result.explained_variance.to_string(index=False))
    print('\nSigned shape cosines (not automatic economic labels):')
    print(result.interpretations.to_string())
    print('\nPooled reconstruction errors:')
    print(result.metrics.query('tau == 0').to_string(index=False))
    print('\nCoverage:', result.metadata['coverage_by_split'])
    print(f'Saved PCA analysis to {destination.resolve()}')


if __name__ == '__main__':
    main()
