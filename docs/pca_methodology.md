# Constant-maturity PCA

PCA is implemented in `src/commodity_factors/pca_baseline.py`. It extracts factors
and reconstructs curves; it does not forecast. Schwartz–Smith, its calibration,
and the existing forecasting/data pipeline are unchanged. Results live separately
under `results/pca/`; the saver refuses to overwrite a non-PCA result bundle.

## Method and preprocessing

The input is the existing contract-level raw or processed Parquet format:
`date_`, `futcode`, `lasttrddate`, `settlement`. Existing cleaning and maturity
utilities are reused, but the complete nearby-rank requirement is not imposed.
Maturities are recalculated from dates rather than trusting potentially stale
`tau` or `log_settlement` columns. Nonpositive/nonfinite prices cannot be logged
and are removed by the existing cleaner. Their original observation dates are
retained so an invalid whole curve cannot silently bridge a daily difference.

Each day is interpolated linearly in **log futures price** onto a fixed,
user-specified maturity grid (default: 1, 2, 3, 4, 5 months; month = 1/12 year).
Underlying maturity fractions use calendar days / 365.25, as in the existing
pipeline. Grid choice is an ex-ante modeling choice, not optimized on holdout
coverage. Duplicate maturities are rejected as ambiguous. Interpolation uses
only same-day observations, no extrapolation, no forward/back filling, and no
interpolation across brackets wider than 62 calendar days by default.
An exact quote is retained even if adjacent contracts are absent.

Thus a switch of nearby contract identity is not directly differenced as a price
shock. A synthetic unchanged linear curve remains exactly unchanged through a
roll. For nonlinear actual curves, changing interpolation brackets can still
produce interpolation error: this is not a claim to remove all roll effects.
Sparse grids, seasonal products and steep front-end curves need sensitivity
analysis. No calendar spread adjustment or synthetic roll P&L is introduced.

Two transformations are supported:

- **`log_changes` (default):** first differences of the aligned log curves.
  This describes common relative curve shocks without allowing drifting price
  levels to dominate the covariance. Each interval ends at its score date.
  Differences across an incomplete endpoint or gaps over seven calendar days
  are missing. Weekend/holiday intervals up to seven days remain included;
  changes are not divided by elapsed time or annualized. Elapsed days are saved.
- **`log_prices`:** aligned log levels, useful for contemporaneous curve
  compression and comparison with latent price-level factors. Persistent levels
  can dominate explained variance; components are sample-dependent statistical
  directions, not structural states. Raw price changes are omitted because their
  scale varies with the price level; fixed-maturity log changes already represent
  relative curve changes.

Only complete transformed rows enter PCA or receive scores. Partial aligned
curves and all exclusion dates remain in the saved tables with NaNs and coverage
flags. A completely absent date cannot be inferred from quotes alone; the elapsed
interval is retained and the maximum-gap rule applies. Coverage selection may
bias results toward liquid curves. There is no statistical missing-value imputer.

`CurvePCA.fit` centers each maturity using **training data only**. Optional
`standardize=True` also divides by training sample standard deviation, with unit
scale for numerically constant columns. Default covariance PCA keeps natural
relative volatility differences across maturities: all features are log-price
units. Correlation PCA is a sensitivity option, not the default.

The existing chronological split helper is reused: dates before 2024 are training,
2024 is validation and 2025 onward is test by default. A change spanning a split
boundary is assigned to its ending date. Model means, scales, eigenvectors and
variance estimates are frozen after training. Training scores are retrospective
calibration diagnostics; the fitted model is only available after `train_end`.

The existing scikit-learn dependency supplies full-SVD PCA (no whitening or
covariance-matrix eigendecomposition). See the [PCA API documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).
Signs are fixed by making each component's largest-magnitude coefficient positive;
this only makes display reproducible. Zero-variation training data are rejected.
Rank-deficient data remain usable and the fitted matrix rank is reported;
zero-variance component directions have no identified interpretation.

## Scores, interpretation and reconstruction

For transformed data `X`, training mean `m`, training scale `s`, PCA centering
correction `a`, and component rows `V`:

```
Z = ((X - m) / s - a) @ V.T
X_reconstructed = (Z @ V + a) * s + m
```

`loadings` contains original-feature-unit directions `V.T * s`. For centered PCA
these are unit eigenvectors; for standardized PCA they describe one score unit's
movement in the original feature units. The saved numeric model includes both
`V` and `s`. Explained variance ratios are relative to total training variance
in the chosen scaled/unscaled space, not renormalized to the retained components.

Economic descriptions are evaluated using signed cosine alignment with
orthogonalized constant, linear and quadratic maturity shapes. These are level,
slope and curvature *reference shapes*, not mandatory identities. Magnitudes
measure alignment; signs are arbitrary. Mixed shapes, sign changes, repeated
or nearly tied eigenvalues and component rotations require caution. The test suite
explicitly recovers a structure whose first component is curvature, not level.

Level reconstruction applies the inverse transform directly. Change reconstruction
adds the reconstructed current change to the **previous observed aligned log
curve**, not a recursively reconstructed path. It uses today's scores and is
same-day compression, not out-of-sample prediction. For changes, feature residual
and reconstructed log-curve residual are identical. Metrics include RMSE/MAE by
split and maturity, with `tau=0` denoting pooled errors. They compare against
interpolated grid observations, not original individual contract settlements;
interpolation error is not included in these PCA residuals.

## Running and outputs

Activate the existing `commodity-factors` conda environment, then run:

```sh
python scripts/run_pca.py
python scripts/run_pca.py --transformation log_prices --output results/pca/wti_levels
# Optional training-only correlation PCA sensitivity:
python scripts/run_pca.py --standardize --output results/pca/wti_standardized
```

The CLI also accepts `--panel`, `--maturities-months`, `--components`,
`--validation-start`, `--test-start`, `--max-interpolation-gap-days` and
`--max-change-days`. Results contain Parquet tables for aligned curves, features,
scores, physical loadings, variance, shape alignments, reconstructions, metrics
and coverage; JSON metadata; and an unpickled numeric `model.npz`.
Saved parameters can reproduce projection using the formula above.

`plot_pca_analysis(result, date=None)` returns a Matplotlib figure containing
explained variance, loadings, factor scores and observed/reconstructed curves.
The CLI saves it to `diagnostics.png`. Scores are divided by their **training**
standard deviation only for display so all three components are visible; saved
scores are unscaled. Plot split boundaries are marked. A chosen eligible date
can be passed to inspect a particular reconstructed curve.

## Local historical WTI results

Using the existing `data/processed/wti_panel_2021_2025.parquet`, the fixed 1–5
month grid and three components, calibrated on pre-2024 observations:

| Input | PC1 variance | PC2 variance | PC3 variance | Retained variance |
|---|---:|---:|---:|---:|
| Log changes | 99.5409% | 0.4473% | 0.0104% | 99.9986% |
| Log levels | 99.7505% | 0.2469% | 0.0023% | 99.9998% |

Absolute cosine alignments of the change components with their best-matching
reference shapes are 0.9992 (PC1/level), 0.9967 (PC2/slope), and 0.9684
(PC3/curvature). For levels these are 0.9990, 0.9969 and 0.9834. These samples
support the familiar interpretation empirically, with no imposed naming rule.
PC3's tiny variance share is not evidence of economic importance or robustness.

| Input | Training RMSE (log) | Validation RMSE (log) | Test RMSE (log) | Test scored / supplied dates |
|---|---:|---:|---:|---:|
| Log changes | 0.000088 | 0.000054 | 0.000062 | 233 / 251 |
| Log levels | 0.000234 | 0.000171 | 0.000272 | 240 / 251 |

Changes retain 700/753 training and 236/252 validation dates; levels retain
720/753 and 242/252. These transformations answer different questions and use
different eligible rows, so their RMSEs are not a model ranking. The high explained
variance and small errors partly reflect short, smooth, interpolated curves.
The saved per-maturity tables provide the full reconstruction error breakdown.

## Validation and limits

Tests cover known-factor/subspace recovery, deliberately nonstandard economic
ordering, full-rank reconstruction, discarded-variance error identities,
ill-conditioned/rank-deficient data, constant-data rejection, training-only
centering/scaling/loadings, holdout perturbations, maturity order, unchanged-curve
rolls, missing contracts/dates, no extrapolation/time filling, gap exclusions,
metric calculations, saved-model score reproduction, plotting, and protection of
non-PCA output bundles.

This is static linear PCA. No new forecasting model, wavelet model, automatic
component selection, seasonality adjustment, probabilistic imputation or claim
of structural identification is included. Factor stability across rolling windows
and interpolation/grid sensitivity remain to be studied. The local historical
input was used as supplied; upstream quote provenance was not independently
revalidated. Existing forecasting regressions remain part of the full test suite.

Validation completed: **83 Python tests passed in 14.05 seconds**, including
20 PCA tests. Both historical CLI runs completed, and their diagnostic figures
were rendered and visually inspected. `git diff --check` passed. No frontend
changes were made; browser tests and live data retrieval were not run.

Files in this change: implemented `pca_baseline.py` and `plotting.py`; added
`scripts/run_pca.py`, `tests/test_pca.py`, and this guide; updated the PCA status
and entry point in `README.md`. Earlier Schwartz–Smith audit changes and unrelated
in-progress documentation edits were preserved. This implemented fixed-maturity
method supersedes the earlier maturity-rank PCA proposal in `PROJECT_PLAN.md`.
