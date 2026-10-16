# WTI PCA factor stability and robustness

## Conclusion from the local 2021–2025 sample

The dominant loading direction is persistent across the tested windows, but the
full factor structure is not invariant. Secondary variance shares change sharply
in 2022, shorter-window third-component directions drift, and reconstruction is
sensitive to missing quotes and the maturity/interpolation specification. These
are descriptive sample findings, not a statistical proof of factor stability.

All figures and tables below use the existing local historical WTI panel,
constant maturities of 1–5 months, and log-price changes. Covariance and
training-standardized correlation PCA are both evaluated. No Schwartz–Smith,
Kalman, forecasting or upstream data-pipeline code was modified in this extension.

## Estimation and alignment

`rolling_pca_analysis` fits on the preceding 126, 252 or 504 **supplied trading-date
rows**, excluding the evaluation date. Windows do not expand backward to replace
missing observations. Complete transformed rows must cover at least 80% of the
window and contain at least five observations with centered rank at least three.
Skipped fits are recorded with a reason. Means and optional maturity scales are
recomputed exclusively from the prior window. Defaults refit daily; `step` samples
a common row-index schedule across window lengths, so window comparisons can use
identical held-out dates. A change is assigned to its ending date; its starting
price may precede the first window date.

Loading directions are mapped back to original log units and normalized to unit
length before comparisons. Maximum-absolute-cosine assignment matches the three
current directions to the preceding aligned directions, then sign correction
makes the match positive. The first valid window supplies the long-run reference.
Saved tables distinguish tracked direction labels from current variance ranks.
Explained variance and K-component reconstruction always use the original
variance ordering, never the alignment ordering. No actual component swaps
occurred in the baseline local runs; a synthetic test verifies swap/sign handling.

Cosines to both the preceding and initial window, relative adjacent eigengaps,
and the largest principal angle between the initial/current top-three subspaces
are recorded. This matters because sign/permutation alignment cannot resolve a
rotation inside a nearly degenerate eigenspace. No Procrustes rotation is imposed
on individual PCs. Low-rank fits are excluded rather than assigning economic
meaning to zero-variance directions. With only three maturity knots, a full-rank
three-component subspace fills the entire feature space, so its angle is
uninformative; use more than three knots for that diagnostic.

Annual fits compare 2021, 2022, 2023, 2024 and 2025 retrospectively, aligned to the
earliest year separately for each PCA scaling. These annual fits are descriptive
and are not used for the held-out error calculations.

## Rolling and subperiod evidence

Covariance PCA, comparing each window with its own first valid reference:

| Window | PC1 variance range | Minimum PC1 cosine | Minimum PC3 cosine | Maximum three-factor subspace angle |
|---|---:|---:|---:|---:|
| 126 dates | 99.0015–99.9250% | 0.999157 | 0.944071 | 18.72° |
| 252 dates | 99.3064–99.7898% | 0.999495 | 0.984411 | 9.32° |
| 504 dates | 99.4568–99.7453% | 0.999901 | 0.992570 | 2.73° |

These maxima cover different available periods and use different anchors. The
504-date evaluation starts in 2023, so its smaller angle cannot establish greater
stability during 2022. Overlapping windows mechanically smooth the paths and
make consecutive-window similarities high. Use initial-reference angles as well.

In descriptive annual covariance fits, PC2 explains **0.2015%** in 2021 versus
**0.6725%** in 2022, while PC1 falls from **99.7898% to 99.3123%**. The cross-maturity
RMS standard deviation of log changes rises from **0.02095 to 0.02880**, about 38%.
The 2022 three-factor subspace is about **6.04°** from 2021. Thus the broad loading
shapes remain similar while the importance and amplitude of curve-shape shocks
change. PC2 also reaches 0.6228% in 2025; variation is not confined to 2022.

Covariance and correlation results are close in this short-end WTI sample. The
126-date correlation subspace maximum is 18.69°, versus covariance's 18.72°.
Correlation PCA does not remove the temporal variation in secondary factors.
This agreement need not hold when maturity-specific volatilities differ more.

## Prior-only held-out reconstruction

For every eligible evaluation date, one prior-window PCA provides K=1,2,3
reconstructions by zeroing unused scores. Each score uses the held-out current
curve, so this evaluates out-of-estimation-sample **compression, not forecasting**.
For log changes, the reconstructed increment is added to the preceding observed
aligned curve. All RMSEs are in original log-price units, including correlation
PCA. More components minimize the scaled loss for correlation PCA, not necessarily
every individual curve's unscaled squared error.

252-date covariance PCA, pooled across the five maturities:

| Evaluation period | K=1 RMSE | K=2 RMSE | K=3 RMSE | Eligible dates |
|---|---:|---:|---:|---:|
| 2022 | 0.002431 | 0.000372 | 0.000132 | 235 |
| 2023 | 0.000983 | 0.000164 | 0.000059 | 234 |
| 2024 | 0.000875 | 0.000156 | 0.000054 | 236 |
| 2025 | 0.001423 | 0.000206 | 0.000062 | 233 |
| All | 0.001555 | 0.000241 | 0.000083 | 938 |

Correlation PCA's pooled K=1,2,3 errors are 0.001560, 0.000241 and 0.000083.
The higher absolute errors in 2022 partly reflect greater shock magnitude;
RMSE increases alone do not establish a structural break.

252-date covariance PCA, K=3 by maturity:

| Maturity | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|
| 1 month | 0.000060 | 0.000029 | 0.000022 | 0.000028 |
| 2 months | 0.000164 | 0.000081 | 0.000071 | 0.000081 |
| 3 months | 0.000100 | 0.000050 | 0.000050 | 0.000056 |
| 4 months | 0.000176 | 0.000070 | 0.000066 | 0.000078 |
| 5 months | 0.000122 | 0.000049 | 0.000048 | 0.000051 |

For a fair window comparison, all windows are also scored on their common dates.
On those 702 dates (after the 504-date warm-up), covariance K=3 RMSE rounds to
0.000058 for all three windows. K=2 RMSE is 0.000172, 0.000177 and 0.000181 for
126, 252 and 504 dates respectively. This does not evaluate the 504-date model in
2022, or establish an optimal window length.

## Interpolation, grid and missing-data sensitivity

The six scenarios are baseline linear interpolation; a 35-day rather than 62-day
maximum bracket; same-day shape-preserving PCHIP interpolation; a shorter 1–4
month grid; a denser half-month grid from 1–5 months; and seeded independent loss
of 10% of contract quotes. PCHIP keeps the same no-extrapolation and maximum-bracket
rules, with derivatives estimated from that day's available contract curve.
All scenarios reuse the same maturity alignment and change-gap policy.

Each scenario retains its own errors and loading/variance paths. Cross-scenario
errors use **identical baseline-linear observed targets**, common maturities and
common held-out dates. Consequently PCHIP errors include disagreement with the
linear reference; they do not establish that linear interpolation is closer to
true constant-maturity prices. Shared-support comparison changes the target to
months 1–4 because the shorter-grid model has no fifth-month estimate.

At the 80% training coverage requirement, random loss of 10% of quotes leaves
**zero eligible windows**. The fraction of usable changes in 252-date windows is
only 43.7–58.7%: a missing endpoint can invalidate a curve and both adjacent
changes. This failed scenario is explicitly listed in metadata, not assigned
an invented error. Common-support metrics use the five scenarios with scored
dates; coverage and excluded scenario names are reported separately.

On 938 common dates and months 1–4, covariance K=3 RMSE is:

| Scenario | RMSE versus baseline-linear observations |
|---|---:|
| Baseline | 0.000085 |
| 35-day brackets | 0.000085 |
| PCHIP | 0.000203 |
| Shorter grid | 0.000049 |
| Denser grid | 0.000087 |

The bracket limit makes no difference because this panel has complete monthly
contracts. That is not evidence of insensitivity to gaps in sparse curves.
Shortening the grid makes reconstruction easier and changes the question; it
is not an overall improvement on the full curve. Its weakest matched direction
has cosine **0.724** to the baseline restricted to common knots, despite a maximum
subspace angle of only **4.69°**. Individual weak directions can rotate inside a
similar span. For PCHIP the maximum matched-subspace angle is **2.38°**.

A separate diagnostic lowers minimum coverage to 50%, holding all other settings
fixed. The quote-loss scenario then has 580 fitted windows and 306 eligible
held-out dates (424 windows still fail coverage). On those common dates and
months 1–4, baseline K=3 RMSE is **0.000094**, versus **0.000232** with quote loss.
The quote-loss subspace can differ from baseline by **13.14°**. This shows material
missing-data sensitivity under one random-loss realization. It is not a guarantee
for market-driven or clustered missingness, and it does not justify silently
relaxing the production coverage rule.

## Reproduce and inspect

In the existing `commodity-factors` environment:

```sh
python scripts/run_pca.py --robustness
python scripts/run_pca.py --robustness --windows 252 --min-coverage 0.5 \
  --output results/pca/wti_coverage50
```

New options: `--windows`, `--step`, `--min-coverage`, `--sensitivity-window`,
`--missing-fraction`, `--seed`, and `--skip-sensitivity`. The robustness mode always
compares both covariance and correlation PCA and K=1,2,3; static `--components`
and `--standardize` choices continue to apply to static PCA only. The existing
`--transformation`, maturity grid and gap controls apply to both modes.

The default extension writes to `results/pca/wti/robustness/`, leaving existing
static PCA files in place. The saver rejects a directory belonging to another
analysis type. Parquet outputs include rolling fits/coverage, aligned loadings,
variance/eigengaps, individual reconstruction errors, annual/maturity metrics,
common-window metrics, subperiod loadings, and sensitivity paths and comparisons.
JSON records the settings, scenario coverage, and unscored scenarios.

Five inspected PNG reports are saved in each robustness directory:

- `rolling_variance.png`: ranked variance shares for all windows/methods.
- `loading_stability.png`: maturity-loading heatmaps and loading-direction angles
  to the first/previous fit for the selected window (252 dates by default).
- `reconstruction_by_k.png`: held-out errors by K and calendar year.
- `subperiod_loadings.png`: descriptive annual factor shapes for both scalings.
- `reconstruction_by_maturity.png`: held-out errors across maturities and years.

## Tests, changes and remaining limits

Extended `pca_baseline.py`, `plotting.py` and `scripts/run_pca.py`; added
`tests/test_pca_robustness.py` and this report; added a README entry point. Static
PCA behavior is preserved, with shared feature construction and optional PCHIP
added to the maturity alignment helper. Earlier uncommitted work is preserved.

The full Python suite passed **99 tests** (16 new robustness cases). The repository's
JavaScript test command also passed **15 tests**, using bundled Node directly
because npm was absent from the shell. Tests verify sign/swap correction,
near-degenerate subspace rotations, detection of a changed factor span, prior-only
fitting and frozen held-out transforms, observed-date window boundaries, shared
refit scheduling, missing coverage, PCHIP locality, reproducible quote removal,
common-support/reference-target metrics, serialization, plotting and output
protection. `git diff --check` passed. Five figures were visually inspected.

No bootstrap confidence intervals, formal structural-break tests, automatic
window/K selection, seasonality adjustment, or new forecasting models were added.
Overlapping windows and interpolation-induced smoothness inflate apparent
continuity. The five-year sample omits the 2020 negative-price episode and covers
only a short maturity span. Constant-maturity values are interpolated, and random
quote loss is only one stress design. Small eigenvalues make weaker components
more sensitive to sampling and preprocessing. The evidence supports persistence
of a dominant common direction, not unconditional stability of three economic
factors across markets or regimes.
