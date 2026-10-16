# Schwartz–Smith implementation audit

Reference: Schwartz and Smith (2000), *Short-Term Variations and Long-Term
Dynamics in Commodity Prices*, [original paper](https://www.anderson.ucla.edu/documents/areas/fac/finance/72.pdf).

## Verified mathematical specification

State order is `(chi, xi)` and log spot is `chi + xi`. Under the physical measure,
`dchi = -kappa chi dt + sigma_chi dW_chi`,
`dxi = mu_xi dt + sigma_xi dW_xi`, with instantaneous correlation `rho`.
Under the pricing measure, drift adjustments are `-lambda_chi` and `-lambda_xi`.
These lambdas are in drift units, not dimensionless Sharpe ratios.

Writing `b(k,t) = (1-exp(-k*t))/k`, the exact discrete system is:

- `F = diag(exp(-kappa*dt), 1)`; `c = (0, mu_xi*dt)`.
- `Q11 = sigma_chi^2 b(2*kappa,dt)`; `Q22 = sigma_xi^2 dt`.
- `Q12 = Q21 = rho sigma_chi sigma_xi b(kappa,dt)`.
- `log futures = exp(-kappa*tau) chi + xi + A(tau)`.
- `A = -lambda_chi b(kappa,tau) + (mu_xi-lambda_xi) tau`
  `+ [sigma_chi^2 b(2*kappa,tau) + sigma_xi^2 tau`
  `+ 2 rho sigma_chi sigma_xi b(kappa,tau)]/2`.

The existing formulas were algebraically correct. The risk-neutral variance
correction belongs in futures pricing; physical state propagation uses `mu_xi`.
`delta` in real-data estimation is `mu_xi-lambda_xi`. The existing forecast helper
returns conditional expected **log** futures. Exponentiating that is not the
conditional arithmetic mean futures price; no forecast semantics were changed.

## Corrections and diagnostics

- `src/commodity_factors/schwartz_smith.py`: replaced subtractive exponential
  expressions with `expm1`; verified against numerical integration down to
  kappa=1e-16. Low-level covariance/pricing helpers include the kappa=0 limit;
  the filter requires positive kappa for the stationary short-factor prior.
- `src/commodity_factors/kalman.py`: removed the default prior's dependence on
  the first observed curve, which was then reused in that observation's
  likelihood. Default prior is now mean `(0,0)` and covariance
  `diag(sigma_chi^2/(2*kappa), 1e4)`, before any observation. Users can supply
  `x0/P0`, including a known deterministic initial state.
- Added validation of model parameters, state/covariance dimensions, symmetry,
  positive semidefiniteness and single-row time inputs. Predictions are
  symmetrized; updates retain Joseph form and Cholesky solves. Nonfinite results
  and failed innovation factorizations report the failing row. No jitter silently
  changes the likelihood.
- NaN observations are skipped using the corresponding observation submatrix.
  Entirely missing dates propagate states/covariances and contribute zero
  likelihood. Unknown maturities are allowed only for missing observations;
  predictions at those entries remain NaN. Zero maturity is valid in the filter.
- Filter output adds marginal standardized innovations, Cholesky-whitened
  innovations, innovation variances, daily normalized innovation squared (NIS),
  observed counts, and innovation covariance condition numbers. Missing entries
  are NaN. Whitened components depend on within-row contract order; NIS and
  likelihood do not. NIS is nominally chi-square with the observed count degrees
  of freedom for known parameters. Marginal standardized errors remain
  cross-correlated. Estimates fitted on the same data affect diagnostic inference.
- `src/commodity_factors/estimation.py`: both estimators accept explicit priors,
  validate optimizer starts, reject observation-free fitting, suppress optional
  filter diagnostics during optimization and report active bounds, fixed
  parameters, iterations, evaluations and projected gradient infinity norm.
  Four-parameter fitting now accepts `maxiter`. Existing L-BFGS-B constraints and
  real-data parameterization are retained. Finite objective is required for success.
- `src/commodity_factors/analysis.py`: persists the extra optimizer diagnostics
  in calibration metadata. Existing forecasting interfaces remain intact.

## Data, maturities and rolls

Reviewed `data.py`, `simulation.py`, `evaluation.py`, `analysis.py`, and `daily.py`.
Historical maturity and elapsed time both use calendar days / 365.25. Synthetic
helpers default to 1/252; callers must keep annual parameter units consistent.
Daily inputs are normalized dates; intraday expiry precision is not represented.
Last trading date is the modeled maturity proxy. Expired contracts are excluded;
nearby ranks can change contract identity. Observation loadings and adjustments
use each date's actual maturity; latent states continue across rolls.

The existing pipeline intentionally requires a complete panel and rejects
incomplete curves. This policy is unchanged. Missing support is available to
callers of the filter/estimators directly; forecasting evaluation still expects
complete input. Rank-based forecasts are not a roll-adjusted trading P&L series.

## Tests and identification limits

`tests/test_schwartz_smith_audit.py` adds independent integrated-covariance checks,
the transition covariance semigroup, physical/risk-neutral separation, a dense
joint Gaussian likelihood comparison with missing data and a roll, contract-order
invariance, prior independence, missing-date propagation, invalid inputs,
near-collinear numerical stability, calendar/leap-day maturities, optimizer
failure reporting, and seeded synthetic recovery. The recovery experiment uses
1,000 observations over approximately 25 years and six maturities spanning
0.05–4 years. It recovers kappa and volatilities within 20% and rho within 0.15,
with fixed drift/risk-premium/noise parameters and known initial state. It also
checks whitened innovation mean, variance and lag-one correlation.

The default prior is a proper finite broad prior, **not** an exact diffuse
initialization. Initial likelihood and estimates can depend on its scale;
likelihoods from different prior conventions should not be compared directly.
Real data were not recalibrated and persisted model bundles were not regenerated.
Changing initialization can change future calibrations.

The real-data estimator has eight parameters (seven with lambda_chi fixed).
Finite samples with narrow maturity coverage can weakly identify kappa and the
short factor; drift and risk premia are especially weak over short time spans.
Curve pricing identifies delta, while physical time-series drift informs mu_xi;
lambda_xi is their difference. A fixed lambda_chi is a calibration restriction,
not a recovered empirical fact. The synthetic test does not claim simultaneous
recovery of all eight parameters. L-BFGS-B success is local convergence, not a
proof of identification or global optimality. Dashboard calibration retains its
two existing starts. Inspect active bounds and projected gradients, and use
profile likelihood or independent starts before interpreting weak parameters.
Measurement noise remains independent, homoskedastic Gaussian noise in log prices.

Validation: the full Python suite passed **63 tests in 12.81 seconds**, including
22 new audit cases, using the existing `commodity-factors` conda environment:

```sh
/opt/homebrew/Caskroom/miniconda/base/envs/commodity-factors/bin/python -m pytest -q
```

`git diff --check` also passed. Existing forecasting, daily evaluation, data,
application, export, and leakage regression tests were included. Browser tests
and live external data services were not exercised; no browser code was changed.
