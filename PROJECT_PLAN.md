# Latent Factor Modeling of Crude Oil Futures

**A study of structural and statistical factors in the futures term structure**

**Market:** NYMEX WTI light sweet crude oil (CL) futures

**Data:** WRDS / LSEG Datastream Futures, daily contract-level settlements, 2021–2025

**Foundation:** the existing data pipeline, state-space models, saved analyses, and dashboards

**Primary reference:** Schwartz, E. & Smith, J. E. (2000), *Short-Term Variations and Long-Term Dynamics in Commodity Prices*, *Management Science* 46(7), 893–911. DOI: https://doi.org/10.1287/mnsc.46.7.893.12034

> **Instructions for Codex:** Build directly on this repository. Preserve the data pipeline, architecture, working model implementations, and forecasting functionality. Prioritize mathematical validation, reproducibility, and honest factor interpretation. This reframing step changes documentation only: do not implement PCA, alter the Schwartz–Smith equations, or implement wavelets. The roadmap below describes later work, not capabilities or results already delivered. Do not invent empirical results, silently make financial-data assumptions, or commit licensed prices or credentials.

## 1. Research question and scope

**Question:** What latent factor structure describes the WTI futures curve, how well does the Schwartz–Smith model reconstruct observed curves, and how does its representation compare with a statistical PCA benchmark?

Research objectives:

1. Estimate and interpret the Schwartz–Smith latent factors driving the WTI futures term structure.
2. Evaluate structural curve explanation and same-day reconstruction across maturities and time.
3. Introduce PCA to study factor dimensionality and maturity-dependent loadings.
4. Compare structural and statistical factors using reconstruction errors, residual diagnostics, and parameter and loading stability.

The Schwartz–Smith framework decomposes log spot prices into a mean-reverting short-term deviation and a persistent long-term equilibrium component. A short-dated futures contract loads relatively heavily on the short-term factor, while a long-dated contract is more sensitive to the equilibrium level.

Current and planned approaches:

| ID | Model | Main role |
|---|---|---|
| **M1** | Existing Brownian one-factor log-price model | Implemented parsimonious structural baseline |
| **M2** | Schwartz–Smith state-space model with Kalman filtering and maximum likelihood | Implemented core model; independent validation remains a priority |
| **M3** | PCA with one and two components, plus a full scree analysis | Planned statistical benchmark; module is currently a placeholder |
| **RW** | Existing rank-wise random walk; same-contract variant in the daily view | Implemented secondary forecast benchmark |

Separate the following evaluations:

1. **Primary modeling evaluation:** factor interpretation, dimensionality, same-day reconstruction, residual structure, and stability.
2. **Secondary forecasting evaluation:** future-curve predictions using information available only through the forecast origin.

**Do not conflate reconstruction and forecasting.** A PCA reconstruction of the test day's observed curve is *not* a forecast. In-sample fitted Kalman-filter curves are also not the same as one-step-ahead predictions.

### Initial boundaries

- Use WTI CL only. Other commodities, options pricing, and trading strategies are outside the current research scope. Wavelets are deferred and are not part of this roadmap.
- Use six nearest unexpired delivery-month contracts per day for the initial panel, subject to data checks. Expand to longer maturities if useful and well covered.
- Work with **log settlement prices** and positive-price observations; 2021–2025 avoids the major negative April 2020 front-month episode, but positivity still must be verified.
- Reuse the existing extraction, preparation, calibration, bundle export, and presentation paths. Add modeling reports incrementally after validation; preserve forecast outputs and commands.

### Audited implementation status

`analysis.run_analysis` fits on the calibration period, then builds filtered states, structural reconstructions, and aligned forecasts. `metrics.parquet` currently contains forecast metrics. Expanded reconstruction summaries, residual diagnostics, uncertainty reports, parameter stability studies, and PCA have not been implemented. Existing tests and a saved synthetic demo provide selected checks, not proof of economic identification or parameter recovery.

The notebooks retain exploratory history. In `02_real_data.ipynb`, an early cell assigns the scalar log likelihood to `fitted_log_prices`, which invalidates the following reconstruction error calculation and curve plot until a later cell replaces it with actual fitted curves. Its calibration windows also differ from the saved-analysis workflow. Use the scripts and saved tables as the reproducible foundation; do not treat notebook output as validated modeling evidence.

## 2. Existing data pipeline and maturity conventions

The existing WRDS extraction uses schema `tr_ds_fut`. Earlier source discovery identified:

- `tr_ds_fut.wrds_contract_info`: per-contract identifiers, delivery label and dates; columns include `contrcode`, `futcode`, `dsmnem`, `contrdate`, `lasttrddate`, `sttlmntdate`, `expirationdate`, `exchtickersymb`.
- `tr_ds_fut.dsfutcontrval`: daily individual-contract prices, keyed by `futcode` and `date_`; columns include `settlement`, `volume`, `openinterest`.
- `tr_ds_fut.wrds_fut_contract`: alternative expanded price table, to be used only if needed.

**Confirmed CL contract family:** `contrcode = 1986`, `contrname = 'CRUDE OIL (LIGHT SWEET)'`, `exchtickersymb = 'CL'`. Do not mix in WTI swaps, Brent, micro CL, or regional differentials.

**Coverage query already executed:** 2021-01-04 through 2025-12-31; **161,354 non-null daily price records**, **193 distinct `futcode` contracts**, **1,256 trading dates** before filtering. These are *coverage counts*, **not** final clean-panel counts or model results.

`lasttrddate` and `sttlmntdate` were populated and equal in the previously inspected far-dated contract rows; `expirationdate` was null. Preserve `lasttrddate` as the current cutoff proxy. Futures pricing may refer to delivery/contract settlement rather than the final trading date; investigate this sensitivity before assigning an economic interpretation to maturity effects.

### Current panel schema

| column | definition |
|---|---|
| `date_` | market observation/trading date |
| `futcode` | unique Datastream futures contract ID |
| `dsmnem`, `contrdate` | Datastream mnemonic / delivery month |
| `lasttrddate` | final trading date from metadata |
| `settlement` | observed daily settlement price, USD per barrel (confirm vendor convention) |
| `volume`, `openinterest` | optional liquidity diagnostics |
| `tau` | `(lasttrddate - date_).days / 365.25` |
| `maturity_rank` | 1 = nearest unexpired listed contract on that date |
| `log_settlement` | natural logarithm of positive `settlement` |

### WRDS extraction and preparation

Use **parameterized SQL**, since `%` wildcard characters in literal SQL strings previously caused a SQLAlchemy/WRDS driver error in this environment.

```python
sql = """
SELECT
    p.date_,
    p.futcode,
    c.dsmnem,
    c.contrdate,
    c.lasttrddate,
    p.settlement,
    p.volume,
    p.openinterest
FROM tr_ds_fut.dsfutcontrval AS p
JOIN tr_ds_fut.wrds_contract_info AS c
    ON p.futcode = c.futcode
WHERE c.contrcode = %(contrcode)s
  AND p.date_ BETWEEN %(start)s AND %(end)s
ORDER BY p.date_, c.lasttrddate
"""
params = {
    "contrcode": 1986,
    "start": "2021-01-01",
    "end": "2025-12-31",
}
raw = db.raw_sql(sql, params=params)
```

`scripts/download_wrds.py` saves raw rows. `data.prepare_panel` cleans dates and positive prices, rejects conflicting contract/date duplicates, computes maturity and log settlement, selects nearby contracts, and requires a complete ordered panel. Dates with insufficient eligible contracts fail validation; a wholly absent session cannot be detected from the panel alone. Cleaning currently reports final panel coverage rather than a full ledger of excluded rows.

`prepare_kalman_data` pivots log settlements and `tau` by maturity rank, preserving chronological order and computing actual date gaps in calendar years. The Databento adapter supplies a separate provider history through the same panel interface; WRDS and Databento histories remain separate. The live path also uses a settlement calendar and records same-contract forecasts without recalibration.

**Data checks to retain or extend:**

1. Verify the metadata table contains exactly one relevant row per `futcode`; otherwise a join can inflate counts. Fail visibly on unexpected duplicates; do not silently drop them.
2. Check uniqueness of `(date, futcode)` after joining.
3. Check `date_ < lasttrddate`, stable expiry metadata for each contract, nonnull dates and prices, strictly positive prices, and consistency of `tau` with those dates.
4. Sort by `date_`, maturity, and a deterministic contract tie-breaker; select ranks 1–6 after filtering invalid/unexpired contracts. Report when a missing near-month quote causes a more distant contract to enter the WRDS panel.
5. Report excluded rows and incomplete or missing sessions against a documented calendar. Preserve the current complete-panel policy; missing-maturity support is future work.
6. Inspect extreme day-to-day jumps, stale settlements, contract roll transitions, and unexpectedly low-volume far contracts. Avoid treating a change in front-month contract identity as a spot return.
7. Inspect representative real curves and maturity coverage before interpreting fitted factors.
8. Store a clean long-form Parquet and optionally a wide-by-rank log-price matrix for PCA. Preserve `tau` and `futcode` in the long dataset.

**Do not commit licensed WRDS price data, downloaded raw files, `.env`, or credentials to a public repository.** Provide a WRDS extraction script and a synthetic/demo-data path so other users can reproduce the code without WRDS access.

## 3. Model M2 — Schwartz–Smith two-factor model

### 3.1 Physical-measure latent dynamics

Let

\[
\log S_t = \chi_t + \xi_t,
\]

where `chi` is a short-run mean-reverting deviation and `xi` a persistent equilibrium log-price:

\[
 d\chi_t=-\kappa\chi_t\,dt + \sigma_\chi\,dW_t^\chi,
 \qquad
 d\xi_t=\mu_P\,dt + \sigma_\xi\,dW_t^\xi,
\]

\[
 d\langle W^\chi,W^\xi\rangle_t = \rho\,dt,
 \qquad
 \kappa>0,\;\sigma_\chi>0,\;\sigma_\xi>0,\;|\rho|<1.
\]

**Use exact Gaussian discretization**, not Euler approximation. For observation gap `delta_t` measured in years:

\[
 x_t = G_t x_{t-1} + d_t + \eta_t,\quad
 x_t=\begin{bmatrix}\chi_t\\\xi_t\end{bmatrix},\quad
 G_t=\begin{bmatrix}e^{-\kappa\Delta_t}&0\\0&1\end{bmatrix},\quad
 d_t=\begin{bmatrix}0\\\mu_P\Delta_t\end{bmatrix}.
\]

The process covariance is

\[
Q_t=\begin{bmatrix}
 \frac{\sigma_\chi^2}{2\kappa}(1-e^{-2\kappa\Delta_t}) &
 \frac{\rho\sigma_\chi\sigma_\xi}{\kappa}(1-e^{-\kappa\Delta_t}) \\
 \frac{\rho\sigma_\chi\sigma_\xi}{\kappa}(1-e^{-\kappa\Delta_t}) &
 \sigma_\xi^2\Delta_t
\end{bmatrix}.
\]

The off-diagonal term is essential when `rho != 0`. Use actual gaps between trading dates so weekends and holidays are correctly handled.

### 3.2 Risk-neutral futures observation equation

For contract `j` at time `t`, with `tau_tj` years to cutoff:

\[
 y_{t,j}=\log F(t,T_j)
 = e^{-\kappa\tau_{t,j}}\chi_t + \xi_t + A(\tau_{t,j})+\epsilon_{t,j}.
\]

A convenient risk-neutral parameterization is

\[
 d\chi_t=(-\kappa\chi_t+q_\chi)dt+\sigma_\chi dW_t^{\chi,Q},
 \qquad d\xi_t=\mu_Q dt+\sigma_\xi dW_t^{\xi,Q}.
\]

It gives

\[
A(\tau)=\frac{q_\chi}{\kappa}(1-e^{-\kappa\tau})+\mu_Q\tau+\tfrac12 V(\tau),
\]

\[
V(\tau)=\frac{\sigma_\chi^2}{2\kappa}(1-e^{-2\kappa\tau})
 +\sigma_\xi^2\tau
 +\frac{2\rho\sigma_\chi\sigma_\xi}{\kappa}(1-e^{-\kappa\tau}).
\]

Here `mu_P` governs the physical filtering transition, while `mu_Q` and `q_chi` govern the pricing curve; the differences represent risk adjustments. The `0.5 * V(tau)` term is the lognormal convexity adjustment and **must not be omitted** from a risk-neutral pricing implementation.

The existing code uses the equivalent conventions `mu_P = mu_xi`, `mu_Q = mu_xi - lambda_xi = delta`, and `q_chi = -lambda_chi`. Document and validate this sign mapping against the primary paper; the reframing does not change these equations.

For `m_t` observable maturities, the Kalman observation system is

\[
 \mathbf y_t=H_t x_t+\mathbf a_t+\epsilon_t,
 \qquad
 (H_t)_{j,:}=[e^{-\kappa\tau_{t,j}},\;1],
 \qquad
 (\mathbf a_t)_j=A(\tau_{t,j}).
\]

The current filter uses `R = sigma_epsilon**2 * I`, with scalar log-price noise estimated during calibration. Maturity-specific or correlated observation noise is a future robustness study, not the current specification.

**Identification warning:** Futures quotes alone may weakly identify some combinations of physical drift and risk premiums. The main saved-analysis calibration fixes `lambda_chi = 0`, equivalent to `q_chi = 0`. Preserve this restriction for the main comparison. The legacy notebook estimator allows an unrestricted `lambda_chi`; do not mix its estimates with the restricted workflow. The short maturity span of the six nearby contracts can limit separation of χ and ξ. Report identification and sensitivity rather than assuming precise economic estimates or replication of the paper's empirical results.

### 3.3 Kalman filter and maximum likelihood

The two-factor filter uses time-varying `H_t`, `a_t`, `G_t`, and `Q_t`. Validate each step independently:

1. Predict state mean/covariance using `G_t`, `d_t`, `Q_t`.
2. Compute the **predicted** observation mean `H_t x_{t|t-1} + a_t`.
3. Compute innovation and `S_t = H_t P_{t|t-1} H_t.T + R_t`.
4. Accumulate multivariate Gaussian predictive log likelihood using Cholesky solves, not matrix inversion.
5. Update state and covariance with a numerically stable covariance formula (Joseph form or verified equivalent).
6. Verify irregular gaps and causal information use. Missing maturities are currently rejected; future support must not impute evaluation-day prices.

The existing MLE uses bounded `L-BFGS-B` on calibration observations only. `run_analysis` uses two deterministic two-factor starting points, fixes `lambda_chi=0`, selects the highest finite likelihood among successful fits, and stops if calibration fails. The one-factor estimator uses one start. Assess start sensitivity, active bounds, gradients, and parameter uncertainty before drawing economic conclusions; optimizer success alone is insufficient.

The current two-factor default initializes `[chi, xi] = [0, mean(first_log_curve)]` and `P0 = diag(0.1, 0.1)`. The first curve is then also used in an innovation likelihood contribution. Audit this data-dependent initialization and first-day likelihood treatment. Compare a documented conditional or diffuse initialization, including the stationary short-factor variance `sigma_chi^2/(2*kappa)`, before proposing estimator changes. Never reset states across consecutive sessions or use future observations in filtering.

The filter returns prior and filtered means, filtered covariances, innovations, and per-date likelihoods. Saved bundles retain filtered means, reconstruction tables, forecast tables, and per-date likelihoods, but do not persist covariance paths or innovation covariance matrices. Expanded uncertainty and whitened-innovation reports will need those outputs. Kalman step functions are reusable, but the full `kalman_filter` loop is specific to Schwartz–Smith; the one-factor filter has a separate loop.

## 4. Existing one-factor and random-walk baselines

Preserve the implemented one-factor model in `one_factor.py`:

- State transition: `x[t] = x[t-1] + mu * dt + eta[t]`, with process variance `sigma**2 * dt`.
- Log futures observation: `y[t,j] = x[t] + (delta + 0.5 * sigma**2) * tau[t,j] + epsilon[t,j]`.
- Scalar log-price measurement noise, bounded MLE, and a Joseph-form covariance update.

This is a Brownian log-price term-structure model, not the OU baseline from the original project proposal. Retain its reconstructions and forecasts as the existing comparison. An OU alternative may be evaluated later as a separately named model after validation; it is not implemented in this step.

The random walk repeats the origin's rank-wise log curve in the main backtest. The daily view instead follows contract identity and needs that contract's previous quote. Random walk is a forecast benchmark, not a fitted latent factor reconstruction.

## 5. Planned PCA benchmark for dimensionality and reconstruction

`pca_baseline.py` is currently a placeholder. The following methodology is planned; this documentation update does not implement it. Fit PCA on calibration log settlement levels, with six columns representing maturity ranks 1–6.

- Center each rank using its calibration-period mean. The primary analysis uses covariance PCA without variance standardization; standardized PCA and changes in log prices are separate sensitivity analyses with a different interpretation.
- Compute the full scree and cumulative explained-variance curves, and compare `K = 1` and `K = 2` reconstructions. Do not assert in advance that two components suffice.
- Hold centering and loadings fixed on validation/evaluation dates. Project each current curve onto those loadings and reconstruct it as the training mean plus its retained component contribution. These scores use current-day observations and are not forecasts.
- Plot rank-dependent loadings and score histories, and evaluate reconstruction errors on the identical dates/ranks used by M1/M2.
- Record component sign conventions, scale, and sensitivity to close eigenvalues. Compare structural loading patterns and factor subspaces as well as score/state correlations; PCA components need not equal χ and ξ.
- Because maturity-rank columns roll at expiration, the PCA panel does not represent perfectly fixed maturities. Explicitly disclose this and consider interpolating to common fixed time-to-maturity nodes in a later extension. No interpolation using future dates.

Structural loadings depend on each observed `tau`; PCA loadings are fixed coefficients on rolling rank columns. Compare them on explicitly aligned maturity support, and label whether an analysis uses contract identity, maturity rank, or a constant-maturity grid.

AR/VAR score forecasting is an optional later extension. It is not required for the primary PCA reconstruction benchmark and must not substitute current-day scores for prior forecasts. Existing structural and random-walk forecasts remain available independently.

## 6. Planned modeling evaluation

### 6.1 Calibration and evaluation windows

Preserve the current chronological split for the main fixed-calibration study:

- **Calibration / train:** 2021-01-01 to 2023-12-31.
- **Validation:** 2024-01-01 to 2024-12-31, for numerical checks and method choices.
- **Evaluation / existing test label:** 2025-01-01 to 2025-12-31. This period was already explored in the notebooks and must not be described as an untouched holdout.

Fit structural parameters, PCA centering/loadings, and any scaling or benchmark dynamics on calibration data only. Keep them fixed for the main validation/evaluation. Current-day curves may update structural filtered states or PCA scores to reconstruct that day; this is reconstruction under fixed calibration, not a future-price prediction. For forecasting, score the prior prediction before updating with the target observation.

Any retrospective full-sample fit or rolling/subperiod recalibration must be separately labeled with its actual calibration window. A stability refit is a different experiment from the existing fixed-parameter backtest and must not silently replace its saved parameters or results.

### 6.2 Reconstruction and dimensionality

For structural models, compute `fitted_log = H_t @ filtered_state + A(tau_t)` using each model's existing observation equation. For planned PCA, compute the training mean plus the retained score/loading contribution. Define reconstruction residuals as `observed_log - fitted_log` for both approaches.

- Report pooled and per-rank log-price RMSE, MAE, and mean residual for calibration, validation, and evaluation separately, with observation counts and a common scoring sample.
- Inspect residuals against actual `tau` as well as maturity rank. Include representative observed/fitted curves and errors through time, with roll dates identified.
- Report dollar errors from exponentiated fitted log values as a supplementary descriptive measure. Do not call them conditional price means or use them as a forecast evaluation.
- Report PCA explained-variance ratios, cumulative variance, and scree plots from the calibration sample. Held-period reconstruction error is a separate measure; any explained-variance statistic on evaluation data must state its definition.
- Compare one-factor, two-factor, one-PC, and two-PC representations without presuming that more factors are economically correct or that training fit establishes superiority.

### 6.3 Residual and innovation diagnostics

Reconstruction residuals use observations already assimilated into the state or score. Kalman innovations use the prior curve before assimilation. Analyze these quantities separately:

- For reconstruction residuals, report maturity-specific mean, variance, autocorrelation, cross-maturity correlation, tail behavior, and dates with large curve errors.
- For predictive innovations, inspect mean, autocorrelation, covariance, and Gaussian calibration. Whiten with the Cholesky factor of the innovation covariance `S_t`, not with the measurement-noise variance alone.
- Interpret innovation calibration after accounting for initialization and parameter estimation. Do not apply a predictive white-noise expectation automatically to filtered reconstruction residuals.
- Compare M1/M2 one-step predictive likelihood on common dates. Deterministic PCA reconstruction has no directly comparable predictive likelihood without an explicit stochastic model.

The current bundles do not contain whitened innovations or their covariance matrices; producing these reports is future implementation work.

### 6.4 Factor interpretation and stability

- Plot estimated χ and ξ, their maturity loadings, and hypothetical factor shocks with fixed parameters. Report short-factor half-life and uncertainty only after mathematical validation and estimation diagnostics support them.
- Compare PCA scores and structural states on aligned dates, with explicit sign/scale conventions. Compare loading vectors and subspaces; close PCA eigenvalues can make individual loading vectors unstable even when the subspace is stable.
- Evaluate structural parameter sensitivity to optimizer starts, initial state/covariance, calibration windows, and maturity coverage. Record convergence, active bounds, objective values, and uncertainty or profile-likelihood results where available.
- Study rolling or subperiod parameter estimates and PCA loading/subspace stability as separately labeled refit experiments. Preserve the original fixed-calibration analysis.
- Treat associations with market episodes as descriptive evidence, not causal identification. Futures prices do not reveal the true latent states.

### 6.5 Secondary forecast evaluation

Preserve rank-wise forecasts at horizons of 1, 5, and 20 trading observations, pooled and per-rank log RMSE/MAE, price-median errors, and one-step state-space predictive likelihood. Forecast origins start at or after calibration completion; longer horizons can have fewer early validation targets. All models within a horizon must share the same dates and target ranks.

Keep the daily same-contract forecasts and immutable issuance records. Label same-contract and rank-based targets explicitly, particularly around rolls. Reconstruction quality and forecast accuracy answer different questions; neither alone identifies an economic factor.

### 6.6 Synthetic verification before factor interpretation

Verify the formulas independently before relying on simulations generated by those same functions. Then simulate exact state dynamics and the observation model to test:

- Transition and pricing limiting cases, zero correlation, correlated shocks of both signs, small maturities/gaps, and irregular calendar gaps. Distinguish mathematical zero-noise limits from filter APIs requiring positive noise.
- Symmetry and positive-semidefiniteness of process and state covariances, and agreement of Gaussian likelihood and posterior updates with independent calculations.
- State recovery, reconstruction error, covariance behavior, and sensitivity to first-day initialization and likelihood treatment.
- Innovation calibration under known parameters after a documented initialization period.
- Recovery of well-identified synthetic parameters across multiple seeds, with explicit uncertainty and optimizer failures. Do not promise recovery of weakly identified drifts or risk premiums.
- One-factor and two-factor data-generating systems to study factor discrimination and overfitting. After PCA exists, assess its recovered dimensionality and subspace on these controlled data.
- A modest Monte Carlo study of approximately 10–30 seeds, reporting variability and failure rates rather than a single favorable run.

The current single-seed demo is not this validation study. Synthetic verification tests numerical correctness and model discrimination; it cannot establish that empirical WTI has exactly two factors.

## 7. Planned research deliverables and presentation

1. A mathematical validation report for the existing Schwartz–Smith equations, Kalman filter, initialization, and MLE, supported by independent numerical checks.
2. Data-quality and maturity-convention documentation built on the current WTI panel pipeline.
3. Structural factor trajectories, loadings, uncertainty where available, and same-day reconstruction tables and plots.
4. A training-fitted PCA benchmark with scree analysis, scores, loadings, and one-/two-component reconstructions.
5. Common-sample structural/PCA reconstruction metrics, residual and innovation diagnostics, and parameter/loading stability studies.
6. The existing forecasting workflow retained and clearly labeled as secondary.

The browser and Streamlit dashboards already consume saved analyses. Preserve historical curves, fitted curves, filtered factors, hypothetical shocks, and forecast/daily views. Later add modeling reports and PCA presentation after validated outputs exist. Do not trigger WRDS queries or MLE on control changes. A dashboard presents results; visual polish does not establish mathematical validity or identification.

## 8. Existing architecture to reuse

| Layer | Files | Reuse and planned extension |
| --- | --- | --- |
| Acquisition and panel | `scripts/download_wrds.py`, `scripts/prepare_panel.py`, `data.py` | Preserve contract-level ingestion, maturity convention, validation, and matrix preparation |
| Structural model math | `schwartz_smith.py`, `one_factor.py`, `simulation.py`, `demo.py` | Validate existing equations and simulation; retain the implemented one-factor baseline |
| Filtering and estimation | `kalman.py`, `estimation.py` | Reuse filtering and MLE; audit initialization, uncertainty, and identification |
| Analysis and artifacts | `analysis.py`, `scripts/run_empirical.py`, `scripts/run_synthetic.py` | Preserve `AnalysisBundle`, Parquet/JSON outputs, fixed-calibration analysis, and forecast tables; extend modeling reports later |
| Forecast applications | `evaluation.py`, `daily.py`, `daily_update.py`, `scripts/update_daily.py` | Preserve horizon propagation, common targets, same-contract reviews, and issued forecasts |
| Live ingestion/publication | `live_feed.py`, `live_update.py`, `settlement_calendar.py`, `blob_store.py`, `scripts/live_daily.py` | Preserve provider separation, calendar checks, and saved-history publication |
| Presentation/export | `app.py`, `web/`, `web_bundle.py`, `scripts/export_web_bundle.py` | Reuse saved-result consumers and views; add modeling presentation after reports exist |
| Statistical benchmark/plots | `pca_baseline.py`, `plotting.py` | Both are placeholders; add PCA and shared modeling plots in later tickets |
| Tests and automation | `tests/`, `.github/workflows/` | Retain data, math, leakage, app, daily, and browser checks; extend independent mathematical and modeling coverage |

Each existing bundle contains `panel.parquet`, `predictions.parquet`, `reconstructions.parquet`, `states.parquet`, `metrics.parquet`, and `metadata.json`. Future artifact extensions need explicit schemas and compatibility with both dashboards; this step changes no files in that pipeline.

Keep the existing Python scientific stack and static JavaScript frontend. Avoid new dependencies or a wholesale architecture rewrite for the research reframing.

## 9. Implementation roadmap after this documentation step

These tickets replace the original build-from-scratch sequence. The pipeline and dashboards already exist. Do not proceed past unresolved mathematical failures, and report incomplete checks without claiming validation. No ticket implements wavelets.

### Ticket A — independent Schwartz–Smith and Kalman validation (next priority)

**Scope:** the existing `schwartz_smith.py`, `kalman.py`, `estimation.py`, simulation functions, and their tests.

**Work:**

1. Derive and independently check the physical transition, correlated process covariance, observation loadings, risk-neutral sign conventions, and convexity adjustment against the primary paper.
2. Validate irregular gaps, covariance composition across successive intervals, symmetry/positive-semidefiniteness, and small-gap/maturity limits. Use independently computed expectations or numerical integration so tests do not simply call the same formula twice.
3. Compare likelihood, gain, posterior mean, and Joseph covariance updates with independent Gaussian calculations, including low-noise and numerically difficult admissible cases.
4. Audit the data-dependent first-curve initialization, first-day likelihood contribution, and sensitivity to `x0`/`P0`. Clearly distinguish current behavior from candidate alternatives.
5. Check known-parameter innovation calibration and state recovery; perform a modest multi-seed MLE recovery/sensitivity study and record bounds, failed starts, and weakly identified parameters.
6. Preserve the existing causal state updates, calibration split, forecast targets, and bundle compatibility.

**Acceptance:** a reproducible report distinguishing confirmed formulas, test coverage, numerical limitations, initialization issues, and identification limits. Propose fixes only for demonstrated discrepancies; do not silently change equations or claim successful parameter recovery.

### Ticket B — structural reconstruction and residual reporting

**Scope:** build modeling reports from existing panel, state, and reconstruction outputs.

**Acceptance:** pooled/per-rank log RMSE, MAE, and bias; curve examples; residual diagnostics; separately identified predictive innovations and their covariance standardization. Add uncertainty outputs only with a documented schema and verified filter covariance paths. Forecast metrics and issued records remain available.

### Ticket C — PCA dimensionality and reconstruction benchmark

**Scope:** implement the planned methodology in Section 5 on the existing rank panel.

**Acceptance:** calibration-only centering/loadings, full scree analysis, one-/two-PC reconstructions and scores, loading plots, explicit sign/scale conventions, and common-sample reconstruction metrics. Tests must verify that changing evaluation prices cannot change the fitted transform. Current-day evaluation prices may change their reconstruction scores. PCA forecasting is optional later work.

### Ticket D — factor comparison, stability, and presentation

**Scope:** align structural and PCA outputs, run separately labeled sensitivity/refit experiments, and present saved modeling reports.

**Acceptance:** loading/subspace and score/state comparisons with ambiguity disclosed, structural parameter and PCA stability diagnostics, provenance for each calibration window, and both dashboards retaining their forecasting functionality. Do not claim economic or causal identification from correlation alone.

An OU one-factor alternative, missing-maturity filtering, constant-maturity interpolation, and richer observation noise are later robustness candidates. Prioritize Tickets A → B → C → D.

## 10. Completion criteria for the modeling study

- [ ] Existing data pipeline and forecast/daily workflows remain reproducible.
- [ ] Schwartz–Smith transitions, pricing, Kalman updates, and likelihood receive independent mathematical checks.
- [ ] Initialization, parameter constraints, convergence, and identification limitations are documented.
- [ ] Synthetic validation reports state/parameter recovery variability and failures across seeds.
- [ ] Structural reconstruction errors and residual diagnostics are separated from predictive innovations and forecasts.
- [ ] PCA reports dimensionality, loadings, scores, and reconstruction using calibration-fitted transforms.
- [ ] Structural/statistical comparisons use common observations and disclose maturity alignment and PCA ambiguity.
- [ ] Stability experiments record calibration windows and retain the original fixed-parameter analysis.
- [ ] Figures and tables report only computed results, with uncertainty and caveats.
- [ ] Dashboards consume saved outputs and preserve secondary forecasting views.

These are future acceptance criteria, not a declaration that the current modeling study is complete.

## 11. Interpretation and limitations

- Filtered χ and ξ are model-dependent estimates; futures prices do not reveal true latent factors.
- Short maturity coverage may weaken separation of short- and long-term effects. A two-factor specification does not prove that the data need exactly two factors.
- High training likelihood or low reconstruction error is insufficient evidence of economic identification or forecast superiority.
- The Gaussian state/noise assumptions, risk-premium restriction, and fixed parameters may be inadequate during jumps or regime shifts; residual and stability analyses should investigate this.
- Physical drift and risk premiums may be weakly identified. Report uncertainty and sensitivity rather than definitive economic claims.
- PCA components reflect a chosen sample, centering, scaling, and maturity representation; sign/rotation ambiguity limits direct naming as structural factors.
- `lasttrddate` is a practical cutoff proxy. Rolling ranks introduce contract changes into PCA columns and forecast targets.
- The 2025 period has already been explored. Record future method choices and avoid describing it as an untouched final holdout.
- Empirical reconstruction, PCA, residual, and stability findings must be computed and documented before publication. The included synthetic demo does not supply them.
- Preserve the separation of licensed WRDS prices, provider-specific live history, and shareable synthetic artifacts.

## 12. Reproducibility and operational continuity

Use the commands in [README.md](README.md) for extraction, panel preparation, saved analysis generation, export, and local dashboards. [DEPLOYMENT.md](DEPLOYMENT.md) documents optional hosting and the secondary daily update workflow. Package names, deployment identifiers, data paths, and public/private storage conventions remain compatible with the existing implementation.

Run the current Python and JavaScript project checks for future implementation changes, plus the relevant mathematical or modeling acceptance checks. Record actual test outcomes and any unavailable runtime. Do not regenerate or overwrite empirical result bundles merely to update research documentation.

## 13. Primary references

1. Schwartz, E., & Smith, J. E. (2000). “Short-Term Variations and Long-Term Dynamics in Commodity Prices.” *Management Science*, 46(7), 893–911. https://doi.org/10.1287/mnsc.46.7.893.12034 — core model and Kalman-style estimation motivation.
2. Author-hosted PDF of Schwartz & Smith: https://www.anderson.ucla.edu/documents/areas/fac/finance/72.pdf — consult the primary paper for the original conventions, estimation, and parameter-identification discussion.
3. Ballestra, L. V., & Tezza, M. (2025). Related multi-factor commodity pricing research: https://arxiv.org/html/2501.15596v1 — possible future extension; not required for the MVP.

---

**Next implementation action:** Execute Ticket A's independent mathematical validation of the existing Schwartz–Smith and Kalman filter implementations before adding reconstruction diagnostics or PCA. The current step updates documentation only.
