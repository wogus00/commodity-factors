# How Many Factors Does It Take to Model Commodity Futures?

**A comparative study of stochastic state-space models and statistical factor models**  
**Market:** NYMEX WTI light sweet crude oil (CL) futures  
**Data:** WRDS / LSEG Datastream Futures, daily contract-level settlements, 2021–2025  
**Target:** a reproducible 3–5 day research prototype, followed by a small Streamlit demo  
**Primary reference:** Schwartz, E. & Smith, J. E. (2000), *Short-Term Variations and Long-Term Dynamics in Commodity Prices*, *Management Science* 46(7), 893–911. DOI: https://doi.org/10.1287/mnsc.46.7.893.12034

> **Instructions for Codex:** Treat this document as the project specification. Build a small, testable research repository. Prioritize mathematical correctness, reproducibility, and honest out-of-sample evaluation over features or visual polish. Do not invent empirical results, silently make financial-data assumptions, or push WRDS credentials/data into Git. Work through the implementation tickets below and state any unresolved assumptions.

## 1. Research question and scope

**Question:** How much of the WTI futures term structure can be explained and forecast by one versus two economically motivated latent factors, and how do those models compare with PCA-based statistical factors?

The Schwartz–Smith framework decomposes log spot prices into a mean-reverting short-term deviation and a persistent long-term equilibrium component. A short-dated futures contract loads relatively heavily on the short-term factor, while a long-dated contract is more sensitive to the equilibrium level.

We will compare three approaches:

| ID | Model | Main role |
|---|---|---|
| **M1** | One-factor mean-reverting Gaussian/OU term-structure model | Parsimonious economic baseline |
| **M2** | Two-factor Schwartz–Smith state-space model, estimated with a Kalman filter and maximum likelihood | Core replication/investigation |
| **M3** | PCA with one and two components | Statistical-factor benchmark |

Two separate questions must be evaluated:

1. **Cross-sectional fit / reconstruction:** Can the model represent the observed curve at time *t* when information from that day is allowed?
2. **Out-of-sample prediction:** Can the model predict tomorrow's curve using only information available through today?

**Do not conflate reconstruction and forecasting.** A PCA reconstruction of the test day's observed curve is *not* a forecast. In-sample fitted Kalman-filter curves are also not the same as one-step-ahead predictions.

### Initial boundaries

- Use WTI CL only; Brent, natural gas, cross-commodity modeling, options pricing, trading strategies, and the 2025 multi-factor paper are **stretch extensions**, not MVP requirements.
- Use six nearest unexpired delivery-month contracts per day for the initial panel, subject to data checks. Expand to longer maturities if useful and well covered.
- Work with **log settlement prices** and positive-price observations; 2021–2025 avoids the major negative April 2020 front-month episode, but positivity still must be verified.
- Prefer a small working end-to-end pipeline to a comprehensive but unfinished library.

## 2. Data source: verified WRDS discovery

The WRDS schema `tr_ds_fut` is accessible. Preliminary inspection found:

- `tr_ds_fut.wrds_contract_info`: per-contract identifiers, delivery label and dates; columns include `contrcode`, `futcode`, `dsmnem`, `contrdate`, `lasttrddate`, `sttlmntdate`, `expirationdate`, `exchtickersymb`.
- `tr_ds_fut.dsfutcontrval`: daily individual-contract prices, keyed by `futcode` and `date_`; columns include `settlement`, `volume`, `openinterest`.
- `tr_ds_fut.wrds_fut_contract`: alternative expanded price table, to be used only if needed.

**Confirmed CL contract family:** `contrcode = 1986`, `contrname = 'CRUDE OIL (LIGHT SWEET)'`, `exchtickersymb = 'CL'`. Do not mix in WTI swaps, Brent, micro CL, or regional differentials.

**Coverage query already executed:** 2021-01-04 through 2025-12-31; **161,354 non-null daily price records**, **193 distinct `futcode` contracts**, **1,256 trading dates** before filtering. These are *coverage counts*, **not** final clean-panel counts or model results.

`lasttrddate` and `sttlmntdate` were populated and equal in the inspected far-dated contract rows; `expirationdate` was null. For the MVP, use `lasttrddate` as the time-to-maturity cutoff, while documenting that futures pricing may conventionally refer to delivery/contract settlement rather than the final trading date. Investigate sensitivity if material.

### Expected tidy dataset

| column | definition |
|---|---|
| `date` | market observation/trading date |
| `futcode` | unique Datastream futures contract ID |
| `contract_label` | Datastream mnemonic / delivery month (`dsmnem`, `contrdate`) |
| `last_trade_date` | final trading date from metadata |
| `settlement` | observed daily settlement price, USD per barrel (confirm vendor convention) |
| `volume`, `open_interest` | optional liquidity diagnostics |
| `tau` | `(last_trade_date - date).days / 365.25` |
| `maturity_rank` | 1 = nearest unexpired listed contract on that date |
| `log_settlement` | natural logarithm of positive `settlement` |

### WRDS extraction (reference SQL)

Use **parameterized SQL**, since `%` wildcard characters in literal SQL strings previously caused a SQLAlchemy/WRDS driver error in this environment.

```python
sql = """
SELECT
    p.date_ AS date,
    p.futcode,
    c.dsmnem AS contract_label,
    c.contrdate,
    c.lasttrddate AS last_trade_date,
    p.settlement,
    p.volume,
    p.openinterest AS open_interest
FROM tr_ds_fut.dsfutcontrval AS p
JOIN tr_ds_fut.wrds_contract_info AS c
    ON p.futcode = c.futcode
WHERE c.contrcode = %(contrcode)s
  AND p.date_ BETWEEN %(start_date)s AND %(end_date)s
  AND c.lasttrddate > p.date_
  AND p.settlement > 0
ORDER BY p.date_, c.lasttrddate
"""
params = {
    "contrcode": 1986,
    "start_date": "2021-01-01",
    "end_date": "2025-12-31",
}
raw = db.raw_sql(sql, params=params)
```

**Important engineering checks:**

1. Verify the metadata table contains exactly one relevant row per `futcode`; otherwise a join can inflate counts. Fail visibly on unexpected duplicates; do not silently drop them.
2. Check uniqueness of `(date, futcode)` after joining.
3. Check `date < last_trade_date`, nonnull dates and prices, strictly positive prices, and `tau > 0`.
4. Sort by `date`, `last_trade_date`, and a deterministic contract tie-breaker; rank contracts within each date. Select ranks 1–6 **after** filtering invalid/unexpired contracts.
5. Count trading dates with 0–5 valid nearby contracts. Choose and document a deterministic missing-data policy (prefer retaining days with all six contracts for the first prototype; later allow missing observation rows in the Kalman filter).
6. Inspect extreme day-to-day jumps, stale settlements, contract roll transitions, and unexpectedly low-volume far contracts. Avoid treating a change in front-month contract identity as a spot return.
7. Confirm at least one real futures-curve plot from sample dates before modeling.
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

For `m_t` observable maturities, the Kalman observation system is

\[
 \mathbf y_t=H_t x_t+\mathbf a_t+\epsilon_t,
 \qquad
 (H_t)_{j,:}=[e^{-\kappa\tau_{t,j}},\;1],
 \qquad
 (\mathbf a_t)_j=A(\tau_{t,j}).
\]

Start with `R = sigma_obs**2 * I` (an estimated scalar noise variance). A diagonal maturity-specific `R` is an optional robustness extension. Model log settlements; do not use price-level noise with log observations.

**Identification warning:** Futures quotes alone may weakly identify some combinations of physical drift and risk premiums. The original paper discusses identification/robustness limitations. For the first working version, **fix `q_chi = 0`** and estimate `mu_Q`, recording this as an explicit restriction. Add `q_chi` only after testing parameter identifiability and optimization stability. Do not claim that a restricted calibration reproduces all original paper parameter estimates. If `mu_P` is poorly identified, report that rather than forcing plausible-looking parameter values.

### 3.3 Kalman filter and maximum likelihood

Implement the linear Gaussian filter with time-varying `H_t`, `a_t`, `G_t`, and `Q_t`:

1. Predict state mean/covariance using `G_t`, `d_t`, `Q_t`.
2. Compute the **predicted** observation mean `H_t x_{t|t-1} + a_t`.
3. Compute innovation and `S_t = H_t P_{t|t-1} H_t.T + R_t`.
4. Accumulate multivariate Gaussian predictive log likelihood using Cholesky solves, not matrix inversion.
5. Update state and covariance with a numerically stable covariance formula (Joseph form or verified equivalent).
6. Support variable `delta_t` and ideally missing maturities without imputing test-day prices.

Estimate parameters by minimizing negative innovation log likelihood on the **training period only** with `scipy.optimize.minimize`. Constrain positive parameters by `exp`/`softplus` and correlation by `tanh`; report actual units and optimization diagnostics. Use multiple initializations if the objective is nonconvex.

Initialize the short-run factor using its stationary variance `sigma_chi^2/(2*kappa)`; use a documented diffuse or estimated prior for the nonstationary long-run factor. Initialization sensitivity should be tested. Never reset the Kalman state across normal consecutive dates or implicitly peek at the next observation.

Save both filtered states `x_{t|t}` (diagnostics, reconstruction) and **one-step predictive** states/curves `x_{t|t-1}` (honest prediction).

## 4. Model M1 — one-factor baseline

Use a simple **economically coherent one-factor log-price OU**:

\[
 dX_t=\kappa_1(\theta_P-X_t)dt+\sigma_1 dW_t^P,
 \qquad \log S_t=X_t.
\]

Under a risk-neutral OU with mean `theta_Q`, the futures price is

\[
 \log F(t,T)=\theta_Q+(X_t-\theta_Q)e^{-\kappa_1\tau}
 +\frac{\sigma_1^2}{4\kappa_1}(1-e^{-2\kappa_1\tau}).
\]

The exact physical transition uses `theta_P + (X_prev-theta_P)*exp(-kappa_1*delta)`, with variance `sigma_1^2*(1-exp(-2*kappa_1*delta))/(2*kappa_1)`. Estimate with the same Kalman/MLE machinery and comparable measurement noise assumptions. Do **not** substitute a naive single time-series OU on front-month returns for a genuine futures-curve model.

This is a useful structural baseline, though it is not the unique possible nested restriction of the two-factor framework; describe it accurately rather than claiming formal nesting.

## 5. Model M3 — PCA benchmarks

Fit PCA on the **training sample only**, using six columns representing *maturity rank* 1 through 6 of `log_settlement`.

- Fit training mean and PCA loadings using training days only; retain `K = 1` and `K = 2` components.
- Report explained-variance ratios and reconstruction errors, clearly labeled as **same-day reconstruction**.
- To make an **out-of-sample forecasting** benchmark, fit a simple `VAR(1)` or separate `AR(1)` models to the training PCA scores; forecast next-day scores from information available at `t-1`; map forecast scores back through the fixed PCA loadings and training mean.
- As an additional sanity check, include the **random-walk/no-change forecast**: tomorrow's observed rank-wise log curve equals today's rank-wise log curve.
- Because maturity-rank columns roll at expiration, the PCA panel does not represent perfectly fixed maturities. Explicitly disclose this and consider interpolating to common fixed time-to-maturity nodes in a later extension. No interpolation using future dates.

For all comparisons, label whether the target is a contract identifier, maturity rank, or constant-maturity curve. Mixing these targets invalidates a forecast comparison.

## 6. Experiment design and evaluation

### 6.1 Time split

Initial fixed chronological split:

- **Train:** 2021-01-01 to 2023-12-31.
- **Validation:** 2024-01-01 to 2024-12-31 (numerical/debugging and model decisions).
- **Final test:** 2025-01-01 to 2025-12-31 (holdout; report only after decisions are fixed).

Fit all parameter estimates, PCA transforms, scaling, observation-noise specifications, and baseline score dynamics without test-period information. On validation/test, **hold fitted parameters fixed**. For a rolling one-day-ahead prediction, allow sequential state updating *after* each day's observation becomes available, but never before scoring that day's prediction.

### 6.2 Primary metrics

- **One-day-ahead log-price RMSE / MAE** across all scored maturities.
- **Per-maturity-rank RMSE / MAE**; short end vs long end.
- **Predictive log likelihood** for probabilistic state-space models (compare M1 with M2; do not pretend deterministic PCA provides this unless you explicitly add an error distribution).
- **Same-day reconstruction RMSE** reported separately.
- Optional: price-level absolute errors, sample curves on representative dates, residual correlations, forecast bias.

For fair headline comparisons, use a *common sample* of dates and the same six ranked observations. If one model predicts an individual contract and another predicts ranked maturities, align the target and say how the roll is treated. Consider an alternative evaluation only on contracts present on both consecutive dates, to isolate genuine same-contract price changes from contract-roll discontinuities.

Do not declare a model superior just because PCA reconstructs curves better in sample. Include measurement noise, estimation uncertainty, likelihood diagnostics, and sensitivity to time split and maturity selection.

### 6.3 Synthetic verification (before empirical claims)

Generate state trajectories with the **exact transition covariance** and generate curves with the **same observation model**; then test:

- Deterministic special cases for `kappa`, `rho = 0`, zero/very small noise, and irregular `delta`.
- Exact covariance matrices are symmetric positive semidefinite (subject to numerical tolerance).
- With known parameters, the Kalman innovation statistics are approximately calibrated over sufficiently long simulations.
- The estimator recovers *well-identified* synthetic parameters within realistic uncertainty across multiple seeds. Do not promise perfect recovery of weakly identified risk-premium parameters.
- Generate a two-factor system with meaningful factor variation and check whether M2 can improve test forecasts/likelihood over M1; also generate one-factor data to test whether M2 overfits.
- Run a small Monte Carlo (`~10–30` seeds for MVP; more later) and show variability, not just a single favorable seed.

The synthetic exercise tests code correctness and model discriminability, **not** whether the empirical WTI market truly has exactly two factors.

## 7. Deliverables and visualizations

**Minimum research deliverables:**

1. A reproducible WRDS-to-Parquet data loader and validated WTI curve panel.
2. Exact two-factor simulation and Kalman-filter implementation, with tests.
3. M1/M2 likelihood estimation and fixed-parameter one-day-ahead predictions.
4. One-/two-component PCA and random-walk forecast baselines.
5. Results table and plots separating reconstruction from forecasting, with clear train/validation/test labels.
6. README explaining model assumptions, data access, setup, and commands.

**Streamlit app — only after core models are working:**

- Historical futures curves for chosen dates, observed vs model-predicted/fitted.
- Filtered `chi` and `xi` states over time, labelled as **estimated latent states** rather than ground truth.
- Per-maturity forecast-error comparison for M1/M2/PCA/no-change.
- Interactive **hypothetical** factor shocks and the resulting futures-curve response (do not present counterfactuals as forecasts).
- Optional: slider for mean-reversion `kappa` and short-/long-factor volatility; keep displayed parameters consistent with actual model formulas.

A dashboard is presentation, **not** evidence that the model is validated.

## 8. Recommended repository layout

```text
commodity-futures-factors/
├── README.md
├── PROJECT_PLAN.md
├── pyproject.toml
├── .gitignore
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/                  # ignored: licensed WRDS data
│   ├── processed/            # ignored: licensed WRDS data
│   └── synthetic/            # small generated samples can be committed
├── src/commodity_factors/
│   ├── __init__.py
│   ├── data.py               # WRDS pull, contracts, panel validation
│   ├── simulation.py         # exact synthetic SDE transitions
│   ├── schwartz_smith.py     # M2 G, Q, H, A formulas
│   ├── one_factor.py         # M1 physical transition and futures equation
│   ├── kalman.py             # generic linear Gaussian Kalman filter
│   ├── estimation.py         # constrained MLE and diagnostics
│   ├── pca_baseline.py       # PCA fit, score forecasts
│   ├── evaluation.py         # aligned forecast targets and metrics
│   └── plotting.py
├── scripts/
│   ├── download_wrds.py
│   ├── prepare_panel.py
│   ├── run_synthetic.py
│   └── run_empirical.py
├── tests/
│   ├── test_data.py
│   ├── test_state_space.py
│   ├── test_kalman.py
│   └── test_no_leakage.py
├── results/                  # generated metrics/figures
└── app.py                    # optional Streamlit app
```

Suggested stack: Python 3.11+, NumPy, pandas, SciPy, scikit-learn, matplotlib/plotly, PyArrow, pytest, PyYAML, WRDS, and Streamlit (dashboard only). Avoid major new dependencies unless necessary.

## 9. Work tickets for Codex

Each ticket should yield code, tests, documentation, and a short summary of assumptions. **Do not start the next ticket if the current ticket's tests fail.**

### Ticket A — WRDS extraction and validation (priority 1)

**Input:** this plan, working local WRDS login, metadata discovered above.  
**Implement:** `data.py`, `scripts/download_wrds.py`, `scripts/prepare_panel.py`, `test_data.py`.  
**Acceptance checks:** distinct `(date, futcode)` keys; contract family 1986; correct expiry logic; six sorted nearby contracts where available; logged exclusion counts; no false assumption of 7,536 clean observations; data excluded from version control.  
**Output:** local Parquet, data-quality report, and a basic curve plot.

**Prompt to give Codex:**

> Read `PROJECT_PLAN.md`, especially Sections 2 and 9. Implement only Ticket A. Use the existing WRDS Datastream Futures tables and CL `contrcode=1986`. Validate metadata join cardinality and curve construction. Produce a concise data-quality report, tests, a minimal CLI, and README usage. Do not fabricate data or commit licensed market prices. Ask for a credential only when needed at execution time; never hard-code one.

### Ticket B — exact state-space math and synthetic simulation (priority 2)

**Implement:** `schwartz_smith.py`, `one_factor.py`, `simulation.py`, and unit tests.  
**Acceptance checks:** equations match Sections 3–4; exact irregular-time transition; correlated shocks; correct risk-neutral convexity term; positive-semidefinite covariance; reproducible seeded simulation.

**Prompt to give Codex:**

> Read `PROJECT_PLAN.md`. Implement only Ticket B: pure NumPy/SciPy state-space matrices and reproducible synthetic simulation for M1 and M2. Test exact transition means/covariances, risk-neutral observation equations, irregular date gaps, and correlated shocks. Keep the formulas explicit and documented. Do not perform empirical estimation or build a dashboard.

### Ticket C — Kalman filter and maximum-likelihood estimation (priority 3)

**Implement:** `kalman.py`, `estimation.py`, likelihood and recovery tests.  
**Acceptance checks:** innovations likelihood computed stably; time-varying maturity observation matrix; no covariance blow-ups; positive parameter constraints; train-only estimation; fixed parameters during holdout; diagnostic optimizer logs.

**Prompt to give Codex:**

> Implement Ticket C in `PROJECT_PLAN.md` on top of Tickets A–B. Write a generic stable Kalman filter for time-varying observation matrices and irregular observation gaps. Optimize training-only negative innovation log likelihood with constrained parameters and multiple starting points. Begin with `q_chi=0` and isotropic measurement noise, disclose weak identification, and test against synthetic data. Expose both prior predictive and posterior filtered curves without confusing them.

### Ticket D — PCA, prediction evaluation, and plots (priority 4)

**Implement:** `pca_baseline.py`, `evaluation.py`, `scripts/run_empirical.py`, basic figures.  
**Acceptance checks:** chronological split; PCA fit and any normalization fit train-only; one-step-ahead PCA score forecasts; fair common-date/model comparisons; RMSE by maturity; reconstruction distinct from prediction; no-peek tests.

**Prompt to give Codex:**

> Implement only Ticket D of `PROJECT_PLAN.md`. Fit 1- and 2-PC PCA on training log settlements, forecast PCA scores with a simple AR(1)/VAR(1), and include a no-change baseline. Compare one-step-ahead predictions with M1/M2 on exactly aligned dates and maturity ranks. Report both pooled and by-rank RMSE, clearly separate reconstruction from forecasting, generate clean plots/tables, and add leakage tests. Never fit PCA on test data.

### Ticket E — Streamlit showcase (priority 5 / optional)

**Implement:** `app.py` consuming **saved, validated outputs**; never re-run costly WRDS queries or full MLE on widget changes.  
**Acceptance checks:** curve charts, latent states, comparison table, factor-shock slider; sensible defaults and labels; deterministic demo mode using synthetic data; no secrets in the frontend.

**Prompt to give Codex:**

> Implement only Ticket E. Build a minimal Streamlit research dashboard from already-generated result files, with a synthetic demo mode so it works without licensed WRDS data. Show observed vs predicted curves, filtered latent states, per-maturity forecast comparisons, and hypothetical short-/long-factor shocks. Do not claim fitted states are observed truth or imply that reconstruction is an out-of-sample prediction. Prioritize correctness and clarity.

## 10. Suggested execution order (3–5 focused days)

| Day | Primary objective | Done when... |
|---|---|---|
| 1 | Ticket A and contract-curve visualization | validated six-nearby WTI panel available locally |
| 2 | Ticket B + core Kalman mechanics | exact-model/synthetic tests pass |
| 3 | Ticket C + initial M2/M1 calibration | finite log likelihood; parameter diagnostics; one-step forecasts |
| 4 | Ticket D + robustness | leakage-free held-out comparison table and plots |
| 5 | Ticket E + README/polish (if time) | demo runs from saved outputs without WRDS connection |

If time is tight, prioritize **A → B → C → D**, skip the dashboard, and report well-validated model comparisons instead. Schedule a brief parameter-recovery experiment before presenting any empirical interpretation.

## 11. Definition of done

- [ ] WTI CL data identified via `contrcode=1986`, cleaned, and validated.
- [ ] All output observation dates and maturity cutoffs are traceable to actual contracts.
- [ ] M1 and M2 implement exact physical transitions and coherent risk-neutral pricing equations.
- [ ] Synthetic tests cover numerical correctness and parameter/forecast recovery limitations.
- [ ] Estimation only uses training observations; no test leakage or contemporaneous forecast peeking.
- [ ] PCA and state-space **forecasts** are compared on the same targets; reconstruction is separate.
- [ ] Results include RMSE by maturity rank, an overall table, at least one curve figure, and caveats.
- [ ] README has setup/run commands, assumptions, data-license note, and limitations.
- [ ] Dashboard exists **only if** it doesn't displace the validation work.

## 12. Interpretation and limitations to acknowledge

- Historical futures prices do **not** reveal the true latent factors. Kalman-filter trajectories are model-dependent estimates.
- A high training likelihood is not proof of forecast superiority; compare out-of-sample results and a no-change baseline.
- There is no guarantee that two factors will beat one factor on a particular five-year sample or horizon.
- The model assumes Gaussian state shocks, a specific risk-premium structure, and a simple measurement-noise model. Real oil markets can have jumps and regime shifts.
- WTI's 2021–2025 sample includes different oil-price regimes, so results may not transport to other periods.
- Some risk-neutral/physical parameters may be weakly identified by futures curves alone; report uncertainty and sensitivity rather than definitive economic claims.
- `lasttrddate` is initially a practical cutoff proxy, not necessarily the economically exact delivery horizon.
- Rank-based PCA has contract-roll artifacts; constant-maturity interpolation is a valuable *later* robustness analysis.
- WRDS price data are licensed; the public repository should provide code and reproducible synthetic datasets, not redistribute the underlying prices.

## 13. Primary references

1. Schwartz, E., & Smith, J. E. (2000). “Short-Term Variations and Long-Term Dynamics in Commodity Prices.” *Management Science*, 46(7), 893–911. https://doi.org/10.1287/mnsc.46.7.893.12034 — core model and Kalman-style estimation motivation.
2. Author-hosted PDF of Schwartz & Smith: https://www.anderson.ucla.edu/faculty/eduardo.schwartz/articles/72.pdf — consult the primary paper for the original conventions, estimation, and parameter-identification discussion.
3. Ballestra, L. V., & Tezza, M. (2025). Related multi-factor commodity pricing research: https://arxiv.org/html/2501.15596v1 — possible future extension; not required for the MVP.

---

**First action:** Execute **Ticket A**. Do not start calibration until you have inspected real curve plots and established that the contract identifiers, expiries, and settlements have been merged and ranked correctly.
