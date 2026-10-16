# Latent Factor Modeling of Crude Oil Futures

[Open the live demo](https://commodity-factors.vercel.app) · [Source code](https://github.com/wogus00/commodity-factors)

This showcase includes the model code, tests, and synthetic data. Research
notebooks are provided without saved outputs; licensed market data and local
results are excluded. The public repository starts from a clean snapshot and
does not include the private research commit history.

A research workspace for estimating and interpreting the latent factors of
the WTI crude oil futures term structure. The primary focus is the
Schwartz–Smith model's factor dynamics, maturity-dependent loadings, and
ability to reconstruct observed futures curves. PCA is a statistical
benchmark for studying dimensionality and comparing structural and statistical
factor representations.

The project builds on the existing data pipeline, model implementations,
saved-result architecture, and dashboards. Python forecast comparisons and daily
updates remain available as secondary applications outside the Vercel frontend. Python calibration runs
separately; the browser frontend and local Streamlit app consume saved results
without fitting on control changes. The package, environment, retain the operational name `commodity-factors`.

## Research objectives

1. Estimate and interpret the Schwartz–Smith short-term deviation χ and
   long-term equilibrium ξ from WTI futures settlements.
2. Evaluate how well the structural model explains and reconstructs observed
   curves across dates and maturities.
3. Introduce PCA as a statistical benchmark for factor dimensionality and
   maturity-dependent loadings.
4. Compare structural and statistical factors through reconstruction errors,
   residual diagnostics, and parameter and loading stability.

These are research objectives, not findings. No claim about the number of
factors needed for WTI, economic identification, or model superiority follows
from the included demo or optimizer convergence.

## Methodology and current implementation

The existing WRDS / LSEG Datastream pipeline selects the six nearest eligible
WTI CL contracts (`contrcode=1986`), retains contract identities and last
trading dates, and builds a complete panel of positive log settlements.
Maturity is years to `lasttrddate`, using 365.25 days per year. This cutoff is a
pricing-horizon proxy; maturity ranks change contract at rolls.

The two-factor state is `[χ, ξ]`: χ follows mean-reverting OU dynamics and ξ
follows Brownian dynamics with physical drift. Log futures prices load on χ
through `exp(-kappa * tau)` and on ξ through a unit loading, plus the
risk-neutral pricing adjustment. The implementation includes correlated
Gaussian transitions, actual calendar gaps, and the lognormal convexity term.
The current calibration fixes `lambda_chi=0` and estimates parameters by
training-period Kalman innovation likelihood. Filtered states produce
same-day curve reconstructions; prior states produce forecasts.

| Component | Current capability | Planned modeling work |
| --- | --- | --- |
| Data pipeline | Contract-level cleaning, maturity ranking, complete-panel validation | Additional data-quality and maturity-proxy diagnostics |
| Schwartz–Smith and Kalman filter | Pricing and transition equations, filtering, bounded MLE | Independent mathematical validation, initialization and identification analysis |
| Structural reconstructions | Saved `reconstructions.parquet` and fitted-curve views | Pooled and per-maturity reconstruction metrics and residual reports |
| Latent factors | Saved filtered χ and ξ, factor histories and hypothetical shocks | State uncertainty, interpretation, and parameter stability studies |
| PCA | Constant-maturity PCA with training-only centering/scaling, diagnostics and reconstruction | Compare statistical factors with structural states |
| Forecasting | Random walk, Brownian one-factor, and two-factor backtests; daily forecasts | Retained as a secondary evaluation |

The one-factor implementation uses a Brownian log-price state with a linear
maturity adjustment. An OU one-factor alternative is future work. The current
tests check selected equations and filter mechanics; independent validation
and multi-seed recovery are still required before interpreting fitted factors
as economic evidence. Wavelets are outside the current scope.

## Planned evaluation framework

The detailed methodology and implementation order are in
[PROJECT_PLAN.md](PROJECT_PLAN.md). The modeling evaluation will distinguish:

- **Curve reconstruction:** pooled and per-rank log-price RMSE, MAE, and bias,
  representative fitted curves, and residual patterns against actual maturity.
  Both structural filtered states and PCA scores may use the current curve;
  these are same-day reconstructions, including on evaluation dates.
- **Factor dimensionality and loadings:** training-sample PCA scree and
  cumulative explained variance, with one- and two-component reconstructions
  compared on the same observations. Center log levels with a training mean;
  any standardization is a separately reported sensitivity analysis.
- **Residual diagnostics:** reconstruction residuals by date and maturity,
  cross-maturity correlation, autocorrelation, and tail behavior. Evaluate
  Kalman innovations and their covariance-standardized form separately;
  filtered reconstruction residuals are not predictive innovations.
- **Structural/statistical factor comparison:** aligned loading patterns,
  score/state associations, and factor subspaces. Account for PCA sign, scale,
  and rotation ambiguity; a principal component has no automatic economic
  identity as χ or ξ. Structural loadings use actual `tau`, while rank-based
  PCA loadings refer to rolling rank columns.
- **Stability:** sensitivity to optimizer starts, state initialization,
  calibration windows, and maturity coverage; parameter uncertainty and PCA
  loading/subspace stability. Separate planned rolling or subperiod refits
  from the existing fixed-parameter evaluation.
- **Secondary forecasts:** preserve aligned rank-wise errors at 1, 5, and 20
  trading observations and one-step predictive likelihood. Daily forecasts
  continue to use the same contract across dates.

Preserve the current 2021–2023 calibration, 2024 validation, and 2025 evaluation
periods. Fit model parameters and planned PCA centering/loadings on calibration
data only, then keep them fixed for the main evaluation. Current-date curve
observations may update filtered states or PCA scores for reconstruction.
Forecasts use observations only through their origin. The 2025 period was
already explored in the notebook and is not an untouched holdout.

The saved bundles currently provide structural reconstructions and forecast
metrics. PCA comparisons, reconstruction summary reports, expanded residual
diagnostics, and stability studies have not yet been implemented. Synthetic
examples demonstrate the workflow and do not establish empirical WTI results.

## Next research step

Validate the existing Schwartz–Smith and Kalman implementations before adding
PCA: independently check exact transition means and correlated covariances,
risk-neutral pricing conventions and convexity, Gaussian likelihood, and
Joseph covariance updates. Add boundary and irregular-gap checks, examine the
first-curve initialization and first-day likelihood treatment, and assess
innovation calibration and parameter recovery across synthetic seeds.
Document any discrepancies before proposing equation or estimator changes.

## Run the web workspace

The static dashboard focuses on latent factor modeling in six sections:
Overview, Futures Term Structure, Schwartz–Smith Model, PCA Factor Analysis,
PCA Robustness, and Model Comparison. It retains the Plotly chart workspace,
local imports, and CSV exports. Same-day reconstruction is labeled separately
from forecasting; the Vercel frontend has no forecasting controls or feed.

```bash
python scripts/export_modeling_bundle.py
npm run test:web
npm run build
python3 -m http.server 8510 --bind 127.0.0.1 --directory dist
```

Open `http://127.0.0.1:8510`. The checked-in example is synthetic and contains
Schwartz–Smith outputs. To include existing historical PCA and robustness
outputs, export them outside the public tree:

```bash
python scripts/export_modeling_bundle.py \
  --bundle results/wti --pca results/pca/wti \
  --robustness results/pca/wti/robustness \
  --relaxed-sensitivity results/pca/wti_coverage50/robustness \
  --output results/wti/modeling.json
```

Choose **Import results** and select that JSON. Omit optional PCA/robustness
arguments when those analyses are unavailable. Export reads saved tables; it
never fits a model. Imports remain in browser memory and disappear on reload.
The public build rejects empirical data in `web/data/`. Approved historical
outputs can instead be loaded from Vercel Blob using the separate
[export and manual upload workflow](docs/modeling_blob_deployment.md).

Filtered states, reconstruction residuals, and saved predictive likelihood are
available. Smoothed states and standardized innovations are not present in the
current saved bundles. Historical structural calibration may predate the model
audit; its provenance is displayed rather than silently recalibrating. Model
Comparison shows only saved PCA scaling/window comparisons on common support;
there is no matched-target Schwartz–Smith versus PCA leaderboard.

## Deploy to Vercel

Use **Other**, repository root **.**, Node.js **22 or later**, build command
`node scripts/build_web.mjs`, output **dist**, and no install command.
`vercel.json` supplies the build configuration. No Python server, feed tokens,
or backend credentials are needed. See [deployment instructions](DEPLOYMENT.md)
for validation and manual deployment. Nothing deploys from the export/build
commands above. The public example remains synthetic; historical JSON is
imported locally.

## Run the local Streamlit app

From the repository root, using Python 3.11 or later:

```bash
python -m pip install -e ".[dev]"
python -m streamlit run app.py
```

The existing local Conda environment is named `commodity-factors`. Activate it
with `conda activate commodity-factors` before running these commands.

A saved synthetic demo is included in `data/synthetic/demo/`. It works without
WRDS access. The dashboard discovers additional saved bundles under
`results/<dataset>/`; local WTI results appear when generated.

The five views cover market history and curve snapshots, forecast histories
and curves, forecast errors by maturity and horizon, and estimated factors
with hypothetical shocks, and the daily observation/forecast review. The sidebar controls dataset, evaluation period,
forecast horizon, rank, models, and evaluation window. CSV download exports the
forecast observations in the selected window.

## Secondary workflow: daily forecasts and automatic updates

Choose **Daily forecast** in the separate local Streamlit app. The Vercel
modeling frontend does not consume this legacy forecasting workflow.
For an observed session such as October 7, the view shows:

- October 7's observed settlement and current filtered states: one-factor x,
  short-term χ, and long-term ξ. Fitted parameters and their calibration date
  are also shown.
- Each model's forecast for October 7, made using data through the previous
  observed session, with its dollar and percentage error.
- Each model's forecast for the next session, such as October 8, using only
  data through October 7. The chart extends beyond the latest observation.

This view follows the **same contract** across dates, including rolls. Its
random walk uses that contract's previous quote, which can come from a
different maturity rank. If that contract was not previously in the panel,
the prior random-walk forecast is unavailable. The existing performance
benchmark continues to compare maturity ranks. Dollar error is observed
minus forecast; percentage error divides this difference by the observed
price. Both apps export the daily review and factor estimates to CSV.

Historical forecasts are labeled **Reconstructed**. The update path saves
next-session forecasts with their origin, target, contract, model, and UTC
recording timestamp. When the target's observation arrives, the view uses
those saved forecasts and labels them **Recorded**. Repeated updates preserve
their original values and timestamps. The CSV retains the recording time;
recording a replay today does not establish that it was issued historically.

The automatic update implementation uses Databento's final daily WTI settlement
curves, GitHub Actions, private Vercel Blob model history, and a public Blob
manifest consumed by the web workspace. It initializes a separate provider
history and keeps your WRDS prices private. The scheduler is disabled until
credentials, approved display rights, and spending limits are configured.
Follow [DEPLOYMENT.md](DEPLOYMENT.md) to activate it once.

The latest observation date is displayed explicitly. Historical replay uses
the next recorded session. Live updates use the NYMEX settlement calendar;
2026 closures are verified against CME's published schedule and future years
are labeled projected. Offline bundles without a calendar offer a clearly
labeled provisional weekday with an editable target date. Forecasts at or
after a contract's cutoff are unavailable. Automatic prices are daily
settlements, rather than streaming intraday quotes.

The provider-independent update entry point is ready for that job:

```bash
python scripts/update_daily.py \
  --bundle results/wti \
  --prices /path/to/completed_settlements.csv \
  --next-session 2026-10-12

# Legacy forecasting export; not accepted by the modeling frontend:
python scripts/export_web_bundle.py \
  --bundle results/wti \
  --output results/wti/browser.json
```

`--next-session` is an example; the adapter must supply the actual next
settlement session from the exchange/feed calendar. CSV or Parquet input needs
`date_`, `futcode`, `lasttrddate`, and `settlement`. Identifiers must map to the
same contracts in the saved history; a new provider's symbols need a mapping
to existing WRDS IDs or a separate newly calibrated bundle. Each new session
must contain at least the configured number of positive-price contracts;
the nearest eligible contracts are selected. Overlapping observations must
match saved history, and updates may not revise or insert past dates. The
job can catch up multiple complete sessions in order. It records a forecast
from the latest received session; missed runs do not create backdated issued
forecasts. No MLE runs during the daily update or on UI changes.

## Generate saved model analyses

For the existing local WTI panel:

```bash
python scripts/run_empirical.py
```

This estimates models on 2021–2023, keeps parameters fixed during 2024
validation and 2025 test, and saves `results/wti/`. The two-factor estimator uses
two deterministic starting points, fixes `lambda_chi=0`, and selects the best
converged training likelihood. A failed calibration stops before saving a
bundle. The scripts print optimizer status and a one-day test comparison.
The bundle also contains filtered factor estimates and same-day structural
reconstructions. Its `metrics.parquet` currently summarizes forecast errors;
the planned modeling evaluation will add reconstruction and diagnostic reports.

Regenerate the reproducible demo with training-only fits:

```bash
python scripts/run_synthetic.py --seed 42
```

Demo data cover business days in 2023–2025. Prices are simulated from the
two-factor model with independent state and measurement-noise random streams.
Monthly contract cutoffs are illustrative, not NYMEX expiry rules. The demo is
an example workflow, not evidence of historical market performance or a
parameter-recovery study.

Prepare licensed local data when needed:

```bash
python scripts/download_wrds.py
python scripts/prepare_panel.py
python scripts/run_empirical.py
```

The download script uses the local WRDS login and never stores credentials in
the application. Raw data, processed market prices, and empirical result
bundles are ignored by version control. Do not redistribute licensed WRDS
quotes or their result tables without permission. The synthetic bundle is
shareable.

## What the existing forecast comparison means

- **Random walk:** repeats the forecast origin's rank-wise log curve.
- **One factor:** the existing Brownian log-price state with physical drift and
  a linear maturity pricing adjustment, retained as the implemented research
  baseline. The project plan lists an OU alternative as future work.
- **Two factors:** a mean-reverting short-term deviation and a persistent
  long-term equilibrium, with exact correlated Gaussian transitions and the
  risk-neutral lognormal pricing adjustment.

All models target identical maturity-rank observations at horizons of 1, 5,
and 20 trading observations. Ranks change contract at rolls. This differs from
the earlier notebook's same-contract random-walk benchmark, so its RMSE values
are not directly interchangeable. Target maturities come from contract
metadata; price forecasts never use target or intermediate observations.

Each forecast originates at or after the final training date. This excludes
early validation targets for longer horizons when their origins would precede
calibration. Parameters stay fixed; filtered states update after each new
observation. Actual calendar gaps, including weekends, enter the transitions.

Headline RMSE/MAE use log prices. Dollar errors use exponentiated log forecasts,
which represent conditional medians rather than price-level conditional
means. Predictive likelihood is available for one-step state-space predictions.
Same-day fitted curves use the current day's observations and are displayed
separately from forecasting. Latent states are model-dependent estimates.

The performance comparisons are historical backtests. The daily view can
project beyond the latest saved observation, but does not supply live quotes.
The 2025 period was already inspected in the research notebook and
should not be described as an untouched holdout. Optimization convergence is
not proof of identification or forecast superiority. Parameter uncertainty,
multi-seed recovery, missing-maturity support, PCA, and an OU one-factor
alternative remain future work. The notebooks are retained as exploratory
history; use the scripts for reproducible dashboard results.

## Add a dataset

Supply a complete long-format panel with these columns:

| Column | Meaning |
| --- | --- |
| `date_` | Observation date |
| `futcode` | Contract identifier |
| `lasttrddate` | Contract's maturity cutoff |
| `settlement` | Strictly positive price |
| `tau` | Years to cutoff, using 365.25 days per year |
| `log_settlement` | Natural log of settlement |
| `maturity_rank` | Contiguous ranks 1 through N for every date |

Optional columns such as `dsmnem`, volume, and open interest are preserved.
The loader supports any complete number of ranks. New raw-data adapters can
produce this shared panel without changing the dashboard or the model APIs.

```bash
python scripts/run_empirical.py \
  --panel data/processed/another_panel.parquet \
  --dataset-id another_dataset \
  --dataset-label "Another futures dataset" \
  --source "Your data source" \
  --output results/another_dataset \
  --validation-start 2024-01-01 \
  --test-start 2025-01-01 \
  --horizons 1 5 20
```

The current UI assumes USD per barrel; other quote units will require updating
the bundle metadata and axis labels. Each bundle contains `panel.parquet`,
`predictions.parquet`, `reconstructions.parquet`, `states.parquet`,
`metrics.parquet`, and `metadata.json`. Predictions retain origin and target
dates, horizon, model, contract, maturity rank, observed and predicted log
prices, and errors. Metadata records the split, fitted parameters, data-quality
counts, assumptions, and optimizer diagnostics.

## Verify

```bash
python -m pytest -q
npm run test:web
```

Checks cover data validation and nullable numeric types, exact model math,
Kalman likelihood and covariance stability, aligned forecast targets,
training-only calibration, future-price perturbations, saved-bundle round
trips, and Streamlit controls across all five views. Daily checks cover causal
state updates, same-contract rolls, expiry and calendar boundaries, fixed
parameters and immutable recorded forecasts.
App tests use Streamlit's
[AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
and the saved synthetic bundle. JavaScript checks validate modeling imports,
all six sections, missing-output behavior, source-value preservation, and CSV
escaping. The optional historical fixture exercises every PCA/robustness view.
Node.js is required for frontend tests and the static production build. Passing
tests alone does not establish factor identification or empirical stability.

## PCA factor extraction

Run `python scripts/run_pca.py` in the `commodity-factors` environment to fit PCA
on the local historical panel. Separate results and plots are saved under
`results/pca/wti`. The default uses fixed-maturity log-price changes;
`--transformation log_prices` supports level-based curve compression.
See [PCA methodology and validation](docs/pca_methodology.md) for preprocessing,
coverage, economic shape checks, historical results, and limitations.

For rolling factor stability and robustness, run `python scripts/run_pca.py --robustness`.
This compares 126/252/504-date prior-only windows, covariance/correlation PCA,
K=1–3 held-out reconstructions, annual loading shapes, and interpolation/grid/quote-loss
sensitivities. Outputs are separate under `results/pca/wti/robustness`.
See [robustness methodology and empirical findings](docs/pca_robustness.md).
