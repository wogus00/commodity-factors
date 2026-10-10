# Commodity Factors

[Open the live demo](https://commodity-factors.vercel.app) · [Source code](https://github.com/wogus00/commodity-factors)

This showcase includes the model code, tests, and synthetic data. Research
notebooks are provided without saved outputs; licensed market data and local
results are excluded. The public repository starts from a clean snapshot and
does not include the private research commit history.

A chart-focused commodity research workspace with a TradingView-inspired dark
layout, built for Vercel. Explore futures curves and compare random-walk,
one-factor, and Schwartz–Smith two-factor forecasts. Python calibration runs
separately; both the browser frontend and the retained local Streamlit app
consume saved results without fitting on control changes.

## Run the web workspace

The browser frontend and generated synthetic demo are included in `web/`.
No JavaScript dependency installation or frontend build is needed.

```bash
python3 -m http.server 8510 --bind 127.0.0.1 --directory web
```

Open `http://127.0.0.1:8510`. The workspace has price, forecast, curve,
factor, and daily forecast views; contract selection; chart zoom and crosshairs; curve snapshots;
error charts; model overlays; factor shocks; and CSV export. The right panel
compares models on the same evaluation window across all maturity ranks.
Use keys 1–5 to change view, F to reset zoom, and L to toggle log scale.

Regenerate the browser demo after updating its Python bundle:

```bash
python scripts/export_web_bundle.py
```

Load local WTI results in the web workspace:

```bash
python scripts/export_web_bundle.py \
  --bundle results/wti \
  --output results/wti/browser.json
```

Choose **Import results** and select `results/wti/browser.json`. File parsing,
charts, metrics, and shocks run in the browser. The application does not upload
the imported file or persist its quotes to browser storage. Reloading the page
clears imported data. The exporter prevents empirical output inside `web/`.

## Deploy to Vercel

Deploy the repository with **Framework Preset: Other**, **Output Directory:
web**, and no build or install command. `vercel.json` supplies these settings.
The deployable tree contains HTML, CSS, JavaScript, a local Plotly library,
and synthetic data; it needs no Python server or backend credentials.

Using Vercel CLI, from the repository root on a machine with Node.js installed:

```bash
npx vercel
npx vercel --prod
```

The first command creates a preview deployment; the second promotes a build
to production. Sign in to Vercel and select your account/project when prompted.
`.vercelignore` restricts uploads to `web/` and `vercel.json`. WRDS data,
notebooks, local result bundles, and Python environments are excluded.

The public deployment opens the synthetic demo until a daily feed is activated.
[Deployment and daily-feed setup](DEPLOYMENT.md) describes the one-time account
setup, cost estimate, initialization, and scheduler activation. Private WTI
results can also be imported locally in the browser.

Streamlit requires a persistent WebSocket connection, while
[Vercel Functions cannot serve WebSockets](https://vercel.com/docs/limits).
The Vercel frontend therefore shares the saved analysis format with the Python
pipeline instead of trying to host Streamlit itself.

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

## Daily forecast review and automatic updates

Choose **Daily** in the web workspace or **Daily forecast** in Streamlit.
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

## Generate results

For the existing local WTI panel:

```bash
python scripts/run_empirical.py
```

This estimates models on 2021–2023, keeps parameters fixed during 2024
validation and 2025 test, and saves `results/wti/`. The two-factor estimator uses
two deterministic starting points, fixes `lambda_chi=0`, and selects the best
converged training likelihood. A failed calibration stops before saving a
bundle. The scripts print optimizer status and a one-day test comparison.

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

## What the comparison means

- **Random walk:** repeats the forecast origin's rank-wise log curve.
- **One factor:** the existing Brownian log-price state with physical drift and
  a linear maturity pricing adjustment. This is not the OU model proposed in
  `PROJECT_PLAN.md`; retaining it preserves the implemented research baseline.
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
parameters, immutable recorded forecasts, and browser/Python forecast parity.
App tests use Streamlit's
[AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
and the saved synthetic bundle. The JavaScript checks compare browser metrics
and pricing equations with saved Python results, test common evaluation
windows and import validation, and verify CSV escaping. Node.js is needed
only to run these checks, not to serve or deploy the frontend.
