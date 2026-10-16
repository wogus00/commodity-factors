# Deploy Latent Factor Modeling of Crude Oil Futures

The Vercel application is a static modeling dashboard. Python analysis runs
separately; frontend requests never run estimation, PCA, or data downloads.
No deployment has been performed as part of this change.

## Validate locally

Use Node.js 22 or later and the existing Python research environment:

```bash
python scripts/export_modeling_bundle.py
python -m pytest -q
npm run test:web
npm run build
python3 -m http.server 8510 --bind 127.0.0.1 --directory dist
```

The build validates the synthetic payload and copies the existing frontend
assets into `dist/`. Plotly is bundled locally; no JavaScript installation is
required. Open http://127.0.0.1:8510 to inspect the production build.

## Historical data from Blob

For approved public historical data, use the separate reproducible export and
manual upload workflow in [Modeling Blob deployment](docs/modeling_blob_deployment.md).
The frontend can load a public versioned dataset without putting it in GitHub.
The uploader defaults to a dry run and requires a hash-bound permission record
for publication. `MODELING_MANIFEST_URL` configures the public source at build
time; write tokens stay in the local upload process, never the frontend.

## Historical analysis import

Export existing results without rerunning any model:

```bash
python scripts/export_modeling_bundle.py \
  --bundle results/wti --pca results/pca/wti \
  --robustness results/pca/wti/robustness \
  --relaxed-sensitivity results/pca/wti_coverage50/robustness \
  --output results/wti/modeling.json
```

Choose **Import results** in the dashboard. These files remain in browser memory;
they are not uploaded or persisted. Reloading clears the import. The exporter
records source hashes and preserves saved analysis metadata. Optional analyses
can be omitted. The checked-in synthetic example has no PCA or robustness
outputs and displays that limitation explicitly.

The existing restriction against publishing empirical research data is retained.
Exports belong outside `web/`; the build accepts only the synthetic example in
`web/data/`. Do not copy historical JSON into the public deployment tree.

With the historical export available, run the additional integration tests:

```bash
MODELING_TEST_BUNDLE=results/wti/modeling.json npm run test:web
```

## Vercel configuration and manual deployment

Import this repository into Vercel, or update the existing project's overrides:

| Setting | Value |
| --- | --- |
| Framework preset | Other |
| Root directory | . |
| Node.js | 22 or later |
| Build command | node scripts/build_web.mjs |
| Output directory | dist |
| Install command | Empty |

These build/output settings are recorded in `vercel.json`. Ensure old dashboard
overrides pointing to output `web` are removed. `.vercelignore` includes only the
static frontend, build script, package manifest, and Vercel configuration.
The optional `MODELING_MANIFEST_URL` is a public URL. No Databento credentials
or Blob write tokens are needed by the deployed application.

After reviewing the changes, deploy a preview with your existing Vercel Git
integration or `npx vercel`. Review it before manually deploying production with
`npx vercel --prod`. Git integrations may deploy automatically on push according
to the project's existing settings; no push or deployment is done by this task.

## Modeling scope and unavailable outputs

Six sections cover Overview, Futures Term Structure, Schwartz–Smith Model,
PCA Factor Analysis, PCA Robustness, and Model Comparison. The last section
compares only saved PCA scaling and window experiments on common support.
Same-day curve reconstruction is not forecasting. PCA preprocessing, loadings,
rolling windows, sample coverage, and limitations are described in the views.

Current structural bundles contain filtered states, reconstruction residuals,
and predictive log-likelihood, but not smoothed states or standardized
innovations. Saved calibration can predate the model audit. No matched-target
Schwartz–Smith/PCA comparison is available; the UI does not manufacture one.

The separate Python forecasting implementations, Streamlit app, and daily
settlement workflow remain in the repository unchanged. The Vercel dashboard
no longer connects to their forecast feed. Existing scheduled jobs are not
modified or disabled by frontend deployment.
