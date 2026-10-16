# Latent Factor Modeling of Crude Oil Futures

[Live demo](https://commodity-factors.vercel.app) · [GitHub repository](https://github.com/wogus00/commodity-factors)

This project studies how a small number of hidden factors can describe crude oil futures curves. The public demo uses **synthetic data** so it can be shared without publishing licensed market data. Its results illustrate the tools and methods; they are not historical WTI findings.

## What’s included

- **Schwartz–Smith model:** estimates short-term and long-term factors and shows fitted futures curves.
- **PCA:** summarizes curve variation with principal components, loadings, and same-day reconstructions.
- **PCA robustness:** explores rolling windows, factor-loading stability, and reconstruction errors.
- **Forecast evaluation:** displays saved backtest results. These forecasts are separate from same-day curve reconstructions, and the demo does not provide live prices or forecasts.
- **Interactive dashboard:** compares term structures, factors, and model diagnostics.

The dashboard reads saved results; it does not fit models in the browser. The synthetic demo is included in `web/data/modeling-demo.json`.

## Run the dashboard locally

You need Node.js 22 or later.

```bash
npm run test:web
npm run build
python -m http.server 8510 --directory dist
```

Open [http://localhost:8510](http://localhost:8510). To preview the site with Vercel settings, follow [DEPLOYMENT.md](DEPLOYMENT.md).

## Run the Python project

Use Python 3.11 or later and install the project with its development tools:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

The local Streamlit app can be started with:

```bash
python -m streamlit run app.py
```

Synthetic data and reproducible run scripts are available in `data/synthetic/demo/` and `scripts/`. Local empirical analyses require authorized market data and are not included in the public demo. Keep raw or licensed data and local credentials out of Git.

## Reproducible outputs

The dashboard uses saved analysis outputs. For local research, the scripts can generate model results and export them for the dashboard. See [PCA methodology](docs/pca_methodology.md), [PCA robustness](docs/pca_robustness.md), and [Blob data deployment](docs/modeling_blob_deployment.md) for details.

Do not interpret synthetic demo results as evidence about real WTI markets. PCA components are statistical directions and do not automatically represent level, slope, or curvature; model fit and optimizer convergence do not establish factor identification or superiority.
