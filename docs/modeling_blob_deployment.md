# Historical modeling data through Vercel Blob

## Scope and permission review

The static frontend remains in GitHub. Raw quotes, Parquet bundles, exported JSON,
and compressed versions remain local under ignored directories. The uploader
sends only the compact modeling payload and its manifest, never raw directories,
credentials, notebooks, forecast tables, or permission documents. Python model
implementations and the separate daily forecasting publisher are unchanged.
The modeling publisher uses `commodity-factors/modeling/wti/`, separate from the
legacy daily workflow's namespace.

Repository inspection found raw/processed WTI and results untracked. The tracked
`data/synthetic/demo/` files are intentional test fixtures. Ignore rules do not
untrack existing files or erase old commits. No local files were deleted and no
Git history was rewritten. Before committing, check `git status --short` and
`git ls-files data results exports credentials secrets`; review any new tracked
files rather than assuming `.gitignore` removes them.

The source is WRDS / LSEG Datastream. No public redistribution authorization was
found in this repository. Calling data "approved" is not evidence of the scope
of a vendor agreement. Confirm that the applicable agreement permits cloud
hosting AND anonymous download of historical settlements and derived outputs.
An approval record is your attestation of that review, not a license grant.
WRDS requires respecting vendor requirements and its terms when copying data:
[WRDS data policy](https://wrds-www.wharton.upenn.edu/pages/about/data-download-and-analysis-policy/).

**Public Blob** is appropriate only for data approved for unrestricted public
distribution. Everyone can download a public Blob URL, regardless of whether they
use this dashboard. This implementation supports that direct-browser path.
See [Vercel public storage](https://vercel.com/docs/vercel-blob/public-storage).

**Private Blob** is appropriate for restricted research, subject to cloud-hosting
permission. It requires an authenticated server endpoint or authorized signed
URLs and user access checks. An anonymous proxy would still redistribute the
same data. This app has no user authentication; this change deliberately does
not pretend a private store can safely serve its public dashboard. Keep the data
local until rights are confirmed, or implement authenticated delivery separately.
See [Vercel private storage](https://vercel.com/docs/vercel-blob/private-storage).

## Prepare locally (no network or uploads)

Use the existing `commodity-factors` environment from the repository root:

```bash
conda activate commodity-factors
python scripts/export_modeling_deployment.py \
  --bundle results/wti \
  --pca results/pca/wti \
  --robustness results/pca/wti/robustness \
  --relaxed-sensitivity results/pca/wti_coverage50/robustness \
  --output-dir exports/wti
python scripts/upload_modeling_blob.py --directory exports/wti
```

Omit `--relaxed-sensitivity` if that optional analysis is absent. Export reads
existing outputs only; it never reruns PCA, Kalman filtering, or MLE. It writes:

- `modeling.json`: the full frontend schema, also usable through local import.
- `<sha256>.json.gz`: deterministic lossless gzip, named by its byte checksum.
- `manifest.json`: version, checksum, byte counts, date coverage and table counts.

No date sampling or numerical rounding is applied. JSON numbers round-trip the
saved float values (browser IEEE-754 numbers). Nulls remain missing. Exported
structural loadings/residuals use the saved parameters/fits as before. The export
includes all rows of the existing frontend tables, including rolling loadings,
spectra, fit status, annual loadings, common-support errors and sensitivities.
It does not add unavailable diagnostics or fabricate comparison results.

The current WTI export contains 20 tables, 7,536 observed contract rows, 1,256
observation dates, and 86,580 rolling loading rows. Coverage is 2021-01-04 to
2025-12-31. JSON is **15,482,511 bytes**; gzip is **2,827,989 bytes**. Future
exports report their actual counts/sizes. A 30 MB compressed/decoded limit
protects the current browser loader; exports beyond it fail upload preflight
rather than silently truncating history.

## Costs to review before upload

Checked 2026-10-15 against [Vercel Blob pricing](https://vercel.com/docs/vercel-blob/usage-and-pricing):
example rates are $0.023/GB-month storage, $0.05/GB transfer, $0.40/million
simple operations and $5/million advanced operations. Region and plan affect
billing. CDN requests and cache-miss origin transfer can add charges; plan fees
and other project usage are separate. Verify the dashboard's current regional
rates and allowances before proceeding.

At those rates, one current compressed version costs approximately **$0.000065
per month** in storage; 1,000 complete downloads use about **2.83 GB** and cost
**$0.1414** in Blob transfer before allowances and other fees. These are estimates,
not spending caps. Public/private storage and operation rates are the same;
private server delivery can incur additional function and delivery costs.
The small manifest adds two requests per cold load (manifest plus payload).
Each publication writes a version and then a manifest; retries check/reuse the
version. Old versions are retained and storage grows with distinct exports.
No automatic deletion or retention cleanup is performed.

## Upload only after reviewing rights, contents and cost

1. In Vercel, manually create a **public** Blob store for approved modeling data.
   Obtain its write token. Install the already-pinned SDK if absent:

   ```bash
   python -m pip install 'vercel==0.11.6'
   ```

2. Create an approval record for the exact export. This command intentionally
   leaves approval false:

   ```bash
   python - <<'PY'
   import json
   from pathlib import Path
   manifest = json.loads(Path('exports/wti/manifest.json').read_text())
   record = {'sha256': manifest['sha256'],
             'public_redistribution_approved': False,
             'reviewed_by': '', 'permission_reference': ''}
   Path('exports/wti/publication.approval.json').write_text(json.dumps(record, indent=2)+'\n')
   PY
   ```

   Review `exports/wti/modeling.json`, the manifest and actual vendor permission.
   Edit the ignored approval file: name the reviewer, reference the agreement or
   written permission, and set approval true only when that review supports it.
   A changed export requires a matching new SHA256 approval.

3. In zsh, read the token without placing it in shell history, then publish:

   ```zsh
   read -s 'BLOB_READ_WRITE_TOKEN?Public Blob write token: '
   export BLOB_READ_WRITE_TOKEN
   python scripts/upload_modeling_blob.py \
     --directory exports/wti \
     --approval exports/wti/publication.approval.json \
     --execute
   unset BLOB_READ_WRITE_TOKEN
   ```

   Without `--execute`, the script performs local validation and prints size/cost
   estimates only; it does not import the SDK or make network requests. With it,
   the version uploads before `latest.json` advances. Failure leaves the prior
   manifest intact; rerun the same command to reuse an uploaded version. Do not
   run concurrent publishers for the same store/prefix. The last successful
   manifest update wins. No objects are deleted.

4. Copy the printed `MODELING_MANIFEST_URL` into the Vercel project's build
   environment variable with that name. It is a **public URL, not a secret**.
   Never set `BLOB_READ_WRITE_TOKEN` in frontend configuration. Only the upload
   process needs it; the static Vercel application needs no credentials.

   For local production verification:

   ```bash
   MODELING_MANIFEST_URL='https://YOUR-STORE.public.blob.vercel-storage.com/commodity-factors/modeling/wti/latest.json' npm run build
   python3 -m http.server 8510 --bind 127.0.0.1 --directory dist
   ```

   Alternatively, the nonsecret URL can be put in `web/modeling-source.json`.
   The build accepts only a public HTTPS Blob URL, no credentials/query tokens.
   Vercel settings stay Other / root `.` / Node 22+ / build
   `node scripts/build_web.mjs` / output `dist` / empty install command.
   Deploy a preview manually with `npx vercel` after review, then production
   with `npx vercel --prod` when satisfied. This task performs neither command.

## Loading, updates and caching

The page starts with the synthetic example, then loads the configured historical
manifest and payload. Failures remain visible, retain the current dataset and
allow **Refresh data** to retry. Local imports still work and are not replaced
by a remote response that completes after an import. Missing files, invalid
schemas, checksum/coverage mismatches, oversized responses and timeouts fail
explicitly. Modern browsers with Web Crypto and DecompressionStream are needed.

The immutable hash URL is cached for a year. The manifest has a 60-second Blob
cache policy, and the browser revalidates it on load/Refresh data. A newly
published pointer may take about a minute to become visible. No polling or
browser-storage database is used. Existing datasets remain in memory on errors;
there is no promised offline persistence across page reloads. Once the URL is
configured, subsequent data publications need no app rebuild. Manifest rollback
can be performed by deliberately publishing a previously reviewed export again.
Public caching means removal cannot recall copies already downloaded.

## Validation

```bash
python -m pytest -q
MODELING_TEST_BUNDLE=exports/wti/modeling.json npm run test:web
npm run build
```

Tests mock Blob writes and browser fetches; they do not upload or require tokens.
Production account billing, store access, CORS and real-network delivery must
still be verified with the first authorized preview. No private delivery or
new forecasting/wavelet feature is included.
