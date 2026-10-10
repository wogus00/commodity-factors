# Deploy Commodity Factors and activate daily WTI updates

The site is static on Vercel. GitHub Actions runs the Python models outside
Vercel and publishes new saved results to Vercel Blob. The browser checks the
public manifest every five minutes; updating prices does not rebuild the site.

The automatic feed uses **final actual daily CME NYMEX settlements** through
Databento, rather than intraday quotes. It downloads the six nearest outright
WTI contracts, initializes a separate 2021–present history, fits parameters
through 2023, and then updates filtered factors with fixed parameters. Your
existing WRDS data is not included in the public site or feed history.

## One-time account setup

1. Import this public GitHub repository into Vercel. Use **Other**, root
   directory **.**, output directory **web**, and empty build/install commands.
   The repository's `vercel.json` contains those settings. Deploy the synthetic
   demo first; the site works before a market feed is configured.
2. Create a [Databento account](https://databento.com/portal) and confirm with
   Databento the entitlement and public display permission for `GLBX.MDP3`
   WTI settlement history and daily updates. An API key alone does not establish
   redistribution permission. Review [pricing](https://databento.com/pricing)
   before buying anything. This repository does not purchase a subscription.
3. In Vercel **Storage**, create two Blob stores: one **private** for saved
   analysis history and one **public** for the browser results. Connect them to
   the project. Review the store's displayed usage charges; they are separate
   from the Databento data cost limit below.
4. In GitHub **Settings → Secrets and variables → Actions**, add these
   repository secrets. Enter their values directly into GitHub, not in chat,
   source files, or browser configuration:

   | Secret | Value |
   | --- | --- |
   | `DATABENTO_API_KEY` | Your Databento API key |
   | `BLOB_PRIVATE_READ_WRITE_TOKEN` | The private store's read/write token |
   | `BLOB_PUBLIC_READ_WRITE_TOKEN` | The public store's read/write token |

## Estimate, initialize, then enable

In GitHub **Actions → Daily WTI settlements → Run workflow**, select
**estimate** first. This calls free provider metadata and prints the estimated
2021–present download cost without requesting price data or writing storage.

After reviewing that estimate and your account's public display permission,
add the following GitHub repository **variables**:

| Variable | Value |
| --- | --- |
| `LIVE_DATA_DISPLAY_APPROVED` | `true` after you confirm display rights |
| `LIVE_INITIALIZE_MAX_COST_USD` | Your approved dollar limit for initialization |
| `LIVE_MAX_COST_USD` | Your approved dollar limit for each update run |
| `LIVE_PUBLICATION_DELAY_HOURS` | Required distribution delay in hours; defaults to `0` |

Each run estimates both the definition and statistics requests before making
either price-data request. It fails before requesting prices when the estimated
total exceeds its limit. Missing limits default to **$0**. These estimates do
not cap subscription, exchange licensing, GitHub, or Blob charges.

Run the workflow in **initialize** mode once. It downloads and calibrates the
provider history, saves a private model archive, and publishes the dashboard.
Initialization fails if saved history already exists, so it cannot reset your
issued forecast archive. Its final log line includes `public_manifest_url`.

Paste that **public URL** into `live_manifest_url` in [web/config.json](web/config.json).
Keep API keys and Blob tokens out of this file. Commit/push the configuration
change, which redeploys a Git-connected Vercel project. For an initial manual
Vercel upload, upload the updated site once or connect the Git repository.
The browser then selects **WTI crude oil · Daily** automatically and opens
the Daily view.

Run **update** once to confirm the account, provider, storage, and browser work
together. Finally set `ENABLE_LIVE_DATA=true` in GitHub repository variables.
The job runs at **01:17 and 05:17 UTC** daily, with serialized publication.
Those times are 9:17 p.m. and 1:17 a.m. during US Eastern daylight saving time,
or 8:17 p.m. and 12:17 a.m. in winter. The second run provides another attempt
after the provider has released final data. Weekend/holiday runs retain the
last settlement and existing forecast timestamps.

Enable email notifications for failed workflows in
[GitHub notification settings](https://github.com/settings/notifications).
The job uses GitHub's existing failure notifications; no email service or
messaging credentials are required.

## What continues automatically

- The job restores its private archive and requests data since the last saved
  observation, including a short receipt-date overlap. It checks for missing
  settlement sessions before publishing catch-up results.
- It records the next session's predictions with their actual UTC issuance
  time. Catch-up runs do not manufacture past issuance records. Predictions
  already issued keep their values and timestamps; factors update as new
  settlements arrive, and fitted parameters remain fixed.
- Both stores receive versioned objects before the `latest.json` pointers
  advance. A failed upload leaves the last public version available. Seven
  recent snapshots are retained in each store; all recorded forecasts remain
  inside the current private analysis archive.
- The workspace shows the actual observation date, source, update time, any
  configured publication delay, and an unavailable/behind-schedule status.
  A failed browser refresh retains the loaded analysis. A reload can retrieve
  the last public snapshot; imported private research files remain local.

CME's published **2026** settlement closure list is included in
[configs/settlement_calendar.json](configs/settlement_calendar.json).
Later years use regular holiday rules and are explicitly labeled **Projected**.
The configuration supports exceptional closures and verified annual lists.
Exchange calendar changes, licensing changes, account expiry, or failed model
calibration can require intervention. The system does not replace settlement
dates with generic weekdays or assume that every Globex trading day settles.

The feed publishes final observations without revising existing historical
prices. Provider corrections require an intentional new analysis version.
Current log-price models reject nonpositive settlements. Historical errors
before activation are labeled reconstructed; recorded forecasts accumulate
from activation onward. A required publication delay can make the latest
settlement too old to issue a still-future next-session prediction; the worker
reports that condition instead of claiming a past prediction was issued in time.

## Local commands

```bash
python -m pip install -e ".[dev,live]"
python scripts/live_daily.py --mode estimate
python scripts/live_daily.py --mode initialize
python scripts/live_daily.py --mode update
python -m pytest -q
npm run test:web
```

Live commands read credentials and approval/cost settings from environment
variables. They do not load `.env` automatically. The same CLI runs in GitHub
Actions. No feed or storage account is needed to run the tests or synthetic demo.
