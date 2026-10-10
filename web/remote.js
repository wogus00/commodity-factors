import {decodeBundle} from "./core.js";

export function publicBlobUrl(value) {
  let url;
  try { url = new URL(value); } catch { throw new Error("Configure a public Vercel Blob URL for daily data."); }
  if (url.protocol !== "https:" || url.username || url.password || url.port || url.search || url.hash
      || !/^[a-z0-9-]+\.public\.blob\.vercel-storage\.com$/.test(url.hostname)) {
    throw new Error("Daily data must use a public Vercel Blob URL.");
  }
  return url;
}

export function liveConfig(value) {
  if (!value || typeof value.live_manifest_url !== "string") throw new Error("Invalid daily-feed configuration.");
  if (value.live_manifest_url) {
    const url = publicBlobUrl(value.live_manifest_url);
    if (url.pathname !== "/commodity-factors/wti/latest.json") throw new Error("Use the daily updater's public manifest URL.");
  }
  return {url: value.live_manifest_url, refreshMs: Math.max(60, Math.min(3600, Number(value.refresh_seconds) || 300)) * 1000};
}

export async function fetchJson(url, {fetcher = fetch, maxBytes = 30_000_000} = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 25_000);
  try {
    const response = await fetcher(url, {cache: "no-store", credentials: "omit", redirect: "error", signal: controller.signal});
    if (!response.ok) throw new Error("Daily feed is unavailable. Last loaded results are retained.");
    if (Number(response.headers.get("content-length")) > maxBytes) throw new Error("Daily data exceeds the browser size limit.");
    const reader = response.body.getReader(), chunks = [];
    let size = 0;
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > maxBytes) { await reader.cancel(); throw new Error("Daily data exceeds the browser size limit."); }
      chunks.push(value);
    }
    const bytes = new Uint8Array(size); let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    return JSON.parse(new TextDecoder().decode(bytes));
  } finally { clearTimeout(timer); }
}

export async function fetchLive(config, {fetcher = fetch, previousUrl = ""} = {}) {
  const source = publicBlobUrl(config.url);
  const manifest = await fetchJson(source.href, {fetcher, maxBytes: 10_000});
  const data = publicBlobUrl(manifest.data_url);
  if (manifest.format !== "commodity-factors-live" || manifest.version !== 1 || manifest.dataset_id !== "databento-wti"
      || data.origin !== source.origin || !/^\/commodity-factors\/wti\/versions\/[a-f0-9]{32}\.json$/.test(data.pathname)
      || !Number.isFinite(Date.parse(manifest.updated_at))) throw new Error("Invalid published daily manifest.");
  if (data.href === previousUrl) return {manifest, bundle: null};
  const bundle = decodeBundle(await fetchJson(data.href, {fetcher}));
  if (bundle.metadata.dataset_id !== "databento-wti" || bundle.metadata.live_feed?.provider !== "Databento"
      || bundle.dates.at(-1) !== manifest.as_of || bundle.metadata.daily_forecast?.next_session !== manifest.next_session
      || bundle.metadata.daily_forecast?.last_update_utc !== manifest.updated_at) throw new Error("Daily manifest does not match its saved results.");
  bundle.remote = true;
  return {manifest, bundle};
}

export function feedStatus(bundle, error, now = new Date()) {
  if (error) return "Update unavailable · Last loaded results";
  const daily = bundle.metadata.daily_forecast;
  const parts = new Intl.DateTimeFormat("en-US", {timeZone: "America/Chicago", year: "numeric", month: "2-digit", day: "2-digit"}).formatToParts(now);
  const part = key => parts.find(p => p.type === key).value;
  const today = `${part("year")}-${part("month")}-${part("day")}`;
  const age = now - new Date(daily.last_update_utc);
  if (age > 96 * 3600_000 || daily.next_session < today) return "Feed behind schedule · Last available settlement";
  const delay = bundle.metadata.live_feed.publication_delay_hours || 0;
  return `Automatic daily settlements${delay ? ` · ${delay}h publication delay` : ""}`;
}
