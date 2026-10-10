import test from "node:test";
import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {liveConfig, fetchJson, fetchLive, feedStatus} from "../web/remote.js";

const origin = "https://test.public.blob.vercel-storage.com";
const url = `${origin}/commodity-factors/wti/latest.json`;
const dataUrl = `${origin}/commodity-factors/wti/versions/${"a".repeat(32)}.json`;
const demo = JSON.parse(await readFile(new URL("../web/data/demo.json", import.meta.url), "utf8"));
function fixture() {
  const payload = structuredClone(demo);
  payload.metadata.dataset_id = "databento-wti";
  payload.metadata.live_feed = {provider: "Databento", publication_delay_hours: 0};
  payload.metadata.daily_forecast = {next_session: "2026-01-02", last_update_utc: "2026-01-01T01:17:00Z", calendar_basis: "CME NYMEX settlement calendar"};
  const manifest = {format: "commodity-factors-live", version: 1, dataset_id: "databento-wti", data_url: dataUrl,
    as_of: "2025-12-31", updated_at: payload.metadata.daily_forecast.last_update_utc, next_session: "2026-01-02"};
  return {payload, manifest};
}

export function runRemoteChecks(register = test) {
register("daily config accepts only public Blob manifests and bounds refresh rate", () => {
  assert.equal(liveConfig({live_manifest_url: ""}).url, "");
  assert.equal(liveConfig({live_manifest_url: url, refresh_seconds: 5}).refreshMs, 60_000);
  for (const bad of [url.replace("public", "private"), "https://example.com/key", `${url}?token=secret`]) {
    assert.throws(() => liveConfig({live_manifest_url: bad}));
  }
});
register("live results must match the manifest and stay in the same public store", async () => {
  const {payload, manifest} = fixture(), calls = [];
  const fetcher = async address => { calls.push(address); return new Response(JSON.stringify(address === url ? manifest : payload)); };
  const result = await fetchLive(liveConfig({live_manifest_url: url}), {fetcher});
  assert.equal(result.bundle.remote, true);
  assert.equal(result.bundle.dates.at(-1), manifest.as_of);
  assert.equal(result.bundle.metadata.daily_forecast.calendar_basis, "CME NYMEX settlement calendar");
  calls.length = 0;
  assert.equal((await fetchLive({url}, {fetcher, previousUrl: dataUrl})).bundle, null);
  assert.deepEqual(calls, [url]);
  manifest.data_url = dataUrl.replace("test.public", "other.public");
  await assert.rejects(fetchLive({url}, {fetcher}), /manifest/);
  manifest.data_url = dataUrl; manifest.as_of = "2025-12-30";
  await assert.rejects(fetchLive({url}, {fetcher}), /does not match/);
});
register("feed failures and oversized responses are rejected without returning replacement data", async () => {
  await assert.rejects(fetchJson(url, {fetcher: async () => new Response("failed", {status: 503})}), /retained/);
  await assert.rejects(fetchJson(url, {maxBytes: 3, fetcher: async () => new Response("12345")}), /size limit/);
  const {payload} = fixture();
  const bundle = {metadata: payload.metadata};
  assert.match(feedStatus(bundle, "network", new Date("2026-01-01T12:00:00Z")), /Last loaded/);
  assert.match(feedStatus(bundle, null, new Date("2026-01-01T12:00:00Z")), /Automatic daily/);
  assert.match(feedStatus(bundle, null, new Date("2026-01-03T12:00:00Z")), /behind schedule/);
});
}
runRemoteChecks();
