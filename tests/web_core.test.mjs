import test from "node:test";
import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {decodeBundle, selectPredictions, summarize, scenarioLogPrices, nearestDate, observationOnOrBefore, dailySnapshot, dailyCsv, csv} from "../web/core.js";

const payload = JSON.parse(await readFile(new URL("../web/data/demo.json", import.meta.url), "utf8"));
const demo = decodeBundle(payload);

export function runCoreChecks(register = test) {
register("browser RMSE and MAE match Python metrics for both periods and every horizon", () => {
  for (const split of ["test", "validation"]) {
    for (const horizon of demo.metadata.horizons) {
      const results = summarize(selectPredictions(demo, {split, horizon}));
      for (const metric of results) {
        const expected = demo.metrics.find(r => r.split === split && r.horizon === horizon && r.model === metric.model && r.maturity_rank === 0);
        assert.ok(Math.abs(metric.rmse - expected.rmse_log) < 1e-10);
        assert.ok(Math.abs(metric.mae - expected.mae_log) < 1e-10);
        assert.equal(metric.n, expected.n);
        assert.equal(metric.n_dates, expected.n_dates);
      }
    }
  }
});

register("browser pricing equation reproduces saved Python reconstructions", () => {
  const parameters = demo.metadata.parameters.two_factor;
  for (const state of demo.states.filter((_, i) => i % 50 === 0)) {
    const curve = demo.byDate.get(state.date_);
    const actual = scenarioLogPrices(state, curve.map(r => r.tau), parameters);
    const fitted = demo.reconstructions.filter(r => r.date_ === state.date_ && r.model === "two_factor").sort((a, b) => a.maturity_rank - b.maturity_rank);
    actual.forEach((value, i) => assert.ok(Math.abs(value - fitted[i].fitted_log) < 1e-10));
  }
});

register("short shocks decay with maturity and long shocks shift every log price equally", () => {
  const state = demo.states.at(-1), tau = [0.1, 0.5, 1], p = demo.metadata.parameters.two_factor;
  const baseline = scenarioLogPrices(state, tau, p);
  const long = scenarioLogPrices(state, tau, p, 0, 0.2);
  const short = scenarioLogPrices(state, tau, p, 0.2, 0);
  baseline.forEach((value, i) => assert.ok(Math.abs(long[i] - value - 0.2) < 1e-12));
  assert.ok(short[0] - baseline[0] > short[2] - baseline[2]);
});

register("evaluation windows keep the same forecast targets for all models", () => {
  const rows = selectPredictions(demo, {split: "test", horizon: 5, start: "2025-03-01", end: "2025-04-30"});
  const results = summarize(rows);
  assert.equal(results.length, 3);
  assert.equal(new Set(results.map(r => r.n)).size, 1);
  assert.equal(new Set(results.map(r => r.n_dates)).size, 1);
  assert.ok(rows.every(r => r.date_ >= "2025-03-01" && r.date_ <= "2025-04-30" && r.origin_date < r.date_));
});

register("incomplete model coverage and invalid origins are rejected on import", () => {
  const incomplete = structuredClone(payload);
  incomplete.tables.predictions.data.pop();
  assert.throws(() => decodeBundle(incomplete), /common scoring sample/);
  const invalid = structuredClone(payload);
  const columns = invalid.tables.predictions.columns, row = invalid.tables.predictions.data[0];
  row[columns.indexOf("origin_date")] = row[columns.indexOf("date_")];
  assert.throws(() => decodeBundle(invalid), /origins/);
});

register("snapshot dates snap to available observations without inventing prices", () => {
  assert.equal(nearestDate(["2025-01-03", "2025-01-06"], "2025-01-05"), "2025-01-06");
});

register("CSV contains origin and target dates and safely escapes text identifiers", () => {
  const result = csv([{...demo.predictions[0], futcode: '=SUM(1,2)'}]);
  assert.ok(result.startsWith("date_,origin_date,horizon,"));
  assert.ok(result.includes('"\'=SUM(1,2)"'));
});

register("daily browser forecasts and states match the Python export", () => {
  const actual = dailySnapshot(demo), expected = payload.daily_latest;
  assert.ok(expected, "Regenerate the demo export to include the Python reference.");
  for (const key of ["as_of", "prior_origin", "target_date", "futcode", "calendar_basis", "expiry"]) assert.equal(actual[key], expected[key]);
  for (const key of Object.keys(expected.state)) assert.ok(Math.abs(actual.state[key] - expected.state[key]) < 1e-10);
  for (const [index, row] of actual.models.entries()) {
    for (const key of ["prior_log", "prior_price", "error_price", "error_percent", "next_log", "next_price"]) assert.ok(Math.abs(row[key] - expected.models[index][key]) < 1e-8, `${row.model}: ${key}`);
  }
});

register("daily prior forecasts use the previous session and match saved one-day predictions", () => {
  const snapshot = dailySnapshot(demo, "2025-10-07");
  assert.equal(snapshot.prior_origin, "2025-10-06");
  assert.equal(snapshot.target_date, "2025-10-08");
  assert.equal(snapshot.prior_recorded_at, null);
  for (const row of snapshot.models) {
    const saved = demo.predictions.find(r => r.date_ === snapshot.as_of && r.model === row.model && r.horizon === 1 && r.maturity_rank === 1);
    assert.ok(Math.abs(row.prior_log - saved.predicted_log) < 1e-10);
    assert.ok(Math.abs(row.error_price - (snapshot.observed_price - row.prior_price)) < 1e-12);
  }
});

register("future observations do not change a daily snapshot", () => {
  const changed = decodeBundle(structuredClone(payload));
  const before = dailySnapshot(changed, "2025-10-07");
  for (const row of changed.panel.filter(r => r.date_ > before.as_of)) { row.settlement *= 2; row.log_settlement = Math.log(row.settlement); }
  for (const row of changed.states.filter(r => r.date_ > before.as_of)) { row.chi += 1; row.xi += 1; row.one_factor_state += 1; }
  assert.deepEqual(dailySnapshot(changed, before.as_of), before);
});

register("daily forecasts hold contract identity across rolls and suppress expired targets", () => {
  const snapshot = dailySnapshot(demo, "2025-01-20");
  const old = demo.byDate.get("2025-01-17")[1];
  assert.equal(snapshot.futcode, String(old.futcode));
  assert.equal(snapshot.models[0].prior_log, old.log_settlement);
  assert.equal(dailySnapshot(demo, "2025-01-20", demo.ranks.at(-1)).models[0].prior_price, null);
  const expired = dailySnapshot(demo, "2025-01-20", 1, snapshot.expiry);
  assert.equal(expired.expired_before_target, true);
  assert.ok(expired.models.every(r => r.next_price === null));
  assert.throws(() => dailySnapshot(demo, "2023-06-01"), /calibration/);
  assert.throws(() => dailySnapshot(demo, "2025-10-07", 1, "2025-10-07"), /after/);
  assert.equal(observationOnOrBefore(demo.dates, "2025-10-12"), "2025-10-10");
});

register("recorded forecasts take precedence and retain issuance timestamps in CSV", () => {
  const snapshot = dailySnapshot(demo, "2025-10-07"), saved = structuredClone(payload);
  const records = snapshot.models.map(row => ({model: row.model, predicted_log: row.prior_log + 0.01,
    origin_date: snapshot.prior_origin, target_date: snapshot.as_of, futcode: snapshot.futcode,
    issued_at_utc: "2025-10-06T23:00:00+00:00"}));
  saved.metadata.daily_forecast = {issued_forecasts: records};
  const actual = dailySnapshot(decodeBundle(saved), snapshot.as_of);
  assert.equal(actual.prior_recorded_at, records[0].issued_at_utc);
  for (const [i, row] of actual.models.entries()) assert.equal(row.prior_log, records[i].predicted_log);
  assert.ok(dailyCsv(actual).includes("prior_recorded_at"));
  assert.ok(dailyCsv(actual).includes(records[0].issued_at_utc));
  saved.metadata.daily_forecast.issued_forecasts.push(records[0]);
  assert.throws(() => decodeBundle(saved), /recorded daily forecasts/);
});
}

runCoreChecks();
