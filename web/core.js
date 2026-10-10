// Shared numerical operations. This module has no browser or network dependencies.
export const MODELS = ["random_walk", "one_factor", "two_factor"];
export const LABELS = {random_walk: "Random walk", one_factor: "One factor", two_factor: "Two factors"};
export const COLORS = {observed: "#2fc6ad", random_walk: "#e4b05c", one_factor: "#b695f9", two_factor: "#5b8cff"};

const REQUIRED = {
  panel: ["date_", "futcode", "lasttrddate", "maturity_rank", "tau", "settlement", "log_settlement"],
  predictions: ["date_", "origin_date", "horizon", "split", "model", "maturity_rank", "tau", "observed_log", "predicted_log", "error_log"],
  reconstructions: ["date_", "model", "maturity_rank", "tau", "observed_log", "fitted_log"],
  states: ["date_", "split", "chi", "xi", "one_factor_state", "one_factor_predictive_ll", "two_factor_predictive_ll"],
  metrics: ["split", "horizon", "model", "maturity_rank", "rmse_log", "mae_log", "n", "n_dates"],
};
const NUMERIC = new Set(["maturity_rank", "tau", "settlement", "log_settlement", "horizon", "observed_log", "predicted_log", "error_log", "fitted_log", "chi", "xi", "one_factor_state", "rmse_log", "mae_log", "n", "n_dates", "one_factor_predictive_ll", "two_factor_predictive_ll"]);

export function decodeBundle(payload) {
  if (payload?.format !== "commodity-factors-web" || payload.version !== 1) throw new Error("Choose a JSON export created by export_web_bundle.py.");
  const meta = payload.metadata;
  if (!meta?.dataset_id || !meta.dataset_label || !meta.parameters?.two_factor || !meta.parameters?.one_factor || !Number.isFinite(meta.quality?.observations) || typeof meta.source !== "string" || typeof meta.price_unit !== "string" || !Array.isArray(meta.horizons) || !meta.horizons.length || meta.horizons.some(h => !Number.isInteger(h) || h < 1)) throw new Error("Result metadata is incomplete.");
  for (const date of [meta.train_end, meta.validation_start, meta.test_start]) dateOnly(date);
  const tables = {};
  for (const [name, required] of Object.entries(REQUIRED)) {
    const table = payload.tables?.[name];
    if (!Array.isArray(table?.columns) || !Array.isArray(table.data) || !table.data.length || !required.every(c => table.columns.includes(c))) throw new Error(`The ${name} table is missing required data.`);
    if (new Set(table.columns).size !== table.columns.length || table.columns.some(c => typeof c !== "string" || ["__proto__", "constructor", "prototype"].includes(c))) throw new Error("Invalid table columns.");
    tables[name] = table.data.map(values => {
      if (!Array.isArray(values) || values.length !== table.columns.length) throw new Error(`Invalid row in ${name}.`);
      const row = Object.create(null);
      table.columns.forEach((column, i) => {
        row[column] = (column === "date_" || column === "origin_date" || column === "lasttrddate") ? dateOnly(values[i]) : values[i];
        if (required.includes(column) && NUMERIC.has(column) && !Number.isFinite(row[column])) throw new Error(`Non-numeric ${column} in ${name}.`);
      });
      if (required.includes("model") && !MODELS.includes(row.model)) throw new Error("Unsupported model in result file.");
      return row;
    });
  }
  for (const row of tables.panel) {
    if (row.settlement <= 0 || row.tau <= 0 || !Number.isInteger(row.maturity_rank) || row.maturity_rank < 1) throw new Error("Prices and maturity ranks must be positive.");
    if (Math.abs(Math.log(row.settlement) - row.log_settlement) > 1e-7) throw new Error("Prices and log prices do not match.");
  }
  const coverage = new Map();
  for (const row of tables.predictions) {
    if (row.origin_date >= row.date_ || !Number.isInteger(row.horizon) || row.horizon < 1) throw new Error("Forecast origins must precede target dates.");
    if (Math.abs(row.observed_log - row.predicted_log - row.error_log) > 1e-7) throw new Error("Forecast errors do not match prices.");
    const key = `${row.horizon}|${row.date_}|${row.maturity_rank}`;
    const models = coverage.get(key) || new Set();
    if (models.has(row.model)) throw new Error("Duplicate forecast observation.");
    models.add(row.model); coverage.set(key, models);
  }
  if ([...coverage.values()].some(models => models.size !== MODELS.length)) throw new Error("Models must share a common scoring sample.");
  if (!["test", "validation"].every(split => meta.horizons.every(h => tables.predictions.some(r => r.split === split && r.horizon === h)))) throw new Error("Every horizon must cover validation and test.");
  tables.panel.sort((a, b) => a.date_.localeCompare(b.date_) || a.maturity_rank - b.maturity_rank);
  const dates = [...new Set(tables.panel.map(row => row.date_))];
  const ranks = [...new Set(tables.panel.map(row => row.maturity_rank))].sort((a, b) => a - b);
  if (ranks.some((rank, index) => rank !== index + 1)) throw new Error("Maturity ranks must be contiguous.");
  const byDate = new Map();
  for (const row of tables.panel) { const curve = byDate.get(row.date_) || []; curve.push(row); byDate.set(row.date_, curve); }
  if ([...byDate.values()].some(curve => curve.length !== ranks.length || new Set(curve.map(r => r.maturity_rank)).size !== ranks.length)) throw new Error("Every observation date must have a complete curve.");
  const p = meta.parameters.two_factor;
  if (!["kappa", "sigma_chi", "sigma_xi", "rho", "mu_xi", "lambda_chi", "lambda_xi"].every(k => Number.isFinite(p[k])) || p.kappa <= 0 || p.sigma_chi <= 0 || p.sigma_xi <= 0 || Math.abs(p.rho) >= 1) throw new Error("Invalid two-factor parameters.");
  if (!["mu", "sigma", "delta"].every(key => Number.isFinite(meta.parameters.one_factor[key])) || meta.parameters.one_factor.sigma <= 0) throw new Error("Invalid one-factor parameters.");
  const stateDates = new Map();
  for (const row of tables.states) stateDates.set(row.date_, (stateDates.get(row.date_) || 0) + 1);
  if (dates.some(date => stateDates.get(date) !== 1)) throw new Error("Every observation date needs one filtered state.");
  const issuedKeys = new Set();
  const issued = meta.daily_forecast?.issued_forecasts || [];
  if (!Array.isArray(issued)) throw new Error("Invalid recorded daily forecasts.");
  for (const row of issued) {
    const origin = dateOnly(row.origin_date), target = dateOnly(row.target_date);
    const key = `${origin}|${target}|${row.futcode}|${row.model}`;
    if (origin >= target || origin < meta.train_end || !MODELS.includes(row.model) || typeof row.futcode !== "string" || !Number.isFinite(row.predicted_log) || !Number.isFinite(Date.parse(row.issued_at_utc)) || issuedKeys.has(key)) throw new Error("Invalid recorded daily forecasts.");
    issuedKeys.add(key);
  }
  if (meta.daily_forecast?.next_session && dateOnly(meta.daily_forecast.next_session) <= dates.at(-1)) throw new Error("The next settlement session must follow the latest observation.");
  return {metadata: meta, ...tables, dates, ranks, byDate, statesByDate: new Map(tables.states.map(row => [row.date_, row]))};
}

export function dateOnly(value) {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}(T|$)/.test(value) || !Number.isFinite(Date.parse(value))) throw new Error("Invalid observation date.");
  if (new Date(`${value.slice(0, 10)}T12:00:00Z`).toISOString().slice(0, 10) !== value.slice(0, 10)) throw new Error("Invalid observation date.");
  return value.slice(0, 10);
}

export function selectPredictions(bundle, {split, horizon, start, end}) {
  return bundle.predictions.filter(r => r.split === split && r.horizon === Number(horizon) && (!start || r.date_ >= start) && (!end || r.date_ <= end));
}

export function summarize(rows) {
  const groups = new Map();
  for (const row of rows) {
    let group = groups.get(row.model);
    if (!group) { group = {model: row.model, squared: 0, absolute: 0, n: 0, dates: new Set()}; groups.set(row.model, group); }
    group.squared += row.error_log ** 2; group.absolute += Math.abs(row.error_log); group.n++; group.dates.add(row.date_);
  }
  return MODELS.filter(model => groups.has(model)).map(model => {
    const g = groups.get(model);
    return {model, rmse: Math.sqrt(g.squared / g.n), mae: g.absolute / g.n, n: g.n, n_dates: g.dates.size};
  });
}

export function pricingAdjustment(tau, p) {
  const decay = Math.exp(-p.kappa * tau);
  const variance = p.sigma_chi ** 2 / (2 * p.kappa) * (1 - decay ** 2)
    + p.sigma_xi ** 2 * tau + 2 * p.rho * p.sigma_chi * p.sigma_xi / p.kappa * (1 - decay);
  return -p.lambda_chi / p.kappa * (1 - decay) + (p.mu_xi - p.lambda_xi) * tau + 0.5 * variance;
}

export function scenarioLogPrices(state, maturities, p, shortShock = 0, longShock = 0) {
  return maturities.map(tau => Math.exp(-p.kappa * tau) * (state.chi + shortShock) + state.xi + longShock + pricingAdjustment(tau, p));
}

export function nearestDate(dates, date) {
  return dates.reduce((nearest, value) => Math.abs(Date.parse(value) - Date.parse(date)) < Math.abs(Date.parse(nearest) - Date.parse(date)) ? value : nearest, dates[0]);
}

export function observationOnOrBefore(dates, date) {
  return dates.filter(value => value <= date).at(-1) || dates[0];
}

export function forecastContract(filtered, quote, targetDate, parameters) {
  const gap = (Date.parse(targetDate) - Date.parse(quote.date_)) / 86400000 / 365.25;
  if (!(gap > 0)) throw new Error("The forecast target must be after the observation date.");
  const tau = (Date.parse(quote.lasttrddate) - Date.parse(targetDate)) / 86400000 / 365.25;
  if (tau <= 0) return null;
  const one = parameters.one_factor, two = parameters.two_factor;
  const advanced = {chi: filtered.chi * Math.exp(-two.kappa * gap), xi: filtered.xi + two.mu_xi * gap};
  return {random_walk: quote.log_settlement,
    one_factor: filtered.one_factor_state + one.mu * gap + (one.delta + 0.5 * one.sigma ** 2) * tau,
    two_factor: scenarioLogPrices(advanced, [tau], two)[0]};
}

export function dailySnapshot(bundle, asOf = bundle.dates.at(-1), rank = 1, targetDate = null) {
  const index = bundle.dates.indexOf(asOf);
  if (index < 0) throw new Error("Choose an observation date present in the dataset.");
  if (asOf < bundle.metadata.train_end) throw new Error("Daily forecasts start when calibration is complete.");
  const quote = bundle.byDate.get(asOf).find(row => row.maturity_rank === Number(rank));
  if (!quote) throw new Error("Choose an available maturity rank.");
  const filtered = bundle.statesByDate.get(asOf), parameters = bundle.metadata.parameters;
  const previousDate = bundle.dates[index - 1];
  let prior = null;
  if (previousDate && previousDate >= bundle.metadata.train_end) {
    const previous = bundle.byDate.get(previousDate).find(row => String(row.futcode) === String(quote.futcode));
    const oldState = bundle.statesByDate.get(previousDate);
    if (previous) prior = forecastContract(oldState, previous, asOf, parameters);
    else {
      const structural = forecastContract(oldState, {...quote, date_: previousDate}, asOf, parameters);
      prior = {...structural, random_walk: null};
    }
  }
  let target, basis;
  let recordedAt = null, priorOrigin = prior ? previousDate : null;
  const issued = (bundle.metadata.daily_forecast?.issued_forecasts || []).filter(row => row.target_date === asOf && row.futcode === String(quote.futcode) && row.origin_date < asOf);
  if (issued.length) {
    const origin = issued.map(row => row.origin_date).sort().at(-1), records = issued.filter(row => row.origin_date === origin);
    if (new Set(records.map(row => row.model)).size === MODELS.length) {
      prior = Object.fromEntries(records.map(row => [row.model, row.predicted_log]));
      priorOrigin = origin; recordedAt = records[0].issued_at_utc;
    }
  }
  if (targetDate) { target = dateOnly(targetDate); basis = "Selected target date"; }
  else if (bundle.dates[index + 1]) { target = bundle.dates[index + 1]; basis = "Next recorded session"; }
  else if (bundle.metadata.daily_forecast?.next_session) { target = dateOnly(bundle.metadata.daily_forecast.next_session); basis = bundle.metadata.daily_forecast.calendar_basis || "Feed settlement calendar"; }
  else {
    const next = new Date(`${asOf}T12:00:00Z`);
    do { next.setUTCDate(next.getUTCDate() + 1); } while ([0, 6].includes(next.getUTCDay()));
    target = next.toISOString().slice(0, 10); basis = "Provisional weekday; exchange holidays not checked";
  }
  const future = forecastContract(filtered, quote, target, parameters);
  return {as_of: asOf, prior_origin: priorOrigin, prior_recorded_at: recordedAt, target_date: target, calendar_basis: basis,
    futcode: String(quote.futcode), contract_label: quote.dsmnem || String(quote.futcode), expiry: quote.lasttrddate,
    rank: Number(rank), observed_price: quote.settlement, observed_log: quote.log_settlement,
    state: {one_factor_state: filtered.one_factor_state, chi: filtered.chi, xi: filtered.xi},
    expired_before_target: future === null,
    models: MODELS.map(model => {
      const priorLog = prior?.[model] ?? null, futureLog = future?.[model] ?? null;
      const priorPrice = priorLog === null ? null : Math.exp(priorLog);
      return {model, prior_log: priorLog, prior_price: priorPrice,
        error_price: priorPrice === null ? null : quote.settlement - priorPrice,
        error_percent: priorPrice === null ? null : (quote.settlement - priorPrice) / quote.settlement * 100,
        next_log: futureLog, next_price: futureLog === null ? null : Math.exp(futureLog)};
    })};
}

export function dailyCsv(snapshot) {
  const columns = ["observed_date", "futcode", "model", "observed_price", "prior_origin", "prior_recorded_at", "prior_price", "error_price", "error_percent", "next_target", "next_price", "one_factor_state", "chi", "xi"];
  const rows = snapshot.models.map(row => ({...row, observed_date: snapshot.as_of, futcode: snapshot.futcode,
    observed_price: snapshot.observed_price, prior_origin: snapshot.prior_origin, prior_recorded_at: snapshot.prior_recorded_at,
    next_target: snapshot.target_date, ...snapshot.state}));
  return csv(rows, columns);
}

export function csv(rows, columns = ["date_", "origin_date", "horizon", "split", "model", "futcode", "maturity_rank", "tau", "observed_log", "predicted_log", "error_log"]) {
  const escape = value => {
    let text = String(value ?? "");
    if (typeof value === "string" && /^[=+@-]/.test(text)) text = `'${text}`;
    return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  return [columns.join(","), ...rows.map(row => columns.map(c => escape(row[c])).join(","))].join("\n");
}
