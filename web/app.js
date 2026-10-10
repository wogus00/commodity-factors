import {MODELS, LABELS, COLORS, decodeBundle, selectPredictions, summarize, scenarioLogPrices, nearestDate, observationOnOrBefore, dailySnapshot, forecastContract, dailyCsv, csv} from "./core.js";
import {liveConfig, fetchLive, feedStatus} from "./remote.js";

const $ = id => document.getElementById(id);
const state = {view: "market", lower: "curve", split: "test", horizon: 1, rank: 1, range: "1Y", log: false,
  models: new Set(MODELS), fitted: false, start: "", end: "", snapshot: "", short: 0, long: 0, dailyTarget: null};
const datasets = new Map();
let bundle, toastTimer, daily;
let liveError = null, liveUrl = "", liveConfigured = false;
const money = value => Number(value).toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2});
const dateLabel = value => new Date(`${value}T12:00:00Z`).toLocaleDateString("en-US", {month: "short", day: "numeric", year: "numeric", timeZone: "UTC"});
const text = (id, value) => { $(id).textContent = value; };
const visibleModels = () => MODELS.filter(model => state.models.has(model));
const level = log => state.log ? log : Math.exp(log);
const percent = value => `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
const yTitle = () => state.log ? "Log price" : bundle.metadata.price_unit;
const selected = () => state.view === "daily"
  ? bundle.predictions.filter(row => row.horizon === 1 && row.date_ <= state.snapshot)
  : selectPredictions(bundle, state);

function toast(message) {
  clearTimeout(toastTimer); text("toast", message); $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 6000);
}

function option(value, label) { const el = document.createElement("option"); el.value = value; el.textContent = label; return el; }
function cell(value, className = "") { const td = document.createElement("td"); td.textContent = value; td.className = className; return td; }
function dot(model) { const el = document.createElement("i"); el.className = `model-dot ${model}`; return el; }

function bindDateCommit(id, handler) {
  const input = $(id);
  // Native segmented date fields emit changes during typing. Commit after
  // leaving the field so rendering cannot interrupt month/day/year entry.
  input.addEventListener("change", event => { if (document.activeElement !== input) handler(event); });
  input.addEventListener("blur", handler);
  input.addEventListener("keydown", event => { if (event.key === "Enter") input.blur(); });
}

function plot(id, traces, overrides = {}) {
  const axis = {gridcolor: "#252a36", zerolinecolor: "#363d4e", tickfont: {size: 10, color: "#7b879d"},
    showline: false, showspikes: true, spikecolor: "#6f7d98", spikethickness: 1, spikedash: "dot", spikemode: "across", ticks: "outside", ticklen: 4, tickcolor: "#303849", automargin: true};
  const layout = {paper_bgcolor: "#131722", plot_bgcolor: "#131722", font: {family: "Arial, sans-serif", size: 11, color: "#aeb9cf"},
    margin: {l: 32, r: 66, t: 18, b: 28}, hovermode: "x unified", dragmode: "pan", showlegend: false,
    hoverlabel: {bgcolor: "#20293a", bordercolor: "#48546a", font: {size: 11, color: "#d9e2f5"}},
    xaxis: {...axis, showgrid: false}, yaxis: {...axis, side: "right"},
    modebar: {bgcolor: "transparent", color: "#7c8aa2", activecolor: "#7da5ff"}, ...overrides};
  const config = {responsive: true, displaylogo: false, scrollZoom: true, displayModeBar: "hover",
    modeBarButtons: [["toImage"], ["zoom2d", "pan2d"], ["zoomIn2d", "zoomOut2d", "resetScale2d"]],
    toImageButtonOptions: {filename: "commodity-factors", format: "png", scale: 2}};
  return Plotly.react($(id), traces, layout, config);
}

function line(x, y, name, color, extra = {}) {
  return {type: "scatter", mode: "lines", x, y, name, line: {color, width: 1.7},
    hovertemplate: "%{y:.4f}<extra>" + name + "</extra>", ...extra};
}

function sampleBounds() {
  const rows = selectPredictions(bundle, {split: state.split, horizon: state.horizon});
  const dates = [...new Set(rows.map(r => r.date_))].sort();
  if (!dates.length) throw new Error("No forecasts are available for that period and horizon.");
  return {first: dates[0], last: dates.at(-1)};
}

function resetWindow() {
  const {first, last} = sampleBounds(); state.start = first; state.end = last;
  for (const id of ["date-start", "date-end"]) { $(id).min = first; $(id).max = last; }
  $("date-start").value = state.start; $("date-end").value = state.end;
}

function loadDataset(key) {
  bundle = datasets.get(key);
  state.rank = bundle.ranks.includes(state.rank) ? state.rank : bundle.ranks[0];
  state.horizon = bundle.metadata.horizons.includes(state.horizon) ? state.horizon : bundle.metadata.horizons[0];
  state.snapshot = bundle.dates.at(-1); state.short = 0; state.long = 0; state.dailyTarget = null;
  $("rank-select").replaceChildren(...bundle.ranks.map(rank => option(rank, `M${rank}${rank === 1 ? " · Front" : ""}`)));
  $("rank-select").value = state.rank;
  $("horizon-select").replaceChildren(...bundle.metadata.horizons.map(h => option(h, `${h}D`)));
  $("horizon-select").value = state.horizon;
  $("split-select").options[0].textContent = `Test · ${bundle.metadata.test_start.slice(0, 4)}`;
  $("split-select").options[1].textContent = `Validation · ${bundle.metadata.validation_start.slice(0, 4)}`;
  $("snapshot-date").min = state.view === "daily" ? bundle.metadata.train_end : bundle.dates[0]; $("snapshot-date").max = bundle.dates.at(-1);
  $("snapshot-date").value = state.snapshot;
  resetWindow(); return render();
}

function setView(view) {
  state.view = view;
  document.querySelectorAll("[data-view]").forEach(button => { button.classList.toggle("active", button.dataset.view === view); button.setAttribute("aria-pressed", button.dataset.view === view); });
  $("scenario-section").hidden = view !== "factors";
  $("daily-panel").hidden = view !== "daily";
  for (const id of ["split-select", "horizon-select", "date-start", "date-end"]) $(id).disabled = view === "daily";
  if (bundle) {
    if (view === "daily" && state.snapshot < bundle.metadata.train_end) state.snapshot = bundle.dates.at(-1);
    $("snapshot-date").min = view === "daily" ? bundle.metadata.train_end : bundle.dates[0];
    $("snapshot-date").value = state.snapshot;
  }
  if (view === "daily") setLower("curve");
  if (view === "factors") setLower("curve");
  render();
}

function renderDaily() {
  daily = dailySnapshot(bundle, state.snapshot, state.rank, state.dailyTarget);
  $("daily-asof").min = bundle.metadata.train_end; $("daily-asof").max = bundle.dates.at(-1);
  if ($("daily-asof").value !== daily.as_of) $("daily-asof").value = daily.as_of;
  if ($("daily-target").value !== daily.target_date) $("daily-target").value = daily.target_date;
  $("daily-target").min = new Date(Date.parse(daily.as_of) + 86400000).toISOString().slice(0, 10);
  text("daily-observed-label", `Observed · ${dateLabel(daily.as_of)}`); text("daily-observed", money(daily.observed_price));
  text("daily-contract", `${daily.contract_label} · ${bundle.metadata.price_unit}`);
  text("daily-review-date", dateLabel(daily.as_of));
  text("daily-origin", daily.prior_origin ? `${daily.prior_recorded_at ? `Recorded ${new Date(daily.prior_recorded_at).toISOString().replace("T", " ").slice(0, 16)} UTC` : "Reconstructed forecast"} · Data through ${dateLabel(daily.prior_origin)}` : "No prior forecast after calibration");
  text("daily-next-date", dateLabel(daily.target_date)); text("daily-calendar", daily.calendar_basis);
  text("daily-prior-heading", `Forecast · ${daily.as_of.slice(5)}`); text("daily-next-heading", `Forecast · ${daily.target_date.slice(5)}`);
  const fragment = document.createDocumentFragment();
  for (const model of daily.models) {
    const tr = document.createElement("tr"), name = cell(""); name.append(dot(model.model), document.createTextNode(LABELS[model.model]));
    const error = model.error_price === null ? "—" : `${model.error_price >= 0 ? "+" : ""}${money(model.error_price)} / ${percent(model.error_percent)}`;
    tr.append(name, cell(model.prior_price === null ? "—" : money(model.prior_price)), cell(error), cell(model.next_price === null ? "—" : money(model.next_price)));
    fragment.append(tr);
  }
  $("daily-rows").replaceChildren(fragment);
  for (const [id, key] of [["daily-x", "one_factor_state"], ["daily-chi", "chi"], ["daily-xi", "xi"]]) text(id, daily.state[key].toFixed(5));
  const one = bundle.metadata.parameters.one_factor, two = bundle.metadata.parameters.two_factor;
  text("daily-parameters", `Parameters fitted through ${bundle.metadata.train_end} · One factor μ ${one.mu.toFixed(4)}, σ ${one.sigma.toFixed(4)}, δ ${one.delta.toFixed(4)} · Two factors κ ${two.kappa.toFixed(4)}, σχ ${two.sigma_chi.toFixed(4)}, σξ ${two.sigma_xi.toFixed(4)}, ρ ${two.rho.toFixed(4)}`);
  const synthetic = bundle.metadata.dataset_id === "synthetic";
  text("daily-freshness", bundle.remote ? `${feedStatus(bundle, liveError)} · Data through ${bundle.dates.at(-1)} · Checked ${new Date(bundle.metadata.daily_forecast.last_update_utc).toISOString().slice(0, 16).replace("T", " ")} UTC`
    : synthetic ? "Synthetic replay" : `Saved data through ${bundle.dates.at(-1)} · Automatic feed not connected`);
  text("daily-note", daily.expired_before_target ? "The selected contract expires before the target. Choose a later contract to make a forecast."
    : "Error = observed − prior forecast; % divides by the observed price. Forecasts hold this contract fixed across dates. Factors update with the observed settlement; parameters stay fixed. Price forecasts are conditional medians.");
}

function setLower(view) {
  state.lower = view;
  document.querySelectorAll("[data-lower]").forEach(button => button.classList.toggle("active", button.dataset.lower === view));
}

function rangeDates() {
  const end = bundle.dates.at(-1);
  if (state.range === "ALL") return [bundle.dates[0], end];
  const months = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12}[state.range];
  const date = new Date(`${end}T12:00:00Z`); date.setUTCMonth(date.getUTCMonth() - months);
  return [date.toISOString().slice(0, 10), end];
}

function headerAndContracts() {
  const curve = bundle.byDate.get(state.snapshot);
  const row = curve.find(r => r.maturity_rank === state.rank);
  const index = bundle.dates.indexOf(state.snapshot);
  const previous = bundle.byDate.get(bundle.dates[index - 1]) || [];
  const previousSame = previous.find(r => String(r.futcode) === String(row.futcode));
  text("instrument-name", `${bundle.metadata.dataset_label} · M${state.rank}`);
  text("last-price", state.log ? row.log_settlement.toFixed(4) : money(row.settlement));
  text("quote-date", `${dateLabel(state.snapshot)} · Settlement`);
  if (previousSame) {
    const delta = row.settlement - previousSame.settlement;
    text("last-change", `${delta >= 0 ? "+" : ""}${money(delta)} (${percent(delta / previousSame.settlement * 100)})`);
    $("last-change").className = `quote-change ${delta >= 0 ? "positive" : "negative"}`;
  } else { text("last-change", "No prior contract quote"); $("last-change").className = "quote-change"; }
  const synthetic = bundle.metadata.dataset_id === "synthetic";
  text("feed-mode", bundle.remote ? "DAILY SETTLEMENTS" : "HISTORICAL");
  text("data-badge", bundle.remote ? "DAILY WTI" : synthetic ? "SYNTHETIC DEMO" : "LOCAL RESULTS");
  $("data-badge").classList.toggle("local", !synthetic);
  text("source-note", bundle.remote ? "CME NYMEX final settlements via Databento. Prices update daily; the latest observed date is shown."
    : synthetic ? "Generated prices. This demonstrates the workflow, not historical WTI performance." : "Imported results stay in this browser. Rank histories change contract at rolls.");
  text("data-source", synthetic ? "Seeded simulation" : bundle.metadata.source);
  text("train-end", bundle.metadata.train_end);
  text("observation-count", bundle.metadata.quality.observations.toLocaleString("en-US"));
  text("status-text", bundle.remote ? feedStatus(bundle, liveError) : synthetic ? `Synthetic demo${liveConfigured && liveError ? " · Daily feed unavailable" : " · Saved model results"}` : "Local results · Browser only");
  text("status-period", `${bundle.dates[0]} — ${bundle.dates.at(-1)}`);
  const fragment = document.createDocumentFragment();
  for (const contract of curve) {
    const tr = document.createElement("tr"); tr.classList.toggle("selected", contract.maturity_rank === state.rank);
    const nameCell = document.createElement("td"), button = document.createElement("button");
    button.textContent = `M${contract.maturity_rank}${contract.maturity_rank === 1 ? " · Front" : ""}`;
    button.setAttribute("aria-label", `Select maturity rank ${contract.maturity_rank}`);
    const code = document.createElement("span"); code.className = "contract-code"; code.textContent = contract.dsmnem || String(contract.futcode);
    button.append(code); nameCell.append(button);
    button.addEventListener("click", () => { state.rank = contract.maturity_rank; $("rank-select").value = state.rank; render(); });
    const prev = previous.find(r => String(r.futcode) === String(contract.futcode));
    const change = prev ? (contract.settlement / prev.settlement - 1) * 100 : null;
    tr.append(nameCell, cell(money(contract.settlement)), cell(change === null ? "—" : percent(change), change === null ? "" : change >= 0 ? "positive" : "negative"));
    fragment.append(tr);
  }
  $("contract-rows").replaceChildren(fragment);
  const spread = curve.at(-1).settlement - curve[0].settlement;
  text("curve-shape", Math.abs(spread) < 0.005 ? "Flat" : spread > 0 ? "Contango" : "Backwardation");
  text("curve-spread", `${spread >= 0 ? "+" : ""}$${money(spread)}`);
}

function renderMetrics(rows) {
  const metrics = summarize(rows), rw = metrics.find(r => r.model === "random_walk")?.rmse;
  const best = Math.min(...metrics.map(r => r.rmse));
  const fragment = document.createDocumentFragment();
  for (const metric of metrics) {
    const tr = document.createElement("tr"); tr.classList.toggle("best", metric.rmse === best);
    const name = cell(""); name.append(dot(metric.model), document.createTextNode(LABELS[metric.model]));
    const improvement = rw > 0 ? (1 - metric.rmse / rw) * 100 : null;
    tr.append(name, cell(metric.rmse.toFixed(4)), cell(metric.mae.toFixed(4)), cell(improvement === null ? "—" : percent(improvement), improvement === null ? "" : improvement >= 0 ? "positive" : "negative"));
    fragment.append(tr);
  }
  $("metric-rows").replaceChildren(fragment);
  text("metric-horizon", state.view === "daily" ? "HISTORY · 1D" : `${state.split.toUpperCase()} · ${state.horizon}D`);
  $("date-start").value = state.view === "daily" ? rows[0]?.date_ || state.snapshot : state.start;
  $("date-end").value = state.view === "daily" ? state.snapshot : state.end;
  text("scored-dates", metrics[0]?.n_dates.toLocaleString("en-US") || "0");
}

function mainChart(rows) {
  $("chart-empty").hidden = true;
  $("fit-button").hidden = state.view !== "curve";
  let traces = [], overrides = {}, empty = "";
  const rankRows = rows.filter(r => r.maturity_rank === state.rank);
  if (state.view === "daily") {
    const history = bundle.panel.filter(row => String(row.futcode) === daily.futcode && row.date_ <= state.snapshot).slice(-60);
    traces.push(line(history.map(row => row.date_), history.map(row => level(row.log_settlement)), "Observed", COLORS.observed));
    for (const model of daily.models.filter(model => state.models.has(model.model))) {
      if (model.prior_log !== null) traces.push({type: "scatter", mode: "markers", x: [daily.as_of], y: [level(model.prior_log)], name: `${LABELS[model.model]} · Prior forecast`, marker: {color: COLORS[model.model], size: 9, symbol: "circle-open"}, hovertemplate: "%{y:.4f}<extra>Prior forecast · " + LABELS[model.model] + "</extra>"});
      if (model.next_log !== null) traces.push(line([daily.as_of, daily.target_date], [level(daily.observed_log), level(model.next_log)], `${LABELS[model.model]} · Next forecast`, COLORS[model.model], {mode: "lines+markers", line: {color: COLORS[model.model], width: 1.4, dash: "dot"}, marker: {size: 7, symbol: "diamond"}}));
    }
    text("chart-heading", "Observed history → next forecast"); text("chart-subtitle", `${daily.contract_label} · Only data through ${daily.as_of}`);
    overrides.shapes = [{type: "line", x0: daily.as_of, x1: daily.as_of, y0: 0, y1: 1, yref: "paper", line: {color: "#5c6a82", width: 1, dash: "dot"}}];
    text("chart-note", "Open circles: prior forecast · Diamonds: next forecast");
  } else if (state.view === "market") {
    const [start, end] = rangeDates();
    const history = bundle.panel.filter(r => r.maturity_rank === state.rank && r.date_ >= start && r.date_ <= end);
    text("chart-heading", "Settlement price"); text("chart-subtitle", "Daily · Rolling maturity rank");
    const y = history.map(r => level(r.log_settlement));
    traces = [line(history.map(r => r.date_), y, "Observed", COLORS.observed, {fill: "tozeroy", fillcolor: "rgba(47,198,173,0.04)"})];
    const pad = Math.max((Math.max(...y) - Math.min(...y)) * 0.12, state.log ? 0.03 : 1);
    overrides.yaxis = {side: "right", range: [Math.min(...y) - pad, Math.max(...y) + pad], gridcolor: "#252a36", tickfont: {size: 10, color: "#7b879d"}, automargin: true};
    const last = history.at(-1);
    overrides.annotations = [{x: 1, xref: "paper", y: level(last.log_settlement), yref: "y", text: state.log ? last.log_settlement.toFixed(3) : money(last.settlement), showarrow: false, xanchor: "left", bgcolor: "#248e80", font: {size: 10, color: "#fff"}, borderpad: 4}];
    text("chart-note", `${history.length} trading observations · ${yTitle()}`);
  } else if (state.view === "forecast") {
    text("chart-heading", `${state.horizon}-day-ahead forecast comparison`);
    text("chart-subtitle", "Predictions made before each target observation");
    const observed = rankRows.filter(r => r.model === "random_walk").sort((a, b) => a.date_.localeCompare(b.date_));
    traces.push(line(observed.map(r => r.date_), observed.map(r => level(r.observed_log)), "Observed", COLORS.observed));
    for (const model of visibleModels()) {
      const group = rankRows.filter(r => r.model === model).sort((a, b) => a.date_.localeCompare(b.date_));
      traces.push(line(group.map(r => r.date_), group.map(r => level(r.predicted_log)), LABELS[model], COLORS[model], {line: {color: COLORS[model], width: 1.25, dash: "dot"}}));
    }
    text("chart-note", `Rank M${state.rank} · Exponentiated log forecasts are price medians`);
    if (!observed.length) empty = "Choose an evaluation window that includes trading dates.";
  } else if (state.view === "curve") {
    const curve = bundle.byDate.get(state.snapshot);
    text("chart-heading", state.fitted ? "Same-day curve reconstruction" : `${state.horizon}-day-ahead futures curve`);
    traces.push(line(curve.map(r => r.tau * 365.25), curve.map(r => level(r.log_settlement)), "Observed", COLORS.observed, {mode: "lines+markers", marker: {size: 5}}));
    if (state.fitted) {
      for (const model of visibleModels().filter(m => m !== "random_walk")) {
        const group = bundle.reconstructions.filter(r => r.model === model && r.date_ === state.snapshot).sort((a, b) => a.maturity_rank - b.maturity_rank);
        traces.push(line(group.map(r => r.tau * 365.25), group.map(r => level(r.fitted_log)), LABELS[model], COLORS[model], {mode: "lines+markers"}));
      }
      text("chart-subtitle", "Uses this day's observations · Not a forecast");
    } else {
      const target = bundle.predictions.filter(r => r.date_ === state.snapshot && r.horizon === state.horizon);
      for (const model of visibleModels()) {
        const group = target.filter(r => r.model === model).sort((a, b) => a.maturity_rank - b.maturity_rank);
        if (group.length) traces.push(line(group.map(r => r.tau * 365.25), group.map(r => level(r.predicted_log)), LABELS[model], COLORS[model], {mode: "lines+markers", line: {color: COLORS[model], width: 1.4, dash: "dot"}}));
      }
      text("chart-subtitle", target.length ? `Origin ${target[0].origin_date}` : "No forecasts on this date · Observed curve only");
    }
    overrides.xaxis = {title: {text: "Days to last trading date", font: {size: 10}}, gridcolor: "#252a36", tickfont: {size: 10, color: "#7b879d"}, automargin: true};
    text("chart-note", `${dateLabel(state.snapshot)} · ${yTitle()}`);
  } else {
    const [start, end] = rangeDates(), rows = bundle.states.filter(r => r.date_ >= start && r.date_ <= end);
    text("chart-heading", "Estimated latent factors"); text("chart-subtitle", "Filtered χ and ξ · Model estimates");
    traces = [line(rows.map(r => r.date_), rows.map(r => r.chi), "Short-term χ", COLORS.one_factor), line(rows.map(r => r.date_), rows.map(r => r.xi), "Long-term ξ", COLORS.two_factor, {yaxis: "y2"})];
    overrides.yaxis = {domain: [0.55, 1], side: "right", gridcolor: "#252a36", tickfont: {size: 10, color: "#7b879d"}};
    overrides.yaxis2 = {domain: [0, 0.42], side: "right", gridcolor: "#252a36", tickfont: {size: 10, color: "#7b879d"}, anchor: "x"};
    overrides.annotations = [{x: 0.01, y: 0.98, xref: "paper", yref: "paper", text: "Short-term χ", showarrow: false, font: {color: COLORS.one_factor, size: 11}}, {x: 0.01, y: 0.39, xref: "paper", yref: "paper", text: "Long-term ξ", showarrow: false, font: {color: COLORS.two_factor, size: 11}}];
    text("chart-note", "Factors in log-price units · Not observed ground truth");
  }
  document.querySelector(".range-buttons").style.visibility = ["market", "factors"].includes(state.view) ? "visible" : "hidden";
  $("model-toggles").hidden = !["forecast", "curve", "daily"].includes(state.view);
  document.querySelector('[data-model="random_walk"]').hidden = state.view === "curve" && state.fitted;
  const rendered = plot("main-chart", traces, overrides);
  if (empty) { text("chart-empty", empty); $("chart-empty").hidden = false; }
  return rendered;
}

function lowerChart(rows) {
  let traces = [], overrides = {};
  if (state.lower === "curve") {
    const curve = bundle.byDate.get(state.snapshot);
    const x = curve.map(r => r.tau * 365.25);
    traces.push(line(x, curve.map(r => level(r.log_settlement)), "Observed", COLORS.observed, {mode: "lines+markers", marker: {size: 4}}));
    if (state.view === "daily") {
      const eligible = curve.filter(row => row.lasttrddate > daily.target_date);
      for (const model of visibleModels()) {
        const forecasts = eligible.map(row => forecastContract(bundle.statesByDate.get(state.snapshot), row, daily.target_date, bundle.metadata.parameters));
        traces.push(line(eligible.map(row => (Date.parse(row.lasttrddate) - Date.parse(daily.target_date)) / 86400000), forecasts.map(row => level(row[model])), `${LABELS[model]} · Next forecast`, COLORS[model], {mode: "lines+markers", line: {color: COLORS[model], width: 1.4, dash: "dot"}}));
      }
      text("lower-context", `Observed ${daily.as_of} · Forecast ${daily.target_date}`);
    } else if (state.view === "factors") {
      const filtered = bundle.states.find(r => r.date_ === state.snapshot);
      const p = bundle.metadata.parameters.two_factor;
      const baseline = scenarioLogPrices(filtered, curve.map(r => r.tau), p);
      const shocked = scenarioLogPrices(filtered, curve.map(r => r.tau), p, state.short, state.long);
      traces = [line(x, baseline.map(level), "Baseline", "#8a96ac"), line(x, shocked.map(level), "Hypothetical", COLORS.two_factor)];
      text("lower-context", "Hypothetical · Not a forecast");
    } else text("lower-context", dateLabel(state.snapshot));
    overrides.xaxis = {title: {text: "Days to last trading date", font: {size: 10}}, gridcolor: "#252a36", tickfont: {size: 10, color: "#7b879d"}, automargin: true};
  } else if (state.lower === "errors") {
    for (const model of visibleModels()) {
      const metrics = bundle.ranks.map(rank => summarize(rows.filter(r => r.maturity_rank === rank)).find(r => r.model === model));
      traces.push({type: "bar", x: bundle.ranks.map(r => `M${r}`), y: metrics.map(r => r?.rmse ?? null), name: LABELS[model], marker: {color: COLORS[model]}, hovertemplate: "%{y:.5f}<extra>" + LABELS[model] + "</extra>"});
    }
    overrides.barmode = "group"; text("lower-context", "Log RMSE · Common sample");
  } else {
    for (const model of visibleModels()) {
      const metrics = bundle.metadata.horizons.map(horizon => summarize(state.view === "daily"
        ? bundle.predictions.filter(row => row.horizon === horizon && row.date_ <= state.snapshot)
        : selectPredictions(bundle, {...state, horizon})).find(r => r.model === model));
      traces.push(line(bundle.metadata.horizons, metrics.map(r => r?.rmse ?? null), LABELS[model], COLORS[model], {mode: "lines+markers", marker: {size: 5}}));
    }
    overrides.xaxis = {title: {text: "Forecast horizon · Trading observations", font: {size: 10}}, tickvals: bundle.metadata.horizons, gridcolor: "#252a36", tickfont: {size: 10, color: "#7b879d"}, automargin: true};
    text("lower-context", state.view === "daily" ? `Log RMSE · Data through ${state.snapshot}` : "Log RMSE · Same window");
  }
  return plot("lower-chart", traces, {margin: {l: 30, r: 66, t: 12, b: 38}, ...overrides});
}

function render() {
  if (!bundle) return;
  const rows = selected();
  if (state.view === "daily") renderDaily();
  headerAndContracts(); renderMetrics(rows);
  const rendered = Promise.all([mainChart(rows), lowerChart(rows)]);
  $("scale-button").textContent = state.log ? "LOG" : "USD"; $("scale-button").setAttribute("aria-pressed", state.log);
  $("fit-button").setAttribute("aria-pressed", state.fitted);
  $("short-shock").value = state.short; $("long-shock").value = state.long;
  text("short-value", state.short.toFixed(2)); text("long-value", state.long.toFixed(2));
  $("export-button").disabled = state.view !== "daily" && !rows.length;
  return rendered;
}

function details() {
  const root = $("details-content"); root.replaceChildren();
  const heading = document.createElement("h3"); heading.textContent = bundle.metadata.dataset_label;
  const description = document.createElement("p"); description.textContent = `${bundle.metadata.source}. Training ends ${bundle.metadata.train_end}; parameters are fixed during validation and test.`;
  const ul = document.createElement("ul");
  for (const assumption of bundle.metadata.assumptions || []) { const li = document.createElement("li"); li.textContent = assumption; ul.append(li); }
  const paramsHeading = document.createElement("h3"); paramsHeading.textContent = "Fitted parameters";
  const table = document.createElement("table"), head = document.createElement("thead"), tr = document.createElement("tr");
  for (const value of ["Parameter", "One factor", "Two factors"]) { const th = document.createElement("th"); th.textContent = value; tr.append(th); } head.append(tr); table.append(head);
  const tbody = document.createElement("tbody"), parameters = bundle.metadata.parameters;
  const names = [...new Set([...Object.keys(parameters.one_factor), ...Object.keys(parameters.two_factor)])];
  for (const name of names) { const row = document.createElement("tr"); row.append(cell(name), cell(parameters.one_factor[name]?.toFixed(5) ?? "—"), cell(parameters.two_factor[name]?.toFixed(5) ?? "—")); tbody.append(row); } table.append(tbody);
  const llHeading = document.createElement("h3"); llHeading.textContent = "One-step predictive likelihood";
  const holdout = bundle.states.filter(r => r.split === state.split);
  const likelihood = document.createElement("p"); likelihood.textContent = ["one_factor", "two_factor"].map(model => `${LABELS[model]}: ${holdout.reduce((sum, r) => sum + r[`${model}_predictive_ll`], 0).toFixed(2)}`).join(" · ") + `. Full ${state.split} period, independent of the selected horizon and evaluation window.`;
  root.append(heading, description, ul, paramsHeading, table, llHeading, likelihood);
  $("details-dialog").showModal();
}

document.querySelectorAll("[data-view]").forEach(button => button.addEventListener("click", () => setView(button.dataset.view)));
document.querySelectorAll("[data-lower]").forEach(button => button.addEventListener("click", () => {
  setLower(button.dataset.lower); lowerChart(selected());
}));
document.querySelectorAll("[data-range]").forEach(button => button.addEventListener("click", () => {
  state.range = button.dataset.range; document.querySelectorAll("[data-range]").forEach(b => b.classList.toggle("active", b === button)); mainChart(selected());
}));
document.querySelectorAll("[data-model]").forEach(button => button.addEventListener("click", () => {
  const model = button.dataset.model;
  state.models.has(model) ? state.models.delete(model) : state.models.add(model);
  button.setAttribute("aria-pressed", state.models.has(model)); render();
}));
$("dataset-select").addEventListener("change", event => loadDataset(event.target.value));
$("split-select").addEventListener("change", event => { state.split = event.target.value; resetWindow(); render(); });
$("horizon-select").addEventListener("change", event => { state.horizon = Number(event.target.value); resetWindow(); render(); });
$("rank-select").addEventListener("change", event => { state.rank = Number(event.target.value); render(); });
$("scale-button").addEventListener("click", () => { state.log = !state.log; render(); });
$("fit-button").addEventListener("click", () => { state.fitted = !state.fitted; render(); });
bindDateCommit("snapshot-date", event => {
  if (!event.target.value) return;
  const dates = state.view === "daily" ? bundle.dates.filter(date => date >= bundle.metadata.train_end) : bundle.dates;
  const date = (state.view === "daily" ? observationOnOrBefore : nearestDate)(dates, event.target.value);
  if (date === state.snapshot && event.target.value === date) return;
  state.snapshot = date; state.dailyTarget = null;
  if (event.target.value !== state.snapshot) event.target.value = state.snapshot;
  render();
});
bindDateCommit("daily-asof", event => {
  if (!event.target.value) return;
  const dates = bundle.dates.filter(date => date >= bundle.metadata.train_end);
  const date = observationOnOrBefore(dates, event.target.value);
  if (date === state.snapshot && event.target.value === date) return;
  state.snapshot = date; state.dailyTarget = null;
  $("snapshot-date").value = state.snapshot; render();
});
$("daily-latest").addEventListener("click", () => { state.snapshot = bundle.dates.at(-1); state.dailyTarget = null; $("snapshot-date").value = state.snapshot; render(); });
bindDateCommit("daily-target", event => {
  if (!event.target.value || event.target.value <= state.snapshot) { toast("Choose a target after the observed date."); event.target.value = daily.target_date; return; }
  if (event.target.value === daily.target_date) return;
  state.dailyTarget = event.target.value; render();
});
for (const id of ["date-start", "date-end"]) $(id).addEventListener("change", () => {
  const start = $("date-start").value, end = $("date-end").value;
  const bounds = sampleBounds();
  if (!start || !end || start > end || start < bounds.first || end > bounds.last) {
    toast("Choose an ordered evaluation window within the selected period."); $("date-start").value = state.start; $("date-end").value = state.end; return;
  }
  state.start = start; state.end = end; render();
});
for (const [id, key] of [["short-shock", "short"], ["long-shock", "long"]]) $(id).addEventListener("input", event => { state[key] = Number(event.target.value); setLower("curve"); render(); });
$("reset-shocks").addEventListener("click", () => { state.short = 0; state.long = 0; setLower("curve"); render(); });
$("reset-chart").addEventListener("click", () => { Plotly.relayout($("main-chart"), {"xaxis.autorange": true, "yaxis.autorange": true}); });
$("help-button").addEventListener("click", () => bundle && details()); $("details-button").addEventListener("click", () => bundle && details());
$("close-dialog").addEventListener("click", () => $("details-dialog").close());
$("import-button").addEventListener("click", () => $("import-file").click());
$("import-file").addEventListener("change", async event => {
  const file = event.target.files[0]; if (!file) return;
  try {
    if (file.size > 30_000_000) throw new Error("This result file exceeds the 30 MB browser import limit.");
    const imported = decodeBundle(JSON.parse(await file.text()));
    const key = `local-${datasets.size}`; datasets.set(key, imported);
    $("dataset-select").append(option(key, `${imported.metadata.dataset_label} · Local`)); $("dataset-select").value = key; loadDataset(key);
    toast("Results loaded locally. The file stays in your browser.");
  } catch (error) { toast(error.message); }
  event.target.value = "";
});
$("export-button").addEventListener("click", () => {
  const url = URL.createObjectURL(new Blob([state.view === "daily" ? dailyCsv(daily) : csv(selected())], {type: "text/csv;charset=utf-8"}));
  const a = document.createElement("a"); a.href = url; a.download = state.view === "daily" ? `${bundle.metadata.dataset_id}_daily_${daily.as_of}.csv` : `${bundle.metadata.dataset_id}_${state.split}_${state.horizon}d.csv`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
document.addEventListener("keydown", event => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName) || event.metaKey || event.ctrlKey || event.altKey || $("details-dialog").open) return;
  const views = {"1": "market", "2": "forecast", "3": "curve", "4": "factors", "5": "daily"};
  if (views[event.key]) setView(views[event.key]);
  if (event.key.toLowerCase() === "f") $("reset-chart").click();
  if (event.key.toLowerCase() === "l") $("scale-button").click();
});

async function initialize() {
  try {
    const response = await fetch("data/demo.json"); if (!response.ok) throw new Error("The saved demo could not be loaded.");
    const demo = decodeBundle(await response.json()); datasets.set("demo", demo);
    $("dataset-select").replaceChildren(option("demo", demo.metadata.dataset_label)); await loadDataset("demo");
    $("main-chart").on("plotly_click", event => {
      const value = event.points[0]?.x;
      if (typeof value === "string" && /^\d{4}-\d{2}-\d{2}/.test(value)) {
        if (state.view === "daily") return;
        state.snapshot = nearestDate(bundle.dates, value.slice(0, 10)); $("snapshot-date").value = state.snapshot; headerAndContracts(); lowerChart(selected());
      }
    });
  } catch (error) {
    text("status-text", "Data unavailable"); toast(error.message);
    text("chart-empty", "Saved results could not be loaded. Try importing a result bundle."); $("chart-empty").hidden = false;
  }
  await initializeLive();
}

async function initializeLive() {
  let config;
  try {
    const response = await fetch("config.json", {cache: "no-store"});
    if (!response.ok) throw new Error("Daily-feed configuration could not be loaded.");
    config = liveConfig(await response.json());
    if (!config.url) return;
    liveConfigured = true;
  } catch (error) { toast(error.message); return; }
  let inFlight = false;
  async function refresh() {
    if (inFlight) return;
    inFlight = true;
    try {
      const result = await fetchLive(config, {previousUrl: liveUrl});
      liveError = null;
      if (result.bundle) {
        const old = datasets.get("live"), snapshot = state.snapshot;
        datasets.set("live", result.bundle); liveUrl = result.manifest.data_url;
        if (!old) {
          $("dataset-select").append(option("live", "WTI crude oil · Daily"));
          if ($("dataset-select").value === "demo" || !bundle) {
            $("dataset-select").value = "live"; await loadDataset("live"); setView("daily");
          }
        } else if ($("dataset-select").value === "live") {
          bundle = result.bundle;
          state.snapshot = snapshot === old.dates.at(-1) ? bundle.dates.at(-1) : snapshot;
          $("snapshot-date").max = bundle.dates.at(-1); $("snapshot-date").value = state.snapshot;
          const bounds = sampleBounds();
          for (const id of ["date-start", "date-end"]) { $(id).min = bounds.first; $(id).max = bounds.last; }
          render();
        }
      } else if (bundle) render();
    } catch (error) {
      if (!liveError) toast(error.message);
      liveError = error.message;
      if (bundle) render();
    } finally { inFlight = false; }
  }
  await refresh();
  setInterval(refresh, config.refreshMs);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
}
initialize();
