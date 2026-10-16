// Browser-only decoding and selection. All fitted quantities come from Python exports.
export const COLORS = {observed:'#2fc6ad', short:'#b695f9', long:'#5b8cff', PC1:'#5b8cff', PC2:'#e4b05c', PC3:'#b695f9'};
export const VIEWS = {
  overview: {label:'Overview', charts:['summary']},
  structure: {label:'Futures Term Structure', charts:['curve', 'history']},
  schwartz: {label:'Schwartz–Smith Model', charts:['factors', 'curve', 'loadings', 'residuals', 'likelihood']},
  forecast: {label:'Forecast Evaluation', charts:['predictions', 'model-errors', 'maturity-errors', 'horizons']},
  pca: {label:'PCA Factor Analysis', charts:['variance', 'loadings', 'scores', 'curve', 'errors']},
  robustness: {label:'PCA Robustness', charts:['variance', 'loadings', 'similarity', 'angles', 'annual', 'components', 'maturity', 'sensitivity']},
  comparison: {label:'Model Comparison', charts:['scaling', 'windows']},
};
export function dateOnly(value) {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}(T.*)?$/.test(value)) throw new Error('Invalid observation date');
  const day = value.slice(0,10), date = new Date(`${day}T00:00:00Z`);
  if (!Number.isFinite(+date) || date.toISOString().slice(0,10) !== day) throw new Error('Invalid observation date');
  return day;
}
const REQUIRED = {
  panel:['date_', 'maturity_rank', 'tau', 'settlement', 'log_settlement', 'futcode', 'lasttrddate'],
  states:['date_', 'chi', 'xi', 'two_factor_predictive_ll'],
  ss_reconstructions:['date_', 'maturity_rank', 'tau', 'observed_log', 'fitted_log', 'residual_log'],
  ss_loadings:['date_', 'maturity_rank', 'tau', 'short_loading', 'long_loading'],
  pca_explained_variance:['component','ratio','cumulative_ratio'], pca_loadings:['tau','PC1','PC2','PC3'],
  pca_scores:['date_','PC1','PC2','PC3'], pca_reconstructions:['date_','tau','observed_log','fitted_log'],
  pca_metrics:['split','tau','rmse_log','n_dates'], pca_interpretations:['component','level','slope','curvature'],
  pca_coverage:['date_','feature_complete'],
  rolling_spectrum:['date_','window','method','track','source_rank','variance_ratio','anchor_cosine','previous_cosine'],
  rolling_fits:['date_','window','method','train_end','coverage','status','scored','anchor_subspace_angle_deg'],
  rolling_loadings:['date_','window','method','track','tau','loading'],
  rolling_metrics:['window','method','k','period','tau','rmse_log','n_dates'],
  subperiod_loadings:['period','method','track','tau','loading'],
  common_window_metrics:['window','method','k','period','tau','rmse_log','n_dates'],
  sensitivity_metrics:['scenario','method','k','period','tau','rmse_log','n_dates'],
  sensitivity_fits:['scenario','window','method','coverage','status','scored'],
  relaxed_sensitivity_metrics:['scenario','method','k','period','tau','rmse_log','n_dates'],
  forecast_predictions:['date_','origin_date','horizon','split','model','maturity_rank','tau','observed_log','predicted_log','error_log'],
  forecast_metrics:['split','horizon','model','maturity_rank','rmse_log','mae_log','rmse_price','mae_price','n','n_dates'],
};
const TEXT = new Set(['futcode','split','component','method','period','scenario','status','model']);
const DATES = new Set(['date_','origin_date','lasttrddate','train_end','window_start']);
const BOOLEAN = new Set(['scored','curve_complete','feature_complete']);
export function decodeBundle(payload) {
  if (payload?.format !== 'crude-oil-factor-modeling' || payload.version !== 1) throw new Error('Import a modeling export from export_modeling_bundle.py, not a legacy forecast bundle.');
  const meta = payload.metadata;
  if (!meta || !['dataset_id','dataset_label','source','price_unit','train_end'].every(k => typeof meta[k] === 'string') || meta.state_estimate !== 'filtered') throw new Error('Missing modeling metadata or unsupported state estimate');
  dateOnly(meta.train_end);
  if (!['kappa','sigma_chi','sigma_xi','rho','mu_xi','lambda_chi','lambda_xi'].every(k => Number.isFinite(meta.parameters?.[k])) || meta.parameters.kappa <= 0) throw new Error('Invalid structural parameters');
  const tables = {};
  const mandatory = ['panel','states','ss_reconstructions','ss_loadings'];
  if (meta.pca) mandatory.push(...Object.keys(REQUIRED).filter(k => k.startsWith('pca_')));
  if (meta.robustness) mandatory.push(...['rolling_spectrum','rolling_fits','rolling_loadings','rolling_metrics','subperiod_loadings','common_window_metrics','sensitivity_metrics','sensitivity_fits'].filter(k => !k.startsWith('sensitivity') || meta.robustness.scenarios && Object.keys(meta.robustness.scenarios).length));
  if (meta.forecast) mandatory.push('forecast_predictions','forecast_metrics');
  if (mandatory.some(k => !payload.tables?.[k])) throw new Error('Incomplete modeling tables');
  for (const [name, table] of Object.entries(payload.tables || {})) {
    if (!REQUIRED[name] || !Array.isArray(table.columns) || !Array.isArray(table.data) || !REQUIRED[name].every(k => table.columns.includes(k)) || new Set(table.columns).size !== table.columns.length || table.columns.some(k => typeof k !== 'string' || ['__proto__','constructor','prototype'].includes(k))) throw new Error(`Invalid ${name} table`);
    tables[name] = table.data.map(values => {
      if (!Array.isArray(values) || values.length !== table.columns.length) throw new Error(`Invalid ${name} row`);
      const row = Object.create(null);
      table.columns.forEach((key,i) => {
        const value = values[i];
        if (DATES.has(key)) row[key] = dateOnly(value);
        else if (TEXT.has(key)) { if (!['string','number'].includes(typeof value)) throw new Error(`Invalid ${key}`); row[key] = String(value); }
        else if (BOOLEAN.has(key)) { if (typeof value !== 'boolean') throw new Error(`Invalid ${key}`); row[key] = value; }
        else { if (value !== null && !Number.isFinite(value)) throw new Error(`Non-numeric ${key}`); row[key] = value; }
      });
      if (row.train_end && row.date_ && row.train_end >= row.date_) throw new Error('Rolling fits must precede held-out observations');
      return row;
    });
  }
  if (!tables.panel.length) throw new Error('No observed curves');
  tables.panel.sort((a,b) => a.date_.localeCompare(b.date_) || a.maturity_rank-b.maturity_rank);
  const byDate = new Map(), keys = new Set();
  for (const row of tables.panel) {
    const key = `${row.date_}|${row.maturity_rank}`;
    if (keys.has(key) || !(row.settlement > 0) || !(row.tau > 0) || !Number.isInteger(row.maturity_rank) || row.maturity_rank < 1 || !Number.isFinite(row.log_settlement) || Math.abs(Math.log(row.settlement)-row.log_settlement) > 1e-7) throw new Error('Invalid or duplicate observed curve');
    keys.add(key); const group = byDate.get(row.date_) || []; group.push(row); byDate.set(row.date_, group);
  }
  const dates = [...byDate.keys()], ranks = [...new Set(tables.panel.map(r => r.maturity_rank))].sort((a,b)=>a-b);
  if ([...byDate.values()].some(rows => rows.length !== ranks.length)) throw new Error('Incomplete observed curve');
  if (new Set(tables.states.map(r=>r.date_)).size !== tables.states.length || dates.some(d=>!tables.states.some(r=>r.date_===d && Number.isFinite(r.chi) && Number.isFinite(r.xi)))) throw new Error('Missing or duplicate filtered states');
  if (meta.forecast && (!Array.isArray(meta.forecast.horizons) || !meta.forecast.horizons.length || tables.forecast_predictions.some(r => r.origin_date >= r.date_ || !['random_walk','one_factor','two_factor'].includes(r.model) || Math.abs(r.observed_log-r.predicted_log-r.error_log)>1e-7))) throw new Error('Invalid or leaky saved forecasts');
  return {metadata:meta, ...tables, dates, ranks, byDate};
}
export function nearestDate(dates, value) {
  if (!dates.length) return null;
  return dates.reduce((best,date) => Math.abs(Date.parse(date)-Date.parse(value)) < Math.abs(Date.parse(best)-Date.parse(value)) ? date : best, dates[0]);
}
export function selectRows(bundle, table, filters={}) {
  return (bundle[table] || []).filter(row => Object.entries(filters).every(([key,value]) => value === 'all' && key === 'year' ? true : key === 'year' ? row.date_?.startsWith(String(value)) : row[key] === value));
}
export function csv(rows, columns=Object.keys(rows[0] || {})) {
  const escape = value => {
    let text = String(value ?? '');
    if (typeof value === 'string' && /^[=+@-]/.test(text)) text = `'${text}`;
    return /[",\n\r]/.test(text) ? `"${text.replaceAll('"','""')}"` : text;
  };
  return [columns.join(','), ...rows.map(row => columns.map(k=>escape(row[k])).join(','))].join('\n');
}
