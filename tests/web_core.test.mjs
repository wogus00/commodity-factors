import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {decodeBundle, VIEWS, nearestDate, csv} from '../web/core.js';
import {viewPanels} from '../web/views.js';
const payload=JSON.parse(await readFile(new URL('../web/data/modeling-demo.json',import.meta.url),'utf8'));
const bundle=decodeBundle(payload);
const selection={view:'overview',chart:'summary',snapshot:bundle.dates.at(-1),rank:1,window:252,method:'covariance',period:'all',k:3,track:1,coverage:'strict',forecastSplit:'test',forecastHorizon:1,log:false};
test('modeling export includes only observed curves and filtered structural outputs',()=>{
 assert.ok(['panel','ss_loadings','ss_reconstructions','states','pca_explained_variance','pca_loadings','pca_scores','pca_reconstructions','rolling_spectrum','rolling_fits','rolling_loadings','rolling_metrics'].every(key=>payload.tables[key]));
 assert.equal(bundle.metadata.state_estimate,'filtered');
 assert.equal(bundle.metadata.smoothed_states_available,false);
 assert.equal(bundle.metadata.dataset_id,'synthetic');
 assert.equal(bundle.metadata.pca.transformation,'log_changes');
 assert.deepEqual(bundle.metadata.robustness.windows,[126,252,504]);
 assert.deepEqual(bundle.metadata.forecast.horizons,[1,5,20]);
 assert.ok(bundle.ss_reconstructions.every(r=>Math.abs(r.observed_log-r.fitted_log-r.residual_log)<1e-10));
});
test('legacy forecasting bundles and missing states are rejected',()=>{
 assert.throws(()=>decodeBundle({...payload,format:'commodity-factors-web'}),/modeling export/);
 const changed=structuredClone(payload);changed.tables.states.data.pop();
 assert.throws(()=>decodeBundle(changed),/filtered states/);
});
test('malformed data and prototype column injection are rejected',()=>{
 const changed=structuredClone(payload);changed.tables.panel.columns[0]='__proto__';
 assert.throws(()=>decodeBundle(changed),/Invalid/);
 const negative=structuredClone(payload);negative.tables.panel.data[0][negative.tables.panel.columns.indexOf('settlement')]=-1;
 assert.throws(()=>decodeBundle(negative),/observed curve/);
});
test('all seven sections and their chart selections render safely with saved synthetic outputs',()=>{
 assert.equal(Object.keys(VIEWS).length,7);
 assert.deepEqual(Object.keys(VIEWS),['overview','structure','schwartz','forecast','pca','robustness','comparison']);
 for(const [view,config] of Object.entries(VIEWS))for(const chart of config.charts){
  const result=viewPanels(bundle,{...selection,view,chart});
  assert.ok(result.main.title);assert.ok(Array.isArray(result.rows));assert.ok(Array.isArray(result.main.traces));
 }
 assert.match(viewPanels(bundle,{...selection,view:'pca'}).note,/Synthetic data; not historical WTI/);
 assert.match(viewPanels(bundle,{...selection,view:'robustness'}).note,/does not assess the 2022 WTI market/);
 assert.match(viewPanels(bundle,{...selection,view:'comparison'}).note,/No matched-target/);
 assert.ok(viewPanels(bundle,{...selection,view:'overview'}).rows.some(row=>row.section==='Forecast evaluation'&&/Saved held-out/.test(row.availability)));
});
test('PCA bundle has finite loadings, reconstruction metrics, and prior-only rolling fits',()=>{
 assert.ok(bundle.pca_explained_variance.every(r=>Number.isFinite(r.ratio)&&r.ratio>=0));
 assert.ok(bundle.pca_loadings.every(r=>[r.PC1,r.PC2,r.PC3].every(Number.isFinite)));
 assert.ok(bundle.pca_reconstructions.some(r=>Number.isFinite(r.fitted_log)));
 assert.ok(bundle.rolling_metrics.some(r=>r.window===126&&r.k===1));
 assert.ok(bundle.rolling_metrics.some(r=>r.window===252&&r.k===2));
 assert.ok(bundle.rolling_metrics.some(r=>r.window===504&&r.k===3));
 assert.ok(bundle.rolling_fits.filter(r=>r.scored).every(r=>r.train_end<r.date_));
});
test('forecast views use saved held-out predictions and metrics without fitting in the browser',()=>{
 const prediction=viewPanels(bundle,{...selection,view:'forecast',chart:'predictions'});
 const source=bundle.forecast_predictions.filter(r=>r.split==='test'&&r.horizon===1&&r.maturity_rank===1);
 assert.equal(prediction.main.traces.length,4);
 assert.equal(prediction.main.traces[0].x.length,new Set(source.map(r=>r.date_)).size);
 assert.ok(bundle.forecast_predictions.every(r=>r.origin_date<r.date_));
 assert.match(prediction.note,/Synthetic data/);
 for(const chart of ['model-errors','maturity-errors','horizons']){
  const result=viewPanels(bundle,{...selection,view:'forecast',chart});
  assert.ok(result.main.traces.length,`${chart} has saved traces`);
  assert.ok(result.main.traces.every(trace=>trace.y.every(Number.isFinite)),`${chart} has finite metrics`);
  if(chart==='maturity-errors')assert.equal(result.main.traces[0].x.length,6);
 }
});
test('structural curves use saved fitted values, not browser-generated future prices',()=>{
 const result=viewPanels(bundle,{...selection,view:'schwartz',chart:'curve',log:true});
 assert.deepEqual(result.main.traces[1].y,bundle.ss_reconstructions.filter(r=>r.date_===selection.snapshot).map(r=>r.fitted_log));
 assert.match(result.note,/filtered|Filtered/);assert.match(result.note,/smoothed states are unavailable/);
});
test('CSV escapes formula-like strings and snapshot selection uses existing dates',()=>{
 assert.match(csv([{name:'=SUM(1,2)',value:1}]),/"'=SUM/);
 assert.equal(nearestDate(['2025-01-03','2025-01-06'],'2025-01-05'),'2025-01-06');
});
test('markup exposes forecast evaluation while excluding the old live daily forecast interface',async()=>{
 const html=await readFile(new URL('../web/index.html',import.meta.url),'utf8');
 assert.match(html,/<title>Latent Factor Modeling of Crude Oil Futures<\/title>/);
 assert.ok(html.indexOf('data-view="forecast"')<html.indexOf('data-view="pca"'));
 assert.match(html,/Synthetic forecast benchmark/);
 assert.match(html,/overview-forecast-open/);
 assert.doesNotMatch(html,/Saved numerical results|evidence-head|evidence-rows|table-caption/);
 assert.match(html,/data-view="forecast"/);
 assert.match(html,/forecast-horizon-select/);
 assert.doesNotMatch(html,/daily-target|data-view="daily"/);
 for(const name of ['app.js','core.js','views.js']){
  const text=await readFile(new URL(`../web/${name}`,import.meta.url),'utf8');
  assert.doesNotMatch(text,/forecastContract|dailySnapshot|selectPredictions|fetchLive/);
 }
});
// Real saved outputs are optional locally; CI's checked-in fixture remains synthetic.
const empiricalPath=process.env.MODELING_TEST_BUNDLE;
if(empiricalPath){
 const empirical=decodeBundle(JSON.parse(await readFile(empiricalPath,'utf8')));
 test('every historical PCA and robustness view uses finite saved traces',()=>{
  for(const [view,config] of Object.entries(VIEWS))for(const chart of config.charts){
   const result=viewPanels(empirical,{...selection,view,chart,snapshot:empirical.dates.at(-1)});
   assert.ok(result.main.traces.length,`${view}/${chart} has no traces`);
   for(const trace of result.main.traces)assert.ok(trace.y.every(y=>y===null||Number.isFinite(y)),`${view}/${chart} contains nonfinite values`);
  }
 });
 test('rolling reconstruction metrics preserve prior-only sample counts and source values',()=>{
  const result=viewPanels(empirical,{...selection,view:'pca',chart:'errors'});
  const expected=empirical.rolling_metrics.filter(r=>r.window===252&&r.method==='covariance'&&r.period==='all'&&r.tau===0);
  assert.deepEqual(result.main.traces[0].y,expected.map(r=>r.rmse_log));
  assert.equal(new Set(expected.map(r=>r.n_dates)).size,1);
 });
 test('sensitivity coverage policies select distinct saved outputs and expose excluded scenarios',()=>{
  const strict=viewPanels(empirical,{...selection,view:'robustness',chart:'sensitivity'});
  const relaxed=viewPanels(empirical,{...selection,view:'robustness',chart:'sensitivity',coverage:'relaxed'});
  assert.match(strict.note,/missing_quotes/);
  assert.ok(!strict.main.traces[0].x.includes('missing_quotes'));
  assert.ok(relaxed.main.traces[0].x.includes('missing_quotes'));
 });
}
