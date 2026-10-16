import {loadBlobDataset} from './blob.js';
import {VIEWS, decodeBundle, nearestDate, csv} from './core.js';
import {viewPanels} from './views.js';
const $ = id => document.getElementById(id);
const state = {view:'overview', chart:'summary', snapshot:'', rank:1, window:252, method:'covariance', period:'all', k:3, track:1, coverage:'strict', forecastSplit:'test', forecastHorizon:1, log:false};
const datasets = new Map();
let bundle, toastTimer, currentRows=[];
let manifestUrl='', remoteBusy=false;
const text = (id,value) => { $(id).textContent=value; };
const number = value => value===null || value===undefined ? '—' : typeof value==='number' ? Number(value.toPrecision(6)).toLocaleString('en-US',{maximumFractionDigits:8}) : String(value);
function toast(message) {
  clearTimeout(toastTimer); text("toast", message); $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 6000);
}

function option(value, label) { const el = document.createElement("option"); el.value = value; el.textContent = label; return el; }
function cell(value, className = "") { const td = document.createElement("td"); td.textContent = value; td.className = className; return td; }

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
    margin: {l: 32, r: 66, t: 18, b: 28}, hovermode: "x unified", dragmode: "pan", showlegend: true, legend: {orientation: "h", y: 1.12},
    hoverlabel: {bgcolor: "#20293a", bordercolor: "#48546a", font: {size: 11, color: "#d9e2f5"}},
    xaxis: {...axis, showgrid: false}, yaxis: {...axis, side: "right"},
    modebar: {bgcolor: "transparent", color: "#7c8aa2", activecolor: "#7da5ff"}, ...overrides};
  const config = {responsive: true, displaylogo: false, scrollZoom: true, displayModeBar: "hover",
    modeBarButtons: [["toImage"], ["zoom2d", "pan2d"], ["zoomIn2d", "zoomOut2d", "resetScale2d"]],
    toImageButtonOptions: {filename: "crude-oil-factor-modeling", format: "png", scale: 2}};
  return Plotly.react($(id), traces, layout, config);
}


function setView(view) {
  if(!VIEWS[view]) view='overview';
  state.view=view; state.chart=VIEWS[view].charts[0];
  const labels={'model-errors':'Model errors','maturity-errors':'Errors by maturity'};
  $('chart-select').replaceChildren(...VIEWS[view].charts.map(key=>option(key,labels[key]||key[0].toUpperCase()+key.slice(1))));
  document.querySelectorAll('[data-view]').forEach(button=>{button.classList.toggle('active',button.dataset.view===view);button.setAttribute('aria-pressed',button.dataset.view===view);});
  if(location.hash!==`#${view}`) history.replaceState(null,'',`#${view}`);
  render();
}
function loadDataset(key) {
  bundle=datasets.get(key); state.snapshot=bundle.dates.at(-1); state.rank=bundle.ranks[0];
  $('snapshot-date').min=bundle.dates[0];$('snapshot-date').max=bundle.dates.at(-1);$('snapshot-date').value=state.snapshot;
  $('rank-select').replaceChildren(...bundle.ranks.map(rank=>option(rank,`M${rank}`)));
  const windows=bundle.metadata.robustness?.windows || [126,252,504];
  state.window=windows.includes(252)?252:windows[0];
  $('window-select').replaceChildren(...windows.map(w=>option(w,`${w} dates`)));$('window-select').value=state.window;
  const periods=[...new Set((bundle.rolling_metrics||[]).map(r=>r.period))].filter(p=>p!=='all').sort();
  $('period-select').replaceChildren(option('all','All held-out years'),...periods.map(p=>option(p,p)));state.period='all';
  $('coverage-select').options[1].disabled=!bundle.metadata.relaxed_sensitivity;
  state.coverage='strict';$('coverage-select').value='strict';
  const strict=bundle.metadata.robustness?.min_coverage, relaxed=bundle.metadata.relaxed_sensitivity?.min_coverage;
  $('coverage-select').options[0].textContent=strict===undefined?'Default coverage':`${strict*100}% minimum`;
  $('coverage-select').options[1].textContent=relaxed===undefined?'Relaxed diagnostic unavailable':`${relaxed*100}% diagnostic`;
  const forecast=bundle.metadata.forecast;
  const splits=[...new Set((bundle.forecast_metrics||[]).map(r=>r.split))];
  $('forecast-split-select').replaceChildren(...splits.map(key=>option(key,key[0].toUpperCase()+key.slice(1))));
  $('forecast-horizon-select').replaceChildren(...(forecast?.horizons||[]).map(h=>option(h,`${h}-day`)));
  state.forecastSplit=splits.includes('test')?'test':splits[0]||'validation';
  state.forecastHorizon=forecast?.horizons?.includes(1)?1:forecast?.horizons?.[0]||1;
  return render();
}
function headerAndContracts() {
  const observed=bundle.byDate.get(state.snapshot), quote=observed.find(r=>r.maturity_rank===state.rank)||observed[0];
  text('instrument-name',bundle.metadata.dataset_label);text('last-price',number(state.log?quote.log_settlement:quote.settlement));
  text('quote-date',`${state.snapshot} · Observed M${quote.maturity_rank}`);text('last-change','Observed settlement');
  const synthetic=bundle.metadata.dataset_id==='synthetic';text('data-badge',synthetic?'SYNTHETIC EXAMPLE':'HISTORICAL WTI');
  text('data-source',bundle.metadata.source);text('train-end',bundle.metadata.train_end);text('observation-count',bundle.panel.length.toLocaleString());
  text('source-note',synthetic?'Seeded synthetic curves. PCA and robustness are computed from simulated data, not observed WTI.':bundle.remote?'Approved historical results loaded from Vercel Blob.':'Saved historical results imported locally; no data is uploaded.');
  text('availability-note',`Filtered states available. Smoothed states and standardized innovations are not included. PCA: ${bundle.metadata.pca?'available':'not included'}. Robustness: ${bundle.metadata.robustness?'available':'not included'}. Forecast evaluation: ${bundle.metadata.forecast?'saved synthetic results':'not included'}.`);
  text('status-text',synthetic?'Synthetic example · Saved model outputs':'Historical WTI · Saved model outputs');text('status-period',`${bundle.dates[0]} — ${bundle.dates.at(-1)}`);
  $('contract-rows').replaceChildren(...observed.map(r=>{
    const tr=document.createElement('tr'), td=cell(''), button=document.createElement('button');
    button.textContent=`M${r.maturity_rank} · ${r.futcode}`;button.addEventListener('click',()=>{state.rank=r.maturity_rank;$('rank-select').value=state.rank;render();});td.append(button);
    const old=(bundle.byDate.get(bundle.dates[bundle.dates.indexOf(state.snapshot)-1])||[]).find(p=>p.futcode===r.futcode);
    tr.append(td,cell(number(r.settlement)),cell(old?`${number((r.settlement/old.settlement-1)*100)}%`:'—'));return tr;
  }));
  const spread=observed.at(-1).settlement-observed[0].settlement;
  text('curve-spread',number(spread));text('curve-shape',spread>0?'Contango':spread<0?'Backwardation':'Flat');
}
function renderOverviewForecast() {
  const panel=$('overview-forecast-summary');
  const rows=(bundle.forecast_metrics||[]).filter(row=>row.split==='test'&&row.horizon===1&&row.maturity_rank===0);
  if(state.view!=='overview'||bundle.metadata.dataset_id!=='synthetic'||!rows.length){panel.hidden=true;return;}
  panel.hidden=false;
  const baseline=rows.find(row=>row.model==='random_walk');
  text('overview-forecast-caption',`Held-out 1-day test forecasts · ${baseline?.n_dates??0} scored dates · training through ${bundle.metadata.forecast.training_end}. Simulated data, not observed WTI.`);
  const colors={random_walk:'#e4b05c',one_factor:'#b695f9',two_factor:'#5b8cff'};
  const names={random_walk:'Random walk',one_factor:'One factor',two_factor:'Two factors'};
  $('overview-forecast-cards').replaceChildren(...rows.map(row=>{
    const card=document.createElement('article');card.className='overview-forecast-card';card.setAttribute('role','listitem');card.style.setProperty('--forecast-color',colors[row.model]);
    const name=document.createElement('span');name.textContent=names[row.model]||row.model;
    const value=document.createElement('strong');value.textContent=`$${row.rmse_price.toFixed(2)}`;
    const detail=document.createElement('small');
    if(row.model==='random_walk'||!baseline)detail.textContent='RMSE · reference baseline';
    else {const change=(1-row.rmse_price/baseline.rmse_price)*100;detail.textContent=`RMSE · ${Math.abs(change).toFixed(1)}% ${change>=0?'lower':'higher'} than random walk`;}
    card.append(name,value,detail);return card;
  }));
}
function renderChart(id,spec) {
  const overrides={xaxis:{title:{text:spec.x},gridcolor:'#252a36',automargin:true},yaxis:{title:{text:spec.y},gridcolor:'#252a36',automargin:true},annotations:[]};
  if(!spec.traces.length) overrides.annotations=[{text:'No saved data for this selection',xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false}];
  if(spec.traces.some(t=>t.yaxis==='y2')) overrides.yaxis2={title:'ξ · log units',overlaying:'y',side:'right',showgrid:false};
  return plot(id,spec.traces,overrides);
}
function render() {
  if(!bundle) return;
  headerAndContracts();renderOverviewForecast();const panels=viewPanels(bundle,state);
  text('methodology-note',panels.note);text('chart-heading',panels.main.title);text('chart-subtitle',VIEWS[state.view].label);
  text('chart-note',panels.main.note);text('lower-heading',panels.lower.title);text('lower-context',panels.lower.note===panels.lower.title?'':panels.lower.note);
  $('lower-chart').hidden=!panels.lower.traces.length;
  const advanced=['robustness','comparison'].includes(state.view)||state.view==='pca'&&state.chart==='errors';
  for(const key of ['window','method','period','k','track','coverage','rank','forecast-split','forecast-horizon']) {
    const visible=key==='rank'?['overview','structure','schwartz','forecast'].includes(state.view)
      :key==='coverage'?state.view==='robustness'&&state.chart==='sensitivity'
      :key==='track'?state.view==='robustness'&&['loadings','similarity','annual'].includes(state.chart)
      :key==='k'?state.view==='robustness'&&['maturity','sensitivity'].includes(state.chart)||state.view==='comparison'&&state.chart==='windows'
      :key==='forecast-split'||key==='forecast-horizon'?state.view==='forecast'
      :key==='period'?advanced&&['errors','components','maturity','sensitivity','scaling','windows'].includes(state.chart)
      :key==='window'?advanced&&!['annual','sensitivity','windows'].includes(state.chart):advanced;
    document.querySelector(`[data-control="${key}"]`).hidden=!visible;
  }
  $('scale-button').textContent=state.log?'LOG':'USD';$('scale-button').setAttribute('aria-pressed',state.log);
  currentRows=panels.rows;$('export-button').disabled=!currentRows.length;$('chart-empty').hidden=true;
  return Promise.all([renderChart('main-chart',panels.main),panels.lower.traces.length?renderChart('lower-chart',panels.lower):Promise.resolve()]);
}
function details() {
  const root=$('details-content');root.replaceChildren();
  const p=document.createElement('p');p.textContent=`${bundle.metadata.dataset_label} · ${bundle.metadata.source}. ${bundle.metadata.state_provenance}`;
  const comparison=document.createElement('p');comparison.textContent=bundle.metadata.comparison_note;
  const heading=document.createElement('h3');heading.textContent='Export provenance and saved methodology';
  const pre=document.createElement('pre');pre.textContent=JSON.stringify(bundle.metadata,null,2);root.append(p,comparison,heading,pre);$('details-dialog').showModal();
}
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>setView(button.dataset.view)));
$('overview-forecast-open').addEventListener('click',()=>setView('forecast'));
$('chart-select').addEventListener('change',event=>{state.chart=event.target.value;render();});
$('dataset-select').addEventListener('change',event=>loadDataset(event.target.value));
for(const key of ['rank','window','method','period','k','track','coverage']) $(`${key}-select`).addEventListener('change',event=>{state[key]=['rank','window','k','track'].includes(key)?Number(event.target.value):event.target.value;render();});
for(const [key,id] of [['forecastSplit','forecast-split-select'],['forecastHorizon','forecast-horizon-select']]) $(id).addEventListener('change',event=>{state[key]=key==='forecastHorizon'?Number(event.target.value):event.target.value;render();});
bindDateCommit('snapshot-date',event=>{if(event.target.value&&bundle){state.snapshot=nearestDate(bundle.dates,event.target.value);event.target.value=state.snapshot;render();}});
$('scale-button').addEventListener('click',()=>{state.log=!state.log;render();});
$('reset-chart').addEventListener('click',()=>Plotly.relayout($('main-chart'),{'xaxis.autorange':true,'yaxis.autorange':true}));
for(const id of ['help-button','details-button']) $(id).addEventListener('click',()=>bundle&&details());
$('close-dialog').addEventListener('click',()=>$('details-dialog').close());
$('import-button').addEventListener('click',()=>$('import-file').click());
$('import-file').addEventListener('change',async event=>{
  const file=event.target.files[0];if(!file)return;
  try{if(file.size>30_000_000)throw new Error('Modeling bundle exceeds the 30 MB import limit');const imported=decodeBundle(JSON.parse(await file.text()));const key=`local-${datasets.size}`;datasets.set(key,imported);$('dataset-select').append(option(key,`${imported.metadata.dataset_label} · Local`));$('dataset-select').value=key;await loadDataset(key);toast('Modeling results loaded locally. Nothing is uploaded.');}catch(error){toast(error.message);}event.target.value='';
});
$('export-button').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([csv(currentRows)],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=`${bundle.metadata.dataset_id}_${state.view}_${state.chart}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
document.addEventListener('keydown',event=>{if(['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName)||event.metaKey||event.ctrlKey||event.altKey||$('details-dialog').open)return;const view=Object.keys(VIEWS)[Number(event.key)-1];if(view)setView(view);if(event.key.toLowerCase()==='f')$('reset-chart').click();if(event.key.toLowerCase()==='l')$('scale-button').click();});
window.addEventListener('hashchange',()=>setView(location.hash.slice(1)));
async function refreshRemote() {
  if(!manifestUrl || remoteBusy)return;
  remoteBusy=true;$('refresh-data').disabled=true;
  const previous=$('dataset-select').value;
  text('remote-status','Loading historical modeling results from Blob…');
  try {
    const remote=await loadBlobDataset(manifestUrl);remote.remote=true;
    datasets.set('blob',remote);
    if(!$('dataset-select').querySelector('option[value="blob"]'))$('dataset-select').append(option('blob',`${remote.metadata.dataset_label} · Blob`));
    // Do not replace an import or dataset selection made while the request ran.
    if($('dataset-select').value===previous && ['','demo','blob'].includes(previous)){$('dataset-select').value='blob';await loadDataset('blob');}
    text('remote-status',`Blob loaded · ${remote.dates[0]} — ${remote.dates.at(-1)} · Saved modeling results`);
  } catch(error) {
    text('remote-status',`Blob unavailable: ${error.message}. Current dataset retained; use Refresh data to retry.`);
  } finally {remoteBusy=false;$('refresh-data').disabled=false;}
}
$('refresh-data').addEventListener('click',refreshRemote);
async function initialize(){
  try{const response=await fetch('data/modeling-demo.json');if(!response.ok)throw new Error('Saved modeling example could not be loaded');const demo=decodeBundle(await response.json());datasets.set('demo',demo);$('dataset-select').replaceChildren(option('demo','Synthetic example'));await loadDataset('demo');setView(location.hash.slice(1)||'overview');}
  catch(error){text('status-text','Modeling data unavailable');toast(error.message);text('chart-empty','Import a saved modeling bundle to begin.');$('chart-empty').hidden=false;}
}
async function initializeSource(){
  try {
    const response=await fetch('modeling-source.json',{cache:'no-cache'});
    if(!response.ok)throw new Error('Data source configuration could not be loaded');
    const config=await response.json();
    if(typeof config.manifest_url!=='string')throw new Error('Invalid data source configuration');
    manifestUrl=config.manifest_url;
    $('refresh-data').hidden=!manifestUrl;
    if(manifestUrl)await refreshRemote();
    else text('remote-status','Blob source not configured · Synthetic example and local imports available');
  }catch(error){text('remote-status',error.message+'; reload to retry. Local imports remain available.');}
}
await initialize();
await initializeSource();
