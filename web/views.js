import {COLORS, selectRows, nearestDate} from './core.js';
const palette = ['#5b8cff','#e4b05c','#b695f9','#2fc6ad','#ef7f89','#a4bacb'];
const line = (rows,x,y,name,color=palette[0],extra={}) => ({type:'scatter',mode:'lines',x:rows.map(r=>r[x]),y:rows.map(r=>r[y]),name,line:{color,width:1.7},...extra});
const grouped = (rows,key,x,y) => [...new Set(rows.map(r=>r[key]))].map((name,i)=>line(rows.filter(r=>r[key]===name).sort((a,b)=>a[x]<b[x]?-1:1),x,y,String(name),palette[i%palette.length]));
const chart = (title,traces=[],x='',y='',note='') => ({title,traces,x,y,note});
const empty = message => chart(message,[], '', '',message);
const components = ['PC1','PC2','PC3'];
export function viewPanels(bundle,s) {
  const observed = bundle.byDate.get(s.snapshot) || [];
  const curveRows = observed.map(r=>({...r,value:s.log?r.log_settlement:r.settlement,days:r.tau*365.25}));
  const observedChart = chart('Observed futures term structure',[line(curveRows,'days','value','Observed',COLORS.observed,{mode:'lines+markers'})],'Days to last trading date',s.log?'Log price':bundle.metadata.price_unit,s.snapshot);
  let main=observedChart, lower=empty(''), rows=[], note='Same-day modeling and reconstruction. No future prices are generated.';
  const hasPCA = Boolean(bundle.metadata.pca), hasRobust = Boolean(bundle.metadata.robustness);
  const pcaCurve = () => {
    const dates=[...new Set((bundle.pca_reconstructions||[]).filter(r=>r.fitted_log!==null).map(r=>r.date_))];
    const date=nearestDate(dates,s.snapshot);
    const data=(bundle.pca_reconstructions||[]).filter(r=>r.date_===date).sort((a,b)=>a.tau-b.tau).map(r=>({...r,months:r.tau*12,observed:s.log?r.observed_log:Math.exp(r.observed_log),fitted:s.log?r.fitted_log:Math.exp(r.fitted_log)}));
    return chart('PCA same-day curve reconstruction',[line(data,'months','observed','Observed · interpolated',COLORS.observed,{mode:'lines+markers'}),line(data,'months','fitted','Reconstructed',palette[0],{mode:'lines+markers'})],'Constant maturity (months)',s.log?'Log price':bundle.metadata.price_unit,date?`Eligible curve ${date}; may differ from selected date. Uses this date’s scores.`:'No eligible PCA curve');
  };
  const metricRows = (table='rolling_metrics') => (bundle[table]||[]).filter(r=>r.method===s.method && r.period===s.period && (table==='common_window_metrics'||r.window===s.window));
  if(s.view==='overview') {
    main=chart('Observed settlement history',grouped(bundle.panel.filter(r=>r.maturity_rank===s.rank).map(r=>({...r,value:s.log?r.log_settlement:r.settlement})), 'maturity_rank','date_','value'),'Observation date',s.log?'Log price':bundle.metadata.price_unit,'Nearby rank changes contract at rolls.');
    lower=observedChart;
    rows=[{section:'Schwartz–Smith',availability:'Saved filtered states and same-day curves'}, {section:'PCA',availability:hasPCA?'Saved loadings, scores and reconstruction':'Not included in this bundle'}, {section:'PCA robustness',availability:hasRobust?'Prior-only rolling fits and sensitivity results':'Not included in this bundle'}, {section:'Forecast evaluation',availability:bundle.metadata.forecast?'Saved held-out predictions and error metrics':'Not included in this bundle'}];
    note=bundle.metadata.dataset_id==='synthetic'?'Synthetic example, not historical WTI evidence. PCA and rolling robustness are computed from simulated curves.':'Historical WTI modeling outputs. All fits are precomputed; changing controls only selects saved results.';
  } else if(s.view==='structure') {
    if(s.chart==='history') main=chart('Observed rank history', [line(bundle.panel.filter(r=>r.maturity_rank===s.rank).map(r=>({...r,value:s.log?r.log_settlement:r.settlement})),'date_','value',`M${s.rank}`,COLORS.observed)],'Date',s.log?'Log price':bundle.metadata.price_unit,'Contract rolls can change rank identity; these are not constant-maturity changes.');
    rows=observed; lower=observedChart;
  } else if(s.view==='schwartz') {
    const reconstruction=bundle.ss_reconstructions.filter(r=>r.date_===s.snapshot).sort((a,b)=>a.tau-b.tau).map(r=>({...r,days:r.tau*365.25,value:s.log?r.fitted_log:Math.exp(r.fitted_log)}));
    const fitted=chart('Observed versus model-implied curve',[...observedChart.traces,line(reconstruction,'days','value','Schwartz–Smith same-day fit',palette[0],{mode:'lines+markers'})],observedChart.x,observedChart.y,'Filtered states include this day’s observations.');
    lower=fitted;
    rows=Object.entries(bundle.metadata.parameters).map(([parameter,value])=>({parameter,value}));
    note=`Filtered estimates only: smoothed states are unavailable. Standardized innovations are unavailable in these saved files. Residuals are observed minus same-day fitted log prices, not Kalman innovations. ${bundle.metadata.state_provenance}`;
    if(s.chart==='factors') main=chart('Filtered short-term χ and long-term ξ',[line(bundle.states,'date_','chi','Short-term χ',COLORS.short),line(bundle.states,'date_','xi','Long-term ξ',COLORS.long,{yaxis:'y2'})],'Observation date','χ · log units','ξ uses the right axis. These are filtered, not smoothed, estimates.');
    if(s.chart==='curve') main=fitted;
    if(s.chart==='loadings') main=chart('Maturity-dependent factor loadings', ['short_loading','long_loading'].map((key,i)=>line(bundle.ss_loadings.filter(r=>r.date_===s.snapshot).map(r=>({...r,days:r.tau*365.25})),'days',key,i?'Long-term: 1':'Short-term: exp(−κτ)',palette[i])),'Days to maturity','Loading','Loadings exported using the saved parameter estimate.');
    if(s.chart==='residuals') { rows=bundle.ss_reconstructions.filter(r=>r.maturity_rank===s.rank); main=chart('Same-day log-price residuals',[line(rows,'date_','residual_log',`M${s.rank}`,palette[1])],'Date','Observed − fitted log price','Not standardized innovations.'); }
    if(s.chart==='likelihood') main=chart('Saved Kalman observation log-likelihood',[line(bundle.states,'date_','two_factor_predictive_ll','Observation log-likelihood',palette[0])],'Date','Log-likelihood contribution','Available scalar diagnostics only; innovation vectors and covariance matrices were not persisted.');
  } else if(s.view==='pca') {
    if(!hasPCA) return {main:empty('PCA outputs are not included in this bundle'),lower:empty('Import a historical modeling bundle containing PCA results.'),rows:[],note:'No PCA estimates are fabricated for the synthetic example.'};
    const p=bundle.metadata.pca;
    note=`${bundle.metadata.dataset_id==='synthetic'?'Synthetic data; not historical WTI. ':''}${p.transformation}; ${p.standardize?'correlation':'covariance'} PCA; training ends ${p.train_end}. Fixed maturity grid, no time filling or extrapolation. Scores use contemporaneous curves. Reconstruction is not forecasting.`;
    lower=pcaCurve(); rows=bundle.pca_explained_variance;
    if(s.chart==='variance') main=chart('Explained and cumulative training variance',[{type:'bar',x:components,y:bundle.pca_explained_variance.map(r=>r.ratio*100),name:'Individual',marker:{color:palette[0]}},line(bundle.pca_explained_variance.map(r=>({...r,percent:r.cumulative_ratio*100})),'component','percent','Cumulative',palette[1],{mode:'lines+markers'})],'Component','Variance (%)');
    if(s.chart==='loadings') {rows=bundle.pca_interpretations;main=chart('PCA maturity loadings',components.map((pc,i)=>line(bundle.pca_loadings.map(r=>({...r,months:r.tau*12})),'months',pc,pc,palette[i],{mode:'lines+markers'})),'Constant maturity (months)','Loading in original feature units','Economic labels are assessed by shape alignment, not assigned automatically.');}
    if(s.chart==='scores') {rows=bundle.pca_scores;main=chart('Principal component scores',components.map((pc,i)=>line(bundle.pca_scores,'date_',pc,pc,palette[i])),'Date','Raw score','Gaps remain missing. Loadings and centering are frozen after training.');}
    if(s.chart==='curve') {main=pcaCurve();rows=bundle.pca_metrics;}
    if(s.chart==='errors') {rows=metricRows().filter(r=>r.tau===0); main=hasRobust?chart('Held-out reconstruction error: K=1,2,3',[line(rows,'k','rmse_log',s.method,palette[0],{mode:'lines+markers'})],'Components K','Log RMSE',`${s.window}-date prior-only fits · ${s.period} · same-day held-out compression`):empty('K=1,2,3 comparisons require the robustness bundle');}
  } else if(s.view==='robustness') {
    if(!hasRobust) return {main:empty('PCA robustness outputs are not included'),lower:empty('Import a modeling bundle with saved rolling results.'),rows:[],note:'No rolling fits run in the browser.'};
    const spec=selectRows(bundle,'rolling_spectrum',{window:s.window,method:s.method});
    note=`${bundle.metadata.dataset_id==='synthetic'?'Synthetic data; available dates are 2023–2025, so this does not assess the 2022 WTI market. ':''}${s.window} supplied trading dates; ${s.method} PCA. Means/scales/loadings are fitted only on prior observations. Tracked loading labels are sign/permutation aligned; reconstruction retains variance order.`;
    rows=metricRows().filter(r=>r.tau===0);
    lower=chart('Reconstruction RMSE by component count',[line(rows,'k','rmse_log',s.method,palette[0],{mode:'lines+markers'})],'K','Log RMSE',`${s.period}; no future-date prediction`);
    if(s.chart==='variance') main=chart('Rolling explained variance',grouped(spec.map(r=>({...r,percent:r.variance_ratio*100})),'source_rank','date_','percent'),'Date','Variance (%)','Series are ordered by variance rank, not tracked identity.');
    if(s.chart==='similarity') {rows=spec.filter(r=>r.track===s.track);main=chart('Aligned loading similarity',[line(rows,'date_','anchor_cosine','To first valid window',palette[0]),line(rows,'date_','previous_cosine','To previous window',palette[1])],'Date','Signed cosine',`Track ${s.track}; strong adjacent similarity can reflect overlapping windows.`);}
    if(s.chart==='angles') {rows=selectRows(bundle,'rolling_fits',{window:s.window,method:s.method});main=chart('Principal subspace angle',[line(rows,'date_','anchor_subspace_angle_deg','Maximum angle to first window',palette[0])],'Date','Degrees','Top-three span; rotations among nearly tied PCs need not change the span.');}
    if(s.chart==='loadings') {rows=selectRows(bundle,'rolling_loadings',{window:s.window,method:s.method,track:s.track});main=chart('Rolling maturity-dependent loading',grouped(rows,'tau','date_','loading'),'Date','Aligned unit loading',`Track ${s.track}; legend shows maturity in years.`);}
    if(s.chart==='annual') {rows=selectRows(bundle,'subperiod_loadings',{method:s.method,track:s.track});main=chart('Annual loading comparison',grouped(rows.map(r=>({...r,months:r.tau*12})),'period','months','loading'),'Maturity (months)','Aligned unit loading','Descriptive full-year fits, including 2022; not held-out fits.');}
    if(s.chart==='components') main=lower;
    if(s.chart==='maturity') {rows=metricRows().filter(r=>r.k===s.k && r.tau>0);main=chart('Reconstruction RMSE by maturity',[line(rows.map(r=>({...r,months:r.tau*12})),'months','rmse_log',`${s.method} · K=${s.k}`,palette[0],{mode:'lines+markers'})],'Maturity (months)','Log RMSE',`${s.period} · K=${s.k}`);}
    if(s.chart==='sensitivity') {
      const table=s.coverage==='relaxed'?'relaxed_sensitivity_metrics':'sensitivity_metrics';
      rows=selectRows(bundle,table,{method:s.method,k:s.k,period:s.period}).filter(r=>r.tau===0);
      main=chart('Preprocessing sensitivity',[{type:'bar',x:rows.map(r=>r.scenario),y:rows.map(r=>r.rmse_log),name:'Common-support RMSE',marker:{color:palette[0]}}],'Scenario','Log RMSE','Common dates and maturities; target is the baseline-linear observed curve. Different coverage policies use different common samples.');
      const meta=s.coverage==='relaxed'?bundle.metadata.relaxed_sensitivity:bundle.metadata.robustness;
      note=`Saved sensitivity window: ${meta?.sensitivity_window ?? "unavailable"} supplied dates; ${s.method} PCA. Minimum training coverage: ${meta?meta.min_coverage*100:'unavailable'}%. Unscored scenarios: ${meta?.unscored_scenarios?.join(', ')||'none'}. Quote loss can prevent fitting; absence is not a zero error.`;
      if(!rows.length) main=empty('No saved sensitivity results for this coverage policy and period');
    }
  } else if(s.view==='comparison') {
    note=bundle.metadata.comparison_note+' Only implemented PCA scaling/window comparisons are shown below; no structural-versus-PCA leaderboard is available.';
    if(!hasRobust) main=empty('Import robustness outputs to compare PCA specifications');
    else if(s.chart==='windows') {rows=(bundle.common_window_metrics||[]).filter(r=>r.period===s.period && r.method===s.method && r.tau===0 && r.k===s.k);main=chart('PCA window comparison on common dates',[line(rows,'window','rmse_log',`K=${s.k}`,palette[0],{mode:'lines+markers'})],'Prior window (supplied dates)','Log RMSE','Common support starts after the longest warm-up; 504-day results do not assess 2022.');}
    else {rows=(bundle.rolling_metrics||[]).filter(r=>r.window===s.window && r.period===s.period && r.tau===0); main=chart('Covariance versus correlation PCA',grouped(rows,'method','k','rmse_log'),'Retained components K','Log RMSE',`${s.window}-date prior windows · ${s.period}`);}
    lower=empty('Schwartz–Smith and PCA use different maturity targets and estimation conventions. No directly comparable empirical result has been exported.');
  } else if(s.view==='forecast') {
    const forecasts=(bundle.forecast_predictions||[]).filter(r=>r.split===s.forecastSplit&&r.horizon===s.forecastHorizon&&r.maturity_rank===s.rank);
    const metricsAll=(bundle.forecast_metrics||[]).filter(r=>r.split===s.forecastSplit&&r.horizon===s.forecastHorizon);
    const metrics=metricsAll.filter(r=>r.maturity_rank===s.rank);
    if(!bundle.metadata.forecast) return {main:empty('Saved forecast evaluation is not included in this bundle'),lower:empty('Import a bundle with precomputed forecast predictions and metrics.'),rows:[],note:'Forecasts are not fit in the browser.'};
    const synthetic=bundle.metadata.dataset_id==='synthetic';
    note=`${synthetic?'Synthetic data; these results are not historical WTI evidence. ':''}Saved ${s.forecastSplit} forecasts at ${s.forecastHorizon}-day horizon, estimated through ${bundle.metadata.forecast.training_end}. Forecasts are evaluated against later observed settlements; no fitting or prediction runs in the browser.`;
    const actual=[...new Map(forecasts.map(r=>[r.date_,r])).values()].sort((a,b)=>a.date_.localeCompare(b.date_));
    const forecastColors={random_walk:'#e4b05c',one_factor:'#b695f9',two_factor:'#5b8cff'};
    const asPrice=value=>s.log?value:Math.exp(value);
    const actualTrace=line(actual,'date_','observed_log','Observed',COLORS.observed,{mode:'lines+markers'});actualTrace.y=actualTrace.y.map(asPrice);
    const predicted=bundle.metadata.forecast.models.map(model=>{const trace=line(forecasts.filter(r=>r.model===model),'date_','predicted_log',model.replace('_',' '),forecastColors[model],{mode:'lines'});trace.y=trace.y.map(asPrice);return trace;});
    lower=chart('Forecast error by maturity',bundle.metadata.forecast.models.map(model=>line(metricsAll.filter(r=>r.model===model&&r.maturity_rank>0).map(r=>({...r,maturity:`M${r.maturity_rank}`})),'maturity','rmse_log',model.replace('_',' '),forecastColors[model],{mode:'lines+markers'})),'Maturity rank','Log RMSE',`${s.forecastSplit} split · ${s.forecastHorizon}-day horizon`);
    rows=metrics;
    if(s.chart==='predictions') main=chart('Saved forecast versus observed',[actualTrace,...predicted],'Target date',s.log?'Log price':bundle.metadata.price_unit,`M${s.rank} · ${s.forecastSplit} · ${s.forecastHorizon}-day horizon`);
    if(s.chart==='model-errors') {
      const baseline=metrics.find(r=>r.model==='random_walk')?.rmse_log;
      const selected=metrics.filter(r=>r.model&&r.maturity_rank===s.rank).map(r=>({...r,rmse_improvement_vs_random_walk_pct:baseline>0?(1-r.rmse_log/baseline)*100:0}));
      main=chart('Forecast RMSE by model',[{type:'bar',x:selected.map(r=>r.model.replace('_',' ')),y:selected.map(r=>s.log?r.rmse_log:r.rmse_price),name:'RMSE',marker:{color:selected.map(r=>forecastColors[r.model])}}],'Model',s.log?'Log RMSE':'RMSE · USD/barrel',`M${s.rank} · ${s.forecastSplit} · ${s.forecastHorizon}-day horizon`);
      rows=selected;
    }
    if(s.chart==='maturity-errors') {
      const selected=metricsAll.filter(r=>r.maturity_rank>0);
      main=chart('Forecast RMSE by maturity',bundle.metadata.forecast.models.map(model=>line(selected.filter(r=>r.model===model).map(r=>({...r,maturity:`M${r.maturity_rank}`})),'maturity',s.log?'rmse_log':'rmse_price',model.replace('_',' '),forecastColors[model],{mode:'lines+markers'})),'Maturity rank',s.log?'Log RMSE':'RMSE · USD/barrel',`${s.forecastSplit} split · ${s.forecastHorizon}-day horizon`);
      rows=selected;
    }
    if(s.chart==='horizons') {
      const selected=(bundle.forecast_metrics||[]).filter(r=>r.split===s.forecastSplit&&r.maturity_rank===s.rank);
      main=chart('Forecast error by horizon',bundle.metadata.forecast.models.map(model=>line(selected.filter(r=>r.model===model),'horizon',s.log?'rmse_log':'rmse_price',model.replace('_',' '),forecastColors[model],{mode:'lines+markers'})),'Forecast horizon (days)',s.log?'Log RMSE':'RMSE · USD/barrel',`M${s.rank} · ${s.forecastSplit} split`);
      rows=selected;
    }
  }
  return {main,lower,rows,note};
}
