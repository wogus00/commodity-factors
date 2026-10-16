import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('modeling_export', ROOT/'scripts/export_modeling_bundle.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def test_export_is_reproducible_and_includes_saved_forecast_outputs(tmp_path):
    a = exporter.export_modeling_bundle(ROOT/'data/synthetic/demo', tmp_path/'a.json')
    b = exporter.export_modeling_bundle(ROOT/'data/synthetic/demo', tmp_path/'b.json')
    assert a.read_bytes() == b.read_bytes()
    payload = json.loads(a.read_text())
    assert set(payload['tables']) == {'panel','states','ss_reconstructions','ss_loadings','forecast_predictions','forecast_metrics'}
    assert 'daily_forecast' not in payload['metadata']
    assert payload['metadata']['forecast']['horizons'] == [1,5,20]
    reconstruction = pd.DataFrame(**payload['tables']['ss_reconstructions'])
    assert ((reconstruction.observed_log - reconstruction.fitted_log - reconstruction.residual_log).abs() < 1e-10).all()
    assert payload['metadata']['state_estimate'] == 'filtered'
    assert not payload['metadata']['standardized_innovations_available']
    forecast = pd.DataFrame(**payload['tables']['forecast_predictions'])
    forecast[['date_', 'origin_date']] = forecast[['date_', 'origin_date']].apply(pd.to_datetime)
    expected = pd.read_parquet(ROOT/'data/synthetic/demo/predictions.parquet')
    pd.testing.assert_frame_equal(forecast, expected[['date_','origin_date','horizon','split','model','maturity_rank','tau','observed_log','predicted_log','error_log']], check_exact=False, atol=1e-11)
    metrics = pd.DataFrame(**payload['tables']['forecast_metrics'])
    expected_metrics = pd.read_parquet(ROOT/'data/synthetic/demo/metrics.parquet')
    pd.testing.assert_frame_equal(metrics, expected_metrics[['split','horizon','model','maturity_rank','rmse_log','mae_log','rmse_price','mae_price','n','n_dates']], check_exact=False, atol=1e-11)


def test_export_reads_precomputed_forecasts_without_model_fitting(monkeypatch, tmp_path):
    original = pd.read_parquet
    read_names = []
    def checked(path, *args, **kwargs):
        read_names.append(Path(path).name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd,'read_parquet',checked)
    exporter.export_modeling_bundle(ROOT/'data/synthetic/demo', tmp_path/'result.json')
    assert {'predictions.parquet','metrics.parquet'} <= set(read_names)


def test_empirical_export_guard_is_preserved(tmp_path):
    source = tmp_path/'empirical';source.mkdir()
    (source/'metadata.json').write_text('{"dataset_id":"wti"}')
    with pytest.raises(ValueError, match='outside web'):
        exporter.export_modeling_bundle(source, ROOT/'web/data/should-not-exist.json')


def test_local_historical_modeling_tables_match_saved_results(tmp_path):
    source = ROOT/'results/pca/wti/robustness'
    if not source.exists():
        pytest.skip('Local empirical outputs are not checked into the repository')
    output = exporter.export_modeling_bundle(ROOT/'results/wti',tmp_path/'wti.json',
        ROOT/'results/pca/wti',source,ROOT/'results/pca/wti_coverage50/robustness')
    data = json.loads(output.read_text())
    actual = pd.DataFrame(**data['tables']['rolling_metrics'])
    expected = pd.read_parquet(source/'rolling_metrics.parquet')
    pd.testing.assert_frame_equal(actual,expected,check_exact=False,atol=1e-11)
    assert data['metadata']['robustness']['unscored_scenarios'] == ['missing_quotes']
