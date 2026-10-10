import numpy as np
import pandas as pd

from commodity_factors.analysis import TABLES, compare_models, load_bundle, save_bundle
from commodity_factors.data import prepare_kalman_data
from commodity_factors.kalman import kalman_filter
from commodity_factors.one_factor import one_factor_kalman_filter


def test_all_models_share_targets_and_random_walk_repeats_origin(small_panel, model_params):
    bundle = compare_models(small_panel, *model_params, 0.004, horizons=(1, 3))
    forecasts = bundle.predictions
    assert (forecasts.groupby(["date_", "maturity_rank", "horizon"]).model.nunique() == 3).all()
    random_walk = forecasts[forecasts.model == "random_walk"]
    prices = small_panel.set_index(["date_", "maturity_rank"]).log_settlement
    expected = prices.reindex(pd.MultiIndex.from_arrays([random_walk.origin_date, random_walk.maturity_rank]))
    np.testing.assert_allclose(random_walk.predicted_log, expected)
    assert set(forecasts.split) == {"validation", "test"}


def test_one_day_forecasts_match_filter_prior_predictions(small_panel, model_params):
    one, two = model_params
    bundle = compare_models(small_panel, one, two, 0.004, horizons=(1,))
    dates, Y, tau, dt = prepare_kalman_data(small_panel)
    one_prior = one_factor_kalman_filter(Y, tau, dt, one)["predicted_prices"]
    two_prior = kalman_filter(Y, tau, two, dt=dt, sigma_epsilon=0.004)["predicted_prices"]
    n_train = (dates < pd.Timestamp("2024-01-01")).sum()
    for model, prior in [("one_factor", one_prior), ("two_factor", two_prior)]:
        values = bundle.predictions.query("model == @model").predicted_log.to_numpy()
        np.testing.assert_allclose(values, prior[n_train:].ravel())


def test_bundle_round_trip_preserves_metrics_and_metadata(tmp_path, small_panel, model_params):
    bundle = compare_models(small_panel, *model_params, 0.004, horizons=(1, 3))
    save_bundle(bundle, tmp_path)
    restored = load_bundle(tmp_path)
    assert restored.metadata == bundle.metadata
    for table in TABLES:
        pd.testing.assert_frame_equal(getattr(bundle, table), getattr(restored, table))
    test = bundle.predictions.query("split == 'test' and horizon == 1 and model == 'two_factor'")
    metric = bundle.metrics.query("split == 'test' and horizon == 1 and model == 'two_factor' and maturity_rank == 0").iloc[0]
    np.testing.assert_allclose(metric.rmse_log, np.sqrt(np.mean(test.error_log ** 2)))
