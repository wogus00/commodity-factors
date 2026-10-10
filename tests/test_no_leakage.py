import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult

from commodity_factors import analysis


def test_future_prices_do_not_change_forecasts_from_an_earlier_origin(small_panel, model_params):
    one, two = model_params
    baseline = analysis.compare_models(small_panel, one, two, 0.004, horizons=(1, 3))
    target = pd.Timestamp("2025-01-08")
    changed = small_panel.copy()
    mask = changed.date_ >= target
    changed.loc[mask, "settlement"] *= 1.2
    changed.loc[mask, "log_settlement"] = np.log(changed.loc[mask, "settlement"])
    perturbed = analysis.compare_models(changed, one, two, 0.004, horizons=(1, 3))
    mask = baseline.predictions.origin_date < target
    np.testing.assert_allclose(baseline.predictions.loc[mask, "predicted_log"],
                               perturbed.predictions.loc[mask, "predicted_log"])
    # Confirm the observation change actually influences subsequent filtering.
    assert not np.allclose(baseline.states.chi, perturbed.states.chi)


def test_calibration_receives_only_training_observations(monkeypatch, small_panel, model_params):
    one, two = model_params
    seen = []
    result = OptimizeResult(nit=1, nfev=1)

    def fake_one(Y, tau, dt, **kwargs):
        seen.append(Y.copy())
        return dict(params=one, success=True, message="test fit", log_likelihood=1.0, optimizer_result=result)

    def fake_two(Y, tau, dt, **kwargs):
        seen.append(Y.copy())
        assert kwargs["fixed_lambda_chi"] == 0.0
        return dict(params=two, sigma_epsilon=0.004, success=True, message="test fit",
                    log_likelihood=2.0, optimizer_result=result)

    monkeypatch.setattr(analysis, "estimate_one_factor", fake_one)
    monkeypatch.setattr(analysis, "estimate_real_parameters", fake_two)
    baseline = analysis.run_analysis(small_panel, horizons=(1, 3))
    changed = small_panel.copy()
    mask = changed.date_ >= pd.Timestamp("2024-01-01")
    changed.loc[mask, "settlement"] *= 1.3
    changed.loc[mask, "log_settlement"] = np.log(changed.loc[mask, "settlement"])
    analysis.run_analysis(changed, horizons=(1, 3))
    assert len(seen) == 6
    for original, altered in zip(seen[:3], seen[3:]):
        assert original.shape == (10, 3)
        np.testing.assert_array_equal(original, altered)
    assert (baseline.predictions.origin_date >= pd.Timestamp(baseline.metadata["train_end"])).all()
