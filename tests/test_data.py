import numpy as np
import pandas as pd
import pytest

from commodity_factors.data import clean_futures_data, prepare_kalman_data, validate_panel


def test_nullable_prices_convert_to_float_with_configurable_ranks(small_panel):
    small_panel["settlement"] = small_panel.settlement.astype("Float64")
    small_panel["log_settlement"] = small_panel.log_settlement.astype("Float64")
    dates, Y, tau, dt = prepare_kalman_data(small_panel)
    assert Y.shape == (30, 3)
    assert Y.dtype == tau.dtype == np.dtype(float)
    np.testing.assert_allclose(np.exp(Y).ravel(), small_panel.settlement.to_numpy(dtype=float))
    np.testing.assert_allclose(dt, np.diff(dates.to_numpy()) / np.timedelta64(1, "D") / 365.25)


def test_conflicting_duplicate_is_rejected(small_panel):
    duplicate = small_panel.iloc[[0]].copy()
    duplicate["settlement"] += 1
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        clean_futures_data(pd.concat([small_panel, duplicate]))


def test_incomplete_panel_is_rejected(small_panel):
    with pytest.raises(ValueError, match="exactly"):
        validate_panel(small_panel.iloc[1:], n=3)
