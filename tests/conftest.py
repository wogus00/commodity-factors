import pytest

from commodity_factors.demo import DEMO_PARAMS, make_demo_panel


@pytest.fixture
def small_panel():
    panel = make_demo_panel(n_ranks=3)
    dates = panel[["date_"]].drop_duplicates()
    chosen = dates.groupby(dates.date_.dt.year).head(10).date_
    return panel[panel.date_.isin(chosen)].reset_index(drop=True)


@pytest.fixture
def model_params():
    return ({"mu": 0.02, "sigma": 0.3, "delta": -0.08, "sigma_epsilon": 0.01},
            DEMO_PARAMS.copy())
