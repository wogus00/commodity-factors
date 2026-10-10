from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def app():
    demo = APP.parent / "data/synthetic/demo/metadata.json"
    if not demo.exists():
        pytest.skip("Generate the saved demo with scripts/run_synthetic.py")
    result = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not result.exception
    result.sidebar.selectbox[0].select("Synthetic oil futures").run()
    assert not result.exception
    return result


@pytest.mark.parametrize("view", ["Market overview", "Forecast comparison", "Performance", "Model exploration", "Daily forecast"])
def test_dashboard_views_and_horizon_controls(app, view):
    app.radio[0].set_value(view).run()
    assert not app.exception
    app.sidebar.selectbox[2].select(5).run()
    assert not app.exception
    app.sidebar.selectbox[1].select("validation").run()
    assert not app.exception


def test_empty_model_selection_has_a_useful_message(app):
    app.sidebar.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any("Select at least one model" in message.value for message in app.info)


def test_daily_forecast_replays_selected_date_and_skips_weekend(app):
    from datetime import date

    app.radio[0].set_value("Daily forecast").run()
    app.date_input[0].set_value(date(2025, 10, 7)).run()
    assert not app.exception
    assert any("Observed · 2025-10-07" == metric.label for metric in app.metric)
    assert any("Next forecast · 2025-10-08" in value.value for value in app.markdown)
    app.date_input[0].set_value(date(2025, 10, 11)).run()
    assert not app.exception
    assert any("Observed · 2025-10-10" == metric.label for metric in app.metric)
