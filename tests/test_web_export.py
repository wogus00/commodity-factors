import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("web_export", ROOT / "scripts/export_web_bundle.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def test_browser_export_preserves_dates_and_metrics(tmp_path):
    output = exporter.export_bundle(ROOT / "data/synthetic/demo", tmp_path / "demo.json")
    payload = json.loads(output.read_text())
    assert payload["format"] == "commodity-factors-web"
    assert payload["metadata"]["dataset_id"] == "synthetic"
    predictions = pd.DataFrame(**payload["tables"]["predictions"])
    assert (pd.to_datetime(predictions.origin_date) < pd.to_datetime(predictions.date_)).all()
    assert set(predictions.model) == {"random_walk", "one_factor", "two_factor"}
    assert set(payload["tables"]) == {"panel", "predictions", "reconstructions", "states", "metrics"}


def test_empirical_quotes_cannot_be_exported_into_the_public_tree(monkeypatch):
    bundle = exporter.load_bundle(ROOT / "data/synthetic/demo")
    bundle.metadata["dataset_id"] = "wti"
    monkeypatch.setattr(exporter, "load_bundle", lambda directory: bundle)
    with pytest.raises(ValueError, match="outside web"):
        exporter.export_bundle("unused", ROOT / "web/data/blocked.json")
