"""Live Supabase test: /predict must write a row to the predictions table.

Skipped unless SUPABASE_URL and SUPABASE_SERVICE_KEY are set (a .env works).
It needs at least one trained, active run in that project
(python -m api.train --config api/configs/default.yaml --activate).

    pytest tests/test_supabase.py -v
"""
from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.skipif(
    not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY")),
    reason="No live Supabase credentials in the environment.",
)

ROW = {
    "age": 37, "education_num": 10, "capital_gain": 0, "capital_loss": 0,
    "hours_per_week": 40, "workclass": "Private", "marital_status": "Never-married",
    "occupation": "Sales", "relationship": "Not-in-family", "native_country": "United-States",
}


def test_predict_writes_a_predictions_row():
    from fastapi.testclient import TestClient

    from api import db
    from api.main import app
    from shared.data import hash_features

    assert db.get_active_run_id() is not None, "Train and --activate a run first."
    request_hash = hash_features(ROW)
    before = db.get_prediction_by_hash(request_hash)

    with TestClient(app) as client:
        resp = client.post("/predict", json={"features": ROW})
    assert resp.status_code == 200
    body = resp.json()

    after = db.get_prediction_by_hash(request_hash)
    assert after is not None
    assert before is None or after["id"] != before["id"]  # a NEW row was written
    assert after["served_by_run_id"] == body["run_id"]
    assert after["predicted_label"] == body["label"]
    assert abs(after["predicted_proba"] - body["proba"]) < 1e-9