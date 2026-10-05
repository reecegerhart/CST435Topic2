"""Regression test: a frozen reference row must keep scoring the same probability."""
from __future__ import annotations

import base64
from pathlib import Path

from api.training import load_predictor

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_frozen_row_via_predictor(reference):
    raw = (FIXTURES / "reference_model.pkl").read_bytes()
    predictor = load_predictor(base64.b64encode(raw).decode("ascii"))
    proba = float(predictor.predict_proba([reference["row"]])[0])
    assert abs(proba - reference["expected_proba"]) < reference["tolerance"]


def test_frozen_row_via_api(client, reference):
    resp = client.post("/predict", json={"features": reference["row"]})
    assert resp.status_code == 200
    assert abs(resp.json()["proba"] - reference["expected_proba"]) < reference["tolerance"]