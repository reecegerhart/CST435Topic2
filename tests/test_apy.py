"""API tests: schema validation, batch row count, ops endpoints."""
from __future__ import annotations

import io

import pandas as pd


# ---- ops -------------------------------------------------------------------
def test_healthz_ok(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["model_loader"] is True and body["supabase"] is True


def test_version_reports_frameworks(client):
    body = client.get("/version").json()
    assert "torch_version" in body and "sklearn_version" in body


def test_schema_lists_features_and_dtypes(client, reference):
    body = client.get("/schema").json()
    assert "age" in body["numeric_features"]
    assert "occupation" in body["categorical_features"]
    assert body["dtypes"]["age"] == "number" and body["dtypes"]["workclass"] == "category"
    assert "sex" not in body["dtypes"]  # protected attributes are not model inputs
    assert body["target_classes"] == ["<=50K", ">50K"]


# ---- /predict schema validation ---------------------------------------------
def test_predict_rejects_missing_features(client):
    assert client.post("/predict", json={"features": {"age": 30}}).status_code == 422


def test_predict_rejects_missing_body_field(client):
    assert client.post("/predict", json={"run_id": 1}).status_code == 422


def test_predict_rejects_wrong_type(client, reference):
    bad = {**reference["row"], "age": "forty"}
    assert client.post("/predict", json={"features": bad}).status_code == 422


def test_predict_rejects_unknown_feature(client, reference):
    bad = {**reference["row"], "sex": "Female"}  # protected attrs are not inputs
    assert client.post("/predict", json={"features": bad}).status_code == 422


def test_predict_valid_row_is_scored_and_logged(client, reference):
    resp = client.post("/predict", json={"features": reference["row"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in (0, 1) and 0.0 <= body["proba"] <= 1.0
    logged = client._store["predictions"]
    assert len(logged) == 1 and logged[0]["served_by_run_id"] == body["run_id"]
    assert len(logged[0]["request_hash"]) == 64  # inputs are stored hashed


# ---- /predict_batch ---------------------------------------------------------
def _csv_file(rows: list[dict]):
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return {"file": ("rows.csv", buf.getvalue().encode(), "text/csv")}


def test_batch_returns_same_row_count(client, reference):
    rows = [{**reference["row"], "age": 20 + i} for i in range(7)]
    resp = client.post("/predict_batch", files=_csv_file(rows))
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_rows"] == 7 and len(body["predictions"]) == 7
    assert len(client._store["predictions"]) == 7  # every row logged


def test_batch_rejects_missing_column(client, reference):
    rows = [{k: v for k, v in reference["row"].items() if k != "age"}]
    assert client.post("/predict_batch", files=_csv_file(rows)).status_code == 422


def test_audit_rejects_bad_attribute(client):
    assert client.get("/audit", params={"by": "not_an_attribute"}).status_code == 422