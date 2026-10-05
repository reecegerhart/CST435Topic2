"""Shared pytest fixtures.

Most tests run with no cloud access: the db module is replaced by an in-memory
fake, and the model is the small frozen fixture in tests/fixtures/ (create it
once with ``python -m tests.make_fixture``). Only tests/test_supabase.py talks
to a real Supabase project.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MISSING_MSG = "Fixture missing. Create it once with: python -m tests.make_fixture"


@pytest.fixture
def reference() -> dict:
    """The frozen row and the probability it must keep scoring."""
    path = FIXTURES / "reference.json"
    if not path.exists():
        pytest.fail(MISSING_MSG)
    return json.loads(path.read_text())


@pytest.fixture
def client(monkeypatch):
    """TestClient with Supabase replaced by an in-memory fake. Run 1 is active."""
    from fastapi.testclient import TestClient

    from api import db, main

    model_path = FIXTURES / "reference_model.pkl"
    artifact = (
        base64.b64encode(model_path.read_bytes()).decode("ascii")
        if model_path.exists() else None
    )
    run_row = {
        "id": 1, "name": "fixture", "hidden_sizes": [16, 8], "activation": "relu",
        "dropout": 0.0, "lr": 0.01, "weight_decay": 0.0, "batch_size": 128,
        "epochs": 15, "best_epoch": 5, "accuracy": 0.8, "precision": 0.7,
        "recall": 0.5, "f1": 0.6, "roc_auc": 0.85, "temperature": 1.0,
        "is_active": True, "created_at": "2026-01-01T00:00:00+00:00",
    }
    store = {"runs": {1: run_row}, "artifacts": {1: artifact}, "predictions": []}

    def insert_predictions(rows):
        for row in rows:
            store["predictions"].append({"id": len(store["predictions"]) + 1, **row})
        return rows

    monkeypatch.setattr(db, "ping", lambda: True)
    monkeypatch.setattr(db, "get_active_run_id", lambda: 1)
    monkeypatch.setattr(db, "get_run", lambda rid: store["runs"].get(rid))
    monkeypatch.setattr(db, "latest_runs", lambda limit=50: list(store["runs"].values()))
    monkeypatch.setattr(db, "get_run_artifact", lambda rid: store["artifacts"].get(rid))
    monkeypatch.setattr(db, "insert_predictions", insert_predictions)
    monkeypatch.setattr(db, "get_audit", lambda run_id=None: [])

    main._predictors.clear()
    with TestClient(main.app) as c:
        c._store = store  # exposed for assertions
        yield c
    main._predictors.clear()