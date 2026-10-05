"""FastAPI model service -- Cloud #2 (deployed on Render.com).

Serves a trained Income-Insight model over HTTP:
  * POST /predict        score one row, log it to Supabase
  * POST /predict_batch  score an uploaded CSV, log every row
  * GET  /schema         expected feature names, dtypes, allowed values
  * GET  /audit          false-positive / false-negative rates by sex or race
  * GET  /runs           training runs (metrics and diagnostics)
  * GET  /healthz, /version

Training is NOT done here. It runs from the CLI (``python -m api.train``), which
writes the run and the fitted artifact to Supabase. This service loads the
artifact from Supabase, so nothing is stored on Render's ephemeral disk and the
model survives free-tier restarts.
"""
from __future__ import annotations

import os
import subprocess
from contextlib import asynccontextmanager
from typing import Dict, Optional

import pandas as pd
import sklearn
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api import db
from api.training import Predictor, load_predictor
from shared.data import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    NUMERIC_COLS,
    PROTECTED_COLS,
    TARGET_CLASSES,
    hash_features,
)
from shared.schemas import (
    AuditResponse,
    AuditRow,
    BatchItem,
    Health,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
    Run,
    SchemaResponse,
    Version,
)

MAX_BATCH_ROWS = 5000  # keeps a CSV upload inside the free tier's memory

# Loaded models, keyed by run_id. This is only a cache: on a miss (or after a
# restart) the artifact is re-read from Supabase.
_predictors: Dict[int, Predictor] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Warm the cache at startup so the first request is not slow."""
    try:
        run_id = db.get_active_run_id()
        if run_id is not None:
            _get_predictor(run_id)
    except Exception:  # noqa: BLE001 - never block startup on a warm-up failure
        pass
    yield


app = FastAPI(
    title="Income-Insight API",
    description="Adult-income MLP classifier with an audit trail and fairness view.",
    version="2.0.0",
    lifespan=lifespan,
)

# The UI lives on a different origin (Streamlit Cloud), so CORS must allow it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _git_sha() -> str:
    if os.environ.get("RENDER_GIT_COMMIT"):
        return os.environ["RENDER_GIT_COMMIT"][:7]
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _project_ref() -> Optional[str]:
    url = os.environ.get("SUPABASE_URL", "")
    if url.startswith("https://"):
        return url.split("//", 1)[1].split(".", 1)[0]
    return None


def _income(label: int) -> str:
    return TARGET_CLASSES[label]


def _resolve_run_id(run_id: Optional[int]) -> int:
    """Use the requested run, else the active one (read from Supabase)."""
    if run_id is not None:
        return run_id
    active = db.get_active_run_id()
    if active is None:
        raise HTTPException(
            status_code=404,
            detail="No trained model yet. Run: python -m api.train "
            "--config api/configs/default.yaml --activate",
        )
    return active


def _get_predictor(run_id: int) -> Predictor:
    if run_id not in _predictors:
        model_b64 = db.get_run_artifact(run_id)
        if model_b64 is None:
            raise HTTPException(status_code=404, detail="run_id not found")
        _predictors[run_id] = load_predictor(model_b64)
    return _predictors[run_id]


def _validate_features(features: dict) -> dict:
    """Check one record against the feature contract; 422 on any violation."""
    unknown = sorted(set(features) - set(FEATURE_COLS))
    if unknown:
        raise HTTPException(422, detail=f"Unknown feature(s): {unknown}")
    missing = [c for c in NUMERIC_COLS if features.get(c) is None]
    if missing:
        raise HTTPException(422, detail=f"Missing numeric feature(s): {missing}")
    for col in NUMERIC_COLS:
        value = features[col]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HTTPException(422, detail=f"'{col}' must be a number")
    for col in CATEGORICAL_COLS:
        value = features.get(col)
        if value is not None and not isinstance(value, str):
            raise HTTPException(422, detail=f"'{col}' must be text or null")
    return {col: features.get(col) for col in FEATURE_COLS}


def _log_predictions(run_id: int, records: list, results: list) -> None:
    """Write one predictions row per scored record (inputs are stored hashed)."""
    db.insert_predictions(
        [
            {
                "request_hash": hash_features(rec),
                "predicted_label": label,
                "predicted_proba": proba,
                "served_by_run_id": run_id,
            }
            for rec, (label, proba) in zip(records, results)
        ]
    )


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest) -> PredictResponse:
    """Score one row and log it to the predictions table."""
    features = _validate_features(req.features)
    run_id = _resolve_run_id(req.run_id)
    results = _get_predictor(run_id).predict([features])
    _log_predictions(run_id, [features], results)
    label, proba = results[0]
    return PredictResponse(run_id=run_id, label=label, income=_income(label), proba=proba)


@app.post("/predict_batch", response_model=PredictBatchResponse, tags=["prediction"])
def predict_batch(
    file: UploadFile = File(...), run_id: Optional[int] = None
) -> PredictBatchResponse:
    """Score an uploaded CSV (one row per person) and log every row."""
    try:
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, detail=f"Could not read the CSV: {exc}")
    if df.empty:
        raise HTTPException(422, detail="The CSV has no rows.")
    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(413, detail=f"Too many rows (max {MAX_BATCH_ROWS}).")

    missing_cols = [c for c in NUMERIC_COLS if c not in df.columns]
    if missing_cols:
        raise HTTPException(422, detail=f"CSV is missing column(s): {missing_cols}")

    df = df.reindex(columns=FEATURE_COLS)  # extra columns ignored; absent categoricals -> blank
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_rows = df.index[df[NUMERIC_COLS].isna().any(axis=1)].tolist()
    if bad_rows:
        shown = [i + 2 for i in bad_rows[:5]]  # +2: header line + 1-based rows
        raise HTTPException(
            422, detail=f"Blank or non-numeric number(s) on CSV line(s) {shown}"
        )

    records = df.astype(object).where(df.notna(), None).to_dict("records")
    rid = _resolve_run_id(run_id)  # reads Supabase on every batch request
    results = _get_predictor(rid).predict(records)
    _log_predictions(rid, records, results)
    items = [BatchItem(label=l, income=_income(l), proba=p) for l, p in results]
    return PredictBatchResponse(run_id=rid, n_rows=len(items), predictions=items)


# ---------------------------------------------------------------------------
# schema, runs, audit
# ---------------------------------------------------------------------------
@app.get("/schema", response_model=SchemaResponse, tags=["meta"])
def schema(run_id: Optional[int] = None) -> SchemaResponse:
    """Feature names, dtypes, allowed categories, numeric ranges."""
    rid = _resolve_run_id(run_id)
    info = _get_predictor(rid).schema_info()
    dtypes = {c: "number" for c in NUMERIC_COLS}
    dtypes.update({c: "category" for c in CATEGORICAL_COLS})
    return SchemaResponse(run_id=rid, dtypes=dtypes, **info)


@app.get("/runs", response_model=list[Run], tags=["runs"])
def list_runs() -> list[Run]:
    """The latest 50 training runs."""
    return [Run(**row) for row in db.latest_runs(limit=50)]


@app.get("/runs/{run_id}", response_model=Run, tags=["runs"])
def get_run(run_id: int) -> Run:
    row = db.get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return Run(**row)


@app.get("/audit", response_model=AuditResponse, tags=["meta"])
def audit(run_id: Optional[int] = None, by: Optional[str] = None) -> AuditResponse:
    """False-positive and false-negative rates by protected attribute.

    Computed in SQL (the ``audit_rates`` view) by joining the logged test-set
    predictions to ``adult_income``. ``by`` filters to 'sex' or 'race'.
    """
    if by is not None and by not in PROTECTED_COLS:
        raise HTTPException(422, detail=f"'by' must be one of {PROTECTED_COLS}")
    rid = _resolve_run_id(run_id)
    rows = [r for r in db.get_audit(rid) if by is None or r["attribute"] == by]
    return AuditResponse(
        run_id=rid,
        rows=[
            AuditRow(
                attribute=r["attribute"],
                group=r["grp"],
                n=r["n"],
                positives=r["positives"],
                negatives=r["negatives"],
                fp=r["fp"],
                fn=r["fn"],
                fpr=r["fpr"],
                fnr=r["fnr"],
                selection_rate=r["selection_rate"],
            )
            for r in sorted(rows, key=lambda r: (r["attribute"], r["grp"]))
        ],
    )


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------
@app.get("/healthz", response_model=Health, tags=["ops"])
def healthz() -> Health:
    """200 when the model loader (torch) and the Supabase client are reachable."""
    supabase_ok = db.ping()
    model_ok = torch.tensor([1.0]).sum().item() == 1.0
    status = "ok" if (supabase_ok and model_ok) else "degraded"
    return Health(status=status, model_loader=model_ok, supabase=supabase_ok)


@app.get("/version", response_model=Version, tags=["ops"])
def version() -> Version:
    return Version(
        git_sha=_git_sha(),
        torch_version=torch.__version__,
        sklearn_version=sklearn.__version__,
        supabase_project_ref=_project_ref(),
    )