"""Pydantic request/response models: the contract between the three clouds.

The Streamlit UI never imports model or SQL code; it only sends and receives the
payloads defined here. Training happens through the CLI (``api/train.py``), so
there are no training request models any more.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """Body for POST /predict: one record keyed by the feature names in /schema."""

    run_id: Optional[int] = Field(
        None, description="Which run to score with. Defaults to the active run."
    )
    features: Dict[str, object] = Field(
        ..., description="One record, e.g. {'age': 39, 'education_num': 13, ...}."
    )


class PredictResponse(BaseModel):
    run_id: int
    label: int = Field(..., description="0 = <=50K, 1 = >50K.")
    income: str = Field(..., description="Human-readable class label.")
    proba: float = Field(..., description="Calibrated P(income > 50K).")


class BatchItem(BaseModel):
    label: int
    income: str
    proba: float


class PredictBatchResponse(BaseModel):
    run_id: int
    n_rows: int
    predictions: List[BatchItem]


# ---------------------------------------------------------------------------
# Runs (one training run = one row in the runs table; no model blob)
# ---------------------------------------------------------------------------
class Run(BaseModel):
    id: int
    name: str
    hidden_sizes: List[int]
    activation: str
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    best_epoch: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    brier: Optional[float] = None
    ece: Optional[float] = None
    ece_uncalibrated: Optional[float] = None
    temperature: float
    confusion_matrix: Optional[List[List[int]]] = None
    per_class: Optional[Dict[str, Dict[str, float]]] = None
    history: Optional[Dict[str, List[float]]] = None
    calibration: Optional[Dict[str, List[Dict[str, float]]]] = None
    permutation_importance: Optional[Dict[str, Dict[str, float]]] = None
    is_active: bool
    created_at: datetime


# ---------------------------------------------------------------------------
# Schema & audit
# ---------------------------------------------------------------------------
class SchemaResponse(BaseModel):
    """Expected feature names and dtypes, so the UI can build its form."""

    run_id: int
    numeric_features: List[str]
    categorical_features: List[str]
    dtypes: Dict[str, str] = Field(..., description="feature -> 'number' | 'category'")
    categories: Dict[str, List[str]]
    numeric_stats: Dict[str, Dict[str, float]]
    target_name: str
    target_classes: List[str]


class AuditRow(BaseModel):
    attribute: str = Field(..., description="Protected attribute, e.g. 'sex'.")
    group: str
    n: int
    positives: int
    negatives: int
    fp: int
    fn: int
    fpr: Optional[float] = Field(None, description="FP / actual <=50K.")
    fnr: Optional[float] = Field(None, description="FN / actual >50K.")
    selection_rate: float = Field(..., description="Share predicted >50K.")


class AuditResponse(BaseModel):
    run_id: int
    rows: List[AuditRow]


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------
class Health(BaseModel):
    status: str
    model_loader: bool
    supabase: bool


class Version(BaseModel):
    git_sha: str
    torch_version: str
    sklearn_version: str
    supabase_project_ref: Optional[str]