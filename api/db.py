"""Supabase persistence helpers for the API tier.

All Supabase access for the model service goes through this module. The
Streamlit UI never imports it; it makes its own read-only queries with the anon
key (runs table and the audit_rates view only).

The fitted model artifact lives in ``run_artifacts`` (no anon policy) so the
large base64 blob is never exposed to the public key.

Environment variables (.env locally, Render dashboard in production):
    SUPABASE_URL              -> https://<project-ref>.supabase.co
    SUPABASE_SERVICE_KEY      -> service-role key (server-side only, secret!)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_client: Optional[Client] = None

# PostgREST returns at most 1000 rows per request, so bulk reads/writes page.
_PAGE = 1000


def get_client() -> Client:
    """Lazily create and cache a Supabase client."""
    global _client
    if _client is None:
        _client = create_client(
            os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"]
        )
    return _client


def ping() -> bool:
    """True if Supabase is reachable and the adult_income table exists."""
    try:
        get_client().table("adult_income").select("id").limit(1).execute()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# adult_income (training data)
# ---------------------------------------------------------------------------
def count_adult_rows() -> int:
    resp = (
        get_client()
        .table("adult_income")
        .select("id", count="exact")
        .limit(1)
        .execute()
    )
    return resp.count or 0


def insert_adult_rows(rows: List[dict]) -> int:
    """Insert rows in chunks; returns the number inserted."""
    for start in range(0, len(rows), _PAGE):
        get_client().table("adult_income").insert(rows[start : start + _PAGE]).execute()
    return len(rows)


def fetch_adult(split: Optional[str] = None) -> List[dict]:
    """Fetch all rows (optionally one split: 'train' | 'val' | 'test'), paged."""
    rows: List[dict] = []
    start = 0
    while True:
        q = get_client().table("adult_income").select("*")
        if split is not None:
            q = q.eq("split", split)
        page = q.order("id").range(start, start + _PAGE - 1).execute().data
        rows.extend(page)
        if len(page) < _PAGE:
            break
        start += _PAGE
    return rows


# ---------------------------------------------------------------------------
# runs (metrics + diagnostics)  and  run_artifacts (model blob)
# ---------------------------------------------------------------------------
def insert_run(run: dict, model_b64: str) -> dict:
    """Insert a runs row (a dict of column -> value) plus its artifact blob."""
    client = get_client()
    row = client.table("runs").insert(run).execute().data[0]
    client.table("run_artifacts").insert(
        {"run_id": row["id"], "model_b64": model_b64}
    ).execute()
    return row


def get_run(run_id: int) -> Optional[dict]:
    resp = get_client().table("runs").select("*").eq("id", run_id).limit(1).execute()
    return resp.data[0] if resp.data else None


def latest_runs(limit: int = 50) -> List[dict]:
    resp = (
        get_client()
        .table("runs")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data


def get_run_artifact(run_id: int) -> Optional[str]:
    resp = (
        get_client()
        .table("run_artifacts")
        .select("model_b64")
        .eq("run_id", run_id)
        .limit(1)
        .execute()
    )
    return resp.data[0]["model_b64"] if resp.data else None


def set_active_run(run_id: int) -> None:
    """Mark one run as the model that /predict serves."""
    client = get_client()
    client.table("runs").update({"is_active": False}).eq("is_active", True).execute()
    client.table("runs").update({"is_active": True}).eq("id", run_id).execute()


def get_active_run_id() -> Optional[int]:
    """The active run's id, falling back to the most recent run."""
    client = get_client()
    resp = client.table("runs").select("id").eq("is_active", True).limit(1).execute()
    if resp.data:
        return resp.data[0]["id"]
    resp = client.table("runs").select("id").order("created_at", desc=True).limit(1).execute()
    return resp.data[0]["id"] if resp.data else None


# ---------------------------------------------------------------------------
# predictions (the audit log)
# ---------------------------------------------------------------------------
def insert_predictions(rows: List[dict]) -> List[dict]:
    """Bulk-insert prediction rows. Each dict needs request_hash,
    predicted_label, predicted_proba, served_by_run_id, and optionally
    adult_income_id."""
    out: List[dict] = []
    for start in range(0, len(rows), _PAGE):
        out.extend(
            get_client().table("predictions").insert(rows[start : start + _PAGE]).execute().data
        )
    return out


def get_prediction_by_hash(request_hash: str) -> Optional[dict]:
    """Most recent prediction with this hash (used by the Supabase test)."""
    resp = (
        get_client()
        .table("predictions")
        .select("*")
        .eq("request_hash", request_hash)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return resp.data[0] if resp.data else None


def get_audit(run_id: Optional[int] = None) -> List[dict]:
    """FPR / FNR / selection rate by protected attribute, from the audit_rates view."""
    q = get_client().table("audit_rates").select("*")
    if run_id is not None:
        q = q.eq("run_id", run_id)
    return q.execute().data