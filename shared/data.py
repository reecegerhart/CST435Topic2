"""Feature contract for the Adult Income model, shared by all three tiers.

The API's sklearn pipeline, the load script, and the UI form all read these
lists, so the feature set is defined in exactly one place. The data itself is
the real UCI Adult Income dataset, loaded into Supabase by ``db/load.py``.

``sex`` and ``race`` are PROTECTED attributes: they are stored so the model can
be audited, but they are deliberately not model inputs. Note that other
features (notably ``relationship`` and ``marital_status``) can act as proxies
for sex, which is worth discussing in the fairness reflection.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, List

NUMERIC_COLS: List[str] = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]
CATEGORICAL_COLS: List[str] = [
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "native_country",
]
FEATURE_COLS: List[str] = NUMERIC_COLS + CATEGORICAL_COLS

PROTECTED_COLS: List[str] = ["sex", "race"]

TARGET_NAME = "income"
TARGET_CLASSES = ["<=50K", ">50K"]  # index 0 / 1


def _normalize(value: object) -> object:
    """Make equivalent inputs hash identically (39 vs 39.0, NaN vs None)."""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            return int(value)
    return value


def hash_features(features: Dict[str, object]) -> str:
    """SHA-256 of the model features, used as ``request_hash`` in predictions.

    Only FEATURE_COLS are hashed, in a fixed order, so the same input always
    gives the same hash and raw values are never stored in the audit log.
    """
    canonical = {c: _normalize(features.get(c)) for c in FEATURE_COLS}
    blob = json.dumps(canonical, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()