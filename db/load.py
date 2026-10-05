"""One-shot loader: UCI Adult Income -> Supabase ``adult_income`` table.

Usage (repo root, with SUPABASE_URL and SUPABASE_SERVICE_KEY in .env):

    python -m db.load

Apply db/migrations/001_init.sql in the Supabase SQL Editor first. This script
only loads data. It downloads Adult from OpenML (~48.8k rows), keeps the model
features plus sex/race for auditing, stores missing values as NULL, and assigns
a stratified 70/15/15 train/val/test split with a fixed seed.

Run it once. If the table already has rows it stops rather than duplicating them.
To reload, run ``truncate adult_income cascade;`` in the SQL Editor (this also
clears predictions).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from api import db
from shared.data import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    NUMERIC_COLS,
    PROTECTED_COLS,
    TARGET_NAME,
)

load_dotenv()

SEED = 42


def load_adult() -> pd.DataFrame:
    """Download Adult and return a clean frame: features + protected + income."""
    raw = fetch_openml("adult", version=2, as_frame=True).frame
    raw.columns = [c.replace("-", "_") for c in raw.columns]
    df = raw.rename(columns={"class": TARGET_NAME})

    df[TARGET_NAME] = (df[TARGET_NAME].astype(str).str.strip() == ">50K").astype(int)
    df = df[FEATURE_COLS + PROTECTED_COLS + [TARGET_NAME]].copy()

    for col in NUMERIC_COLS:
        df[col] = df[col].astype(int)
    for col in CATEGORICAL_COLS + PROTECTED_COLS:
        # category -> object, trim whitespace; missing values stay NaN -> NULL
        df[col] = df[col].astype(object).map(
            lambda v: v.strip() if isinstance(v, str) else v
        )
    return df.reset_index(drop=True)


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    """Stratified 70/15/15 train/val/test split, reproducible via SEED."""
    idx = np.arange(len(df))
    y = df[TARGET_NAME].to_numpy()
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, stratify=y, random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=SEED
    )
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def main() -> None:
    existing = db.count_adult_rows()
    if existing > 0:
        raise SystemExit(
            f"adult_income already has {existing} rows. "
            "Run `truncate adult_income cascade;` in the SQL Editor to reload."
        )

    df = add_split(load_adult())
    assert len(df) >= 30_000, f"expected >=30k rows, got {len(df)}"

    # to_json turns NaN into null and numpy ints into plain JSON numbers.
    rows = json.loads(df.to_json(orient="records"))
    n = db.insert_adult_rows(rows)

    print(f"Loaded {n} rows into adult_income.")
    print(f"Positive rate (>50K): {df[TARGET_NAME].mean():.3f}")
    print(df["split"].value_counts().to_string())
    print("Missing values per column:")
    print(df.isna().sum()[df.isna().sum() > 0].to_string())


if __name__ == "__main__":
    main()