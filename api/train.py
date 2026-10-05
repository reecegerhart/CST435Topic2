"""Training CLI: train one configuration, persist it, and log the audit data.

Usage (repo root, with SUPABASE_URL and SUPABASE_SERVICE_KEY in .env):

    python -m api.train --config api/configs/default.yaml
    python -m api.train --config api/configs/gelu.yaml --activate

What it does:
  1. reads the train / val / test rows from Supabase (``adult_income``),
  2. trains the MLP and evaluates it on the held-out test split,
  3. inserts a ``runs`` row (+ the model artifact) in Supabase,
  4. writes the checkpoint file to ``models/``,
  5. scores every test row and logs it to ``predictions`` with its
     ``adult_income_id`` -- this is what the /audit fairness query joins on.

Pass --activate to make this run the one /predict serves.
"""
from __future__ import annotations

import argparse
import base64
from pathlib import Path

import pandas as pd
import yaml

from api import db
from api.training import serialize_artifact, train_model
from shared.data import FEATURE_COLS, hash_features

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _load_split(split: str) -> pd.DataFrame:
    print(f"  reading {split} rows from Supabase...")
    return pd.DataFrame(db.fetch_adult(split))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--activate", action="store_true",
                        help="make this run the model /predict serves")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    print(f"Training '{config['name']}' from {args.config}")

    df_train, df_val, df_test = (_load_split(s) for s in ("train", "val", "test"))
    print(f"  rows: train={len(df_train)} val={len(df_val)} test={len(df_test)}")

    print("  training...")
    run_fields, artifact, test_proba = train_model(config, df_train, df_val, df_test)

    raw = serialize_artifact(artifact)
    run = db.insert_run(run_fields, base64.b64encode(raw).decode("ascii"))
    run_id = run["id"]

    MODELS_DIR.mkdir(exist_ok=True)
    checkpoint = MODELS_DIR / f"run_{run_id}_{config['name']}.pkl"
    checkpoint.write_bytes(raw)

    # Log every test-row prediction so the SQL audit can join to adult_income.
    records = df_test[FEATURE_COLS].to_dict("records")
    log_rows = [
        {
            "request_hash": hash_features(rec),
            "predicted_label": int(p >= 0.5),
            "predicted_proba": float(p),
            "served_by_run_id": run_id,
            "adult_income_id": int(row_id),
        }
        for rec, p, row_id in zip(records, test_proba, df_test["id"])
    ]
    db.insert_predictions(log_rows)

    if args.activate:
        db.set_active_run(run_id)

    r = run_fields
    top = sorted(r["permutation_importance"].items(), key=lambda kv: -kv[1]["mean"])[:3]
    print(f"\nRun {run_id} saved ({'active' if args.activate else 'not active'}).")
    print(f"  best epoch {r['best_epoch']}, temperature {r['temperature']:.3f}")
    print(f"  accuracy {r['accuracy']:.3f}  precision {r['precision']:.3f}  "
          f"recall {r['recall']:.3f}  f1 {r['f1']:.3f}  roc_auc {r['roc_auc']:.3f}")
    print(f"  ECE {r['ece_uncalibrated']:.4f} -> {r['ece']:.4f} after calibration")
    print("  top features: " + ", ".join(f"{k} ({v['mean']:.3f})" for k, v in top))
    print(f"  checkpoint: {checkpoint}")
    print(f"  logged {len(log_rows)} test predictions for the audit")


if __name__ == "__main__":
    main()