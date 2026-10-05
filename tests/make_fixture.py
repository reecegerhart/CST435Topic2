"""Create the frozen reference model used by the regression test.

Run ONCE from the repo root, then commit the two files it writes:

    python -m tests.make_fixture

It trains a tiny MLP on a small made-up dataset (no Supabase needed), pickles the
artifact to tests/fixtures/reference_model.pkl, and records the probability it
gives one frozen row in tests/fixtures/reference.json. The regression test later
checks that this row still scores the same probability, within 1e-3.

Re-run it only when you change the model code on purpose, and commit the result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from api.training import serialize_artifact, train_model
from shared.data import TARGET_NAME

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# One frozen row. Do not edit unless you regenerate the fixture.
FROZEN_ROW = {
    "age": 42, "education_num": 13, "capital_gain": 0, "capital_loss": 0,
    "hours_per_week": 45, "workclass": "Private", "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty", "relationship": "Husband", "native_country": "United-States",
}

CONFIG = {
    "name": "fixture", "hidden_sizes": [16, 8], "activation": "relu", "dropout": 0.0,
    "lr": 0.01, "weight_decay": 0.0, "batch_size": 128, "epochs": 15, "patience": 5, "seed": 0,
}


def make_data(n: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.integers(17, 80, n),
        "education_num": rng.integers(1, 17, n),
        "capital_gain": np.where(rng.random(n) < 0.1, rng.integers(500, 10000, n), 0),
        "capital_loss": np.zeros(n, dtype=int),
        "hours_per_week": rng.integers(20, 60, n),
        "workclass": rng.choice(["Private", "Self-emp-not-inc", "Local-gov", None], n),
        "marital_status": rng.choice(["Married-civ-spouse", "Never-married", "Divorced"], n),
        "occupation": rng.choice(["Prof-specialty", "Sales", "Craft-repair", None], n),
        "relationship": rng.choice(["Husband", "Not-in-family", "Own-child"], n),
        "native_country": rng.choice(["United-States", "Mexico", None], n),
    })
    z = (0.04 * (df.age - 40) + 0.3 * (df.education_num - 10) + 0.03 * (df.hours_per_week - 40)
         + 0.0004 * df.capital_gain + 0.8 * (df.marital_status == "Married-civ-spouse") - 1.0)
    df[TARGET_NAME] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


def main() -> None:
    df = make_data()
    train, val, test = df.iloc[:2000], df.iloc[2000:2500], df.iloc[2500:]
    _, artifact, _ = train_model(CONFIG, train, val, test)

    import base64
    from api.training import load_predictor

    raw = serialize_artifact(artifact)
    predictor = load_predictor(base64.b64encode(raw).decode("ascii"))
    proba = float(predictor.predict_proba([FROZEN_ROW])[0])

    FIXTURES.mkdir(exist_ok=True)
    (FIXTURES / "reference_model.pkl").write_bytes(raw)
    (FIXTURES / "reference.json").write_text(
        json.dumps({"row": FROZEN_ROW, "expected_proba": proba, "tolerance": 1e-3}, indent=2)
    )
    print(f"Wrote fixtures. Frozen row probability = {proba:.6f}")


if __name__ == "__main__":
    main()