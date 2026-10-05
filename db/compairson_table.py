"""Generate the README configuration-comparison table from SQL.

    python -m db.comparison_table            # print the markdown table
    python -m db.comparison_table --write    # also update README.md in place

Reads the ``run_comparison`` view (db/migrations/002_run_comparison.sql). With
--write, the table replaces whatever sits between these two markers in README.md:

    <!-- RUN_COMPARISON_START -->
    <!-- RUN_COMPARISON_END -->
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from api import db

README = Path(__file__).resolve().parent.parent / "README.md"
START, END = "<!-- RUN_COMPARISON_START -->", "<!-- RUN_COMPARISON_END -->"

COLUMNS = [
    ("run_id", "Run"), ("name", "Config"), ("hidden_sizes", "Hidden sizes"),
    ("activation", "Activation"), ("dropout", "Dropout"), ("best_epoch", "Best epoch"),
    ("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
    ("f1", "F1"), ("roc_auc", "ROC-AUC"), ("ece", "ECE"),
]


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, list):
        return "/".join(str(v) for v in value)
    return str(value)


def build_table() -> str:
    rows = db.get_client().table("run_comparison").select("*").order("run_id").execute().data
    header = "| " + " | ".join(label for _, label in COLUMNS) + " |"
    divider = "|" + "|".join("---" for _ in COLUMNS) + "|"
    body = ["| " + " | ".join(_fmt(r[key]) for key, _ in COLUMNS) + " |" for r in rows]
    return "\n".join([header, divider, *body])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="update README.md in place")
    table = build_table()
    print(table)
    if parser.parse_args().write:
        text = README.read_text(encoding="utf-8")
        if START not in text or END not in text:
            raise SystemExit(f"Add the {START} / {END} markers to README.md first.")
        pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
        README.write_text(pattern.sub(f"{START}\n{table}\n{END}", text), encoding="utf-8")
        print("\nREADME.md updated.")


if __name__ == "__main__":
    main()