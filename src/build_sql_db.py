"""
build_sql_db.py
----------------
Loads the cleaned+scored dataset into a lightweight SQLite database so
the project can demonstrate real SQL analytics, without standing up a
full data warehouse for a 30K-row dataset (that would be architectural
overkill — see README for the ETL vs ELT discussion).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "credit_risk.db"


def build_database():
    featured = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "featured.csv")
    scored = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "scored_test_set.csv")

    conn = sqlite3.connect(DB_PATH)
    featured.to_sql("customers", conn, if_exists="replace", index=False)
    scored.to_sql("scored_customers", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Built {DB_PATH} with tables: customers ({len(featured)} rows), "
          f"scored_customers ({len(scored)} rows)")


if __name__ == "__main__":
    build_database()
