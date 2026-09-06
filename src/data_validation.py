"""
data_validation.py
-------------------
Produces a data-quality audit of the raw credit-card-clients dataset.

Every number in the generated report is computed directly from the
dataframe passed in — nothing here is a placeholder or an assumed
value from the UCI documentation. The report exists precisely to
check whether the Kaggle/UCI file actually matches its documentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TARGET_COL_CANDIDATES = ["default payment next month", "default.payment.next.month", "default"]


def find_target_column(df: pd.DataFrame) -> str:
    for candidate in TARGET_COL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"None of the expected target column names found: {TARGET_COL_CANDIDATES}")


def audit_schema(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }


def audit_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    dup_rows = int(df.duplicated().sum())
    dup_ids = int(df["ID"].duplicated().sum()) if "ID" in df.columns else None
    return {"duplicate_rows": dup_rows, "duplicate_ids": dup_ids}


def audit_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(4)
    out = pd.DataFrame({"missing_count": missing_count, "missing_pct": missing_pct})
    return out[out["missing_count"] >= 0]  # keep all columns for completeness


def audit_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    desc = numeric_df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    return desc


def audit_categorical_values(df: pd.DataFrame) -> dict[str, dict[Any, int]]:
    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    out = {}
    for c in cat_cols:
        if c in df.columns:
            out[c] = df[c].value_counts(dropna=False).sort_index().to_dict()
    return out


def audit_pay_columns(df: pd.DataFrame) -> dict[str, dict[Any, int]]:
    pay_cols = [c for c in df.columns if c.startswith("PAY_") and not c.startswith("PAY_AMT")]
    out = {}
    for c in pay_cols:
        out[c] = df[c].value_counts(dropna=False).sort_index().to_dict()
    return out


def audit_target(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    counts = df[target_col].value_counts(dropna=False).sort_index().to_dict()
    total = len(df)
    rate = {k: round(v / total * 100, 3) for k, v in counts.items()}
    return {"counts": counts, "rate_pct": rate, "total": total}


def audit_negative_bills(df: pd.DataFrame) -> dict[str, int]:
    bill_cols = [c for c in df.columns if c.startswith("BILL_AMT")]
    return {c: int((df[c] < 0).sum()) for c in bill_cols}


def audit_id_uniqueness(df: pd.DataFrame) -> dict[str, Any]:
    if "ID" not in df.columns:
        return {}
    return {
        "n_unique_ids": int(df["ID"].nunique()),
        "n_rows": int(len(df)),
        "id_min": int(df["ID"].min()),
        "id_max": int(df["ID"].max()),
        "id_is_sequential_1_to_n": bool(
            set(df["ID"].tolist()) == set(range(1, len(df) + 1))
        ),
    }


def run_full_audit(df: pd.DataFrame) -> dict[str, Any]:
    target_col = find_target_column(df)
    audit = {
        "schema": audit_schema(df),
        "duplicates": audit_duplicates(df),
        "missing": audit_missing(df).to_dict(orient="index"),
        "numeric_summary": audit_numeric_summary(df).round(2).to_dict(orient="index"),
        "categorical_values": audit_categorical_values(df),
        "pay_columns": audit_pay_columns(df),
        "target": audit_target(df, target_col),
        "negative_bill_counts": audit_negative_bills(df),
        "id_uniqueness": audit_id_uniqueness(df),
        "target_col_name": target_col,
    }
    return audit


if __name__ == "__main__":
    from data_ingestion import load_raw_data

    df = load_raw_data()
    result = run_full_audit(df)
    import json

    print(json.dumps(result, indent=2, default=str))
