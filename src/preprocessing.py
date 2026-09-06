"""
preprocessing.py
-----------------
Non-destructive cleaning of the raw credit-card-clients dataset.

Every transformation here is documented in reports/data_quality_report.md.
No rows are deleted. Categorical re-grouping only touches EDUCATION and
MARRIAGE, and is limited to folding undocumented codes into an
"Other/Unknown" bucket that already existed conceptually in the source
documentation (EDUCATION=4 / MARRIAGE=3).
"""

from __future__ import annotations

import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TARGET_RAW = "default payment next month"
TARGET_CLEAN = "default_next_month"

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


def clean_education(series: pd.Series) -> pd.Series:
    """Fold undocumented codes 0, 5, 6 into 4 ('Other/Unknown')."""
    return series.replace({0: 4, 5: 4, 6: 4})


def clean_marriage(series: pd.Series) -> pd.Series:
    """Fold undocumented code 0 into 3 ('Other')."""
    return series.replace({0: 3})


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all documented, non-destructive cleaning steps.

    Returns a new dataframe; does not mutate the input.
    """
    out = df.copy()

    # Standardize target column name so downstream code never has to
    # guess between the two spellings seen in the wild.
    if TARGET_RAW in out.columns:
        out = out.rename(columns={TARGET_RAW: TARGET_CLEAN})

    n_before = len(out)

    out["EDUCATION"] = clean_education(out["EDUCATION"])
    out["MARRIAGE"] = clean_marriage(out["MARRIAGE"])

    assert len(out) == n_before, "Cleaning must not change row count (no deletions)."

    # PAY_* columns are intentionally left with their original integer
    # codes here — they are treated as ordinal/categorical downstream in
    # feature_engineering.py and modeling.py, not "fixed" at this stage.

    logger.info(
        "Cleaned dataset: %d rows (unchanged), EDUCATION/MARRIAGE undocumented "
        "codes folded into Other/Unknown buckets.",
        len(out),
    )
    return out


if __name__ == "__main__":
    from data_ingestion import load_raw_data

    raw = load_raw_data()
    cleaned = clean_dataset(raw)
    print(cleaned["EDUCATION"].value_counts().sort_index())
    print(cleaned["MARRIAGE"].value_counts().sort_index())
    out_path = "../data/processed/cleaned.csv"
    cleaned.to_csv(out_path, index=False)
    print("Saved:", out_path, cleaned.shape)
