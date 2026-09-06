"""
scoring.py
----------
Applies the trained, calibrated model to new customer records and
returns PD, risk band, and predicted class. This is the only module
the Streamlit dashboard and any future batch job should import for
generating predictions — it is the single source of truth for how a
raw customer record becomes a PD.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from feature_engineering import engineer_features
from preprocessing import clean_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def load_scoring_model():
    return joblib.load(MODELS_DIR / "calibrated_model.joblib")


def score_customers(raw_df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """
    raw_df: dataframe with the same raw schema as the original dataset
    (ID, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0..PAY_6,
    BILL_AMT1..6, PAY_AMT1..6). Target column is not required.
    """
    model = load_scoring_model()
    cleaned = clean_dataset(raw_df)
    featured = engineer_features(cleaned)

    X = featured.drop(columns=[c for c in ["default_next_month"] if c in featured.columns])
    proba = model.predict_proba(X)[:, 1]

    out = featured.copy()
    out["predicted_pd"] = proba
    out["risk_band"] = pd.cut(
        out["predicted_pd"],
        bins=[0, thresholds["low_max"], thresholds["moderate_max"], thresholds["high_max"], 1.0],
        labels=["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"],
        include_lowest=True,
    )
    out["predicted_class"] = (out["predicted_pd"] >= 0.5).astype(int)
    return out


if __name__ == "__main__":
    import json

    from data_ingestion import load_raw_data

    with open(PROJECT_ROOT / "reports" / "experiment_results.json") as f:
        results = json.load(f)
    thresholds = results["segmentation_thresholds"]

    sample = load_raw_data().head(5)
    scored = score_customers(sample, thresholds)
    print(scored[["ID", "predicted_pd", "risk_band", "predicted_class"]])
