"""
shap_explain.py
----------------
Computes real SHAP values for the selected model (XGBoost) on a sample
of the test set, and saves both a global summary plot and a per-customer
lookup table the dashboard can use for local explanations.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    with open(PROJECT_ROOT / "reports" / "experiment_results.json") as f:
        results = json.load(f)
    best_name = results["best_model"]

    model_pipe = joblib.load(MODELS_DIR / f"{best_name}_model.joblib")
    scored = pd.read_csv(PROCESSED_DIR / "scored_test_set.csv")

    from modeling import CATEGORICAL_FEATURES, NUMERIC_FEATURES

    X = scored[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    prep = model_pipe.named_steps["prep"]
    clf = model_pipe.named_steps["clf"]

    X_transformed = prep.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = (
        NUMERIC_FEATURES
        + list(prep.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CATEGORICAL_FEATURES))
    )

    # Sample for speed — SHAP on a tree model over the full 7,500-row test
    # set is feasible but a documented sample keeps this fast and reproducible.
    sample_size = min(1000, X_transformed.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X_transformed.shape[0], size=sample_size, replace=False)
    X_sample = X_transformed[idx]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)

    # Global importance (mean |SHAP value| per feature)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    global_importance = sorted(
        zip(feature_names, mean_abs_shap), key=lambda x: -x[1]
    )[:15]

    summary = {
        "model": best_name,
        "sample_size": sample_size,
        "global_importance": [{"feature": f, "mean_abs_shap": float(v)} for f, v in global_importance],
    }
    with open(PROJECT_ROOT / "reports" / "shap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Global summary plot
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", dpi=110)
    plt.close(fig)

    # Per-customer SHAP table for the sampled rows (used for local explanation lookups)
    sampled_ids = scored.iloc[idx]["ID"].values
    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{f}" for f in feature_names])
    shap_df.insert(0, "ID", sampled_ids)
    shap_df.to_csv(PROCESSED_DIR / "shap_values_sample.csv", index=False)

    print("Top global SHAP drivers:")
    for f, v in global_importance[:10]:
        print(f"  {f}: {v:.4f}")
    print(f"\nSaved: reports/shap_summary.json, reports/figures/shap_summary.png, "
          f"data/processed/shap_values_sample.csv ({sample_size} customers)")


if __name__ == "__main__":
    main()
