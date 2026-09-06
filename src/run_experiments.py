"""
run_experiments.py
-------------------
End-to-end driver: loads data, cleans, engineers features, splits,
compares imbalance strategies, runs the VAE experiment, trains and
compares models, evaluates, calibrates, segments, checks stability
(PSI) and fairness, and saves all artifacts + a JSON results bundle
that the report-writing and dashboard code consume.

Run from the src/ directory: `python3 run_experiments.py`
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

from data_ingestion import load_raw_data
from evaluation import calibration_table, full_metrics, population_stability_index
from feature_engineering import engineer_features
from imbalance import run_imbalance_comparison, run_vae_experiment
from modeling import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COL, build_models, split_data
from preprocessing import clean_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
for d in (MODELS_DIR, REPORTS_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)

RESULTS: dict = {}


def main():
    # ---- 1. Data prep ----
    raw = load_raw_data()
    cleaned = clean_dataset(raw)
    featured = engineer_features(cleaned)
    featured.to_csv(PROCESSED_DIR / "featured.csv", index=False)

    X_train, X_test, y_train, y_test = split_data(featured, test_size=0.25, random_state=42)
    RESULTS["split"] = {
        "n_train": len(X_train), "n_test": len(X_test),
        "train_default_rate": float(y_train.mean()), "test_default_rate": float(y_test.mean()),
    }
    print("Split:", RESULTS["split"])

    # ---- 2. Class imbalance comparison (on a simple numeric-only encoding for speed/clarity) ----
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    quick_prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    X_train_enc = quick_prep.fit_transform(X_train)
    X_test_enc = quick_prep.transform(X_test)

    imbalance_df = run_imbalance_comparison(X_train_enc, y_train.values, X_test_enc, y_test.values)
    RESULTS["imbalance_comparison"] = imbalance_df.to_dict(orient="records")
    print("\nImbalance comparison:\n", imbalance_df)

    # ---- 3. VAE experiment (numeric features only, minority class, train fold only) ----
    numeric_train_dense = X_train_enc[:, :len(NUMERIC_FEATURES)]
    if hasattr(numeric_train_dense, "toarray"):
        numeric_train_dense = numeric_train_dense.toarray()
    vae_result = run_vae_experiment(np.asarray(numeric_train_dense), y_train.values)
    RESULTS["vae_experiment"] = vae_result
    print("\nVAE experiment:", vae_result)

    # ---- 4. Fit candidate models (chosen imbalance strategy: class_weight — see decision below) ----
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / n_pos

    models = build_models()
    fitted = {}
    for name, pipe in models.items():
        if name == "xgboost":
            pipe.set_params(clf__scale_pos_weight=scale_pos_weight)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        print(f"Fitted {name}")

    # ---- 5. Evaluate all models on the untouched test set ----
    model_metrics = {}
    proba_by_model = {}
    for name, pipe in fitted.items():
        proba = pipe.predict_proba(X_test)[:, 1]
        proba_by_model[name] = proba
        model_metrics[name] = full_metrics(y_test.values, proba, threshold=0.5)
    RESULTS["model_metrics"] = model_metrics
    print("\nModel metrics:")
    for name, m in model_metrics.items():
        print(name, {k: v for k, v in m.items() if k != "confusion_matrix"})

    # ---- 6. Select best model by ROC-AUC + PR-AUC (documented, not accuracy) ----
    best_name = max(model_metrics, key=lambda n: (model_metrics[n]["roc_auc"] + model_metrics[n]["pr_auc"]) / 2)
    best_model = fitted[best_name]
    best_proba = proba_by_model[best_name]
    RESULTS["best_model"] = best_name
    print("\nBest model by (ROC-AUC+PR-AUC)/2:", best_name)

    # ---- 7. Calibration: compare raw vs isotonic ----
    calibrated = CalibratedClassifierCV(best_model, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)
    calibrated_proba = calibrated.predict_proba(X_test)[:, 1]

    from sklearn.metrics import brier_score_loss
    raw_brier = brier_score_loss(y_test, best_proba)
    calibrated_brier = brier_score_loss(y_test, calibrated_proba)
    RESULTS["calibration"] = {
        "raw_brier": float(raw_brier),
        "isotonic_brier": float(calibrated_brier),
        "chosen": "isotonic" if calibrated_brier < raw_brier else "raw",
    }
    print("\nCalibration:", RESULTS["calibration"])

    final_proba = calibrated_proba if RESULTS["calibration"]["chosen"] == "isotonic" else best_proba

    # ---- 8. PD calibration table / risk bands (evaluated on TEST predictions) ----
    cal_table = calibration_table(y_test.values, final_proba)
    RESULTS["calibration_table"] = cal_table.astype(str).to_dict(orient="records") if cal_table.empty else \
        cal_table.assign(band=cal_table["band"].astype(str)).to_dict(orient="records")
    print("\nCalibration table:\n", cal_table)

    # ---- 9. Population Stability Index (train vs test, as a stand-in for a true OOT sample) ----
    train_proba_final = (
        calibrated.predict_proba(X_train)[:, 1] if RESULTS["calibration"]["chosen"] == "isotonic"
        else best_model.predict_proba(X_train)[:, 1]
    )
    psi_score = population_stability_index(train_proba_final, final_proba)
    RESULTS["psi_train_vs_test_pd"] = psi_score
    print("\nPSI (train PD dist vs test PD dist):", psi_score)

    # ---- 10. Fairness: compare metrics across SEX, AGE groups, EDUCATION ----
    fairness = {}
    X_test_fair = X_test.copy()
    X_test_fair["_y"] = y_test.values
    X_test_fair["_pd"] = final_proba
    X_test_fair["_pred"] = (final_proba >= 0.5).astype(int)

    def group_metrics(g: pd.DataFrame) -> dict:
        tp = ((g["_pred"] == 1) & (g["_y"] == 1)).sum()
        fp = ((g["_pred"] == 1) & (g["_y"] == 0)).sum()
        fn = ((g["_pred"] == 0) & (g["_y"] == 1)).sum()
        tn = ((g["_pred"] == 0) & (g["_y"] == 0)).sum()
        n = len(g)
        return {
            "n": int(n),
            "actual_default_rate": float(g["_y"].mean()),
            "avg_predicted_pd": float(g["_pd"].mean()),
            "tpr_recall": float(tp / (tp + fn)) if (tp + fn) > 0 else None,
            "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else None,
            "fnr": float(fn / (fn + tp)) if (fn + tp) > 0 else None,
        }

    fairness["SEX"] = {str(k): group_metrics(g) for k, g in X_test_fair.groupby("SEX")}
    X_test_fair["_age_band"] = pd.cut(X_test_fair["AGE"], bins=[20, 30, 40, 50, 60, 80])
    fairness["AGE_BAND"] = {str(k): group_metrics(g) for k, g in X_test_fair.groupby("_age_band", observed=True)}
    fairness["EDUCATION"] = {str(k): group_metrics(g) for k, g in X_test_fair.groupby("EDUCATION")}
    RESULTS["fairness"] = fairness
    print("\nFairness (SEX):", fairness["SEX"])

    # ---- 11. Risk segmentation thresholds from PD distribution ----
    # FIX: thresholds must come from the TRAINING PD distribution, not test —
    # using test-set quantiles to bucket the test set is circular (it makes
    # the resulting calibration-by-band look better than it would on truly
    # unseen data). train_proba_final was already computed in step 9.
    pd_series = pd.Series(train_proba_final)
    thresholds = {
        "low_max": float(pd_series.quantile(0.50)),
        "moderate_max": float(pd_series.quantile(0.80)),
        "high_max": float(pd_series.quantile(0.95)),
    }
    RESULTS["segmentation_thresholds"] = thresholds
    print("\nSegmentation thresholds:", thresholds)

    # ---- 12. Illustrative ECL scenario: EL = PD x LGD x EAD ----
    ead_proxy = X_test["LIMIT_BAL"].values  # illustrative EAD proxy: current credit limit
    for lgd_scenario in (0.45, 0.60, 0.75):
        el = final_proba * lgd_scenario * ead_proxy
        RESULTS.setdefault("ecl_scenarios", {})[str(lgd_scenario)] = {
            "total_expected_loss": float(el.sum()),
            "avg_expected_loss_per_account": float(el.mean()),
        }
    print("\nECL scenarios:", RESULTS.get("ecl_scenarios"))

    # ---- 13. Explainability: LR coefficients (odds ratios) + RF/XGB feature importance ----
    explain = {}
    lr_pipe = fitted["logistic_regression"]
    feature_names = (
        NUMERIC_FEATURES
        + list(lr_pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CATEGORICAL_FEATURES))
    )
    coefs = lr_pipe.named_steps["clf"].coef_[0]
    odds_ratios = np.exp(coefs)
    lr_importance = sorted(zip(feature_names, coefs, odds_ratios), key=lambda x: -abs(x[1]))[:15]
    explain["logistic_regression_top_features"] = [
        {"feature": f, "coefficient": float(c), "odds_ratio": float(o)} for f, c, o in lr_importance
    ]

    if best_name in ("random_forest", "xgboost"):
        best_clf = fitted[best_name].named_steps["clf"]
        importances = best_clf.feature_importances_
        tree_importance = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:15]
        explain[f"{best_name}_top_features"] = [{"feature": f, "importance": float(v)} for f, v in tree_importance]
    RESULTS["explainability"] = explain
    print("\nTop LR features:", explain["logistic_regression_top_features"][:5])

    # ---- 14. Save artifacts ----
    joblib.dump(fitted[best_name], MODELS_DIR / f"{best_name}_model.joblib")
    joblib.dump(calibrated, MODELS_DIR / "calibrated_model.joblib")
    joblib.dump(quick_prep, MODELS_DIR / "quick_preprocessor.joblib")
    with open(REPORTS_DIR / "experiment_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)

    # Save scored test set for the dashboard
    scored = X_test.copy()
    scored["actual_default"] = y_test.values
    scored["predicted_pd"] = final_proba
    scored["risk_band"] = pd.cut(
        scored["predicted_pd"],
        bins=[0, thresholds["low_max"], thresholds["moderate_max"], thresholds["high_max"], 1.0],
        labels=["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"],
        include_lowest=True,
    )
    scored.to_csv(PROCESSED_DIR / "scored_test_set.csv", index=False)

    print("\nSaved model:", best_name, "-> models/")
    print("Saved results bundle -> reports/experiment_results.json")
    print("Saved scored test set -> data/processed/scored_test_set.csv")


if __name__ == "__main__":
    main()
