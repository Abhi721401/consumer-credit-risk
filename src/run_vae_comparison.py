"""
run_vae_comparison.py
----------------------
Implements and evaluates the VAE-based class-imbalance approach:

  1. Preprocess (fit ONLY on training fold).
  2. Train a real VAE (src/vae_model.py) on the minority (default=1)
     class of the TRAINING fold only.
  3. Generate synthetic minority samples to fully balance the training
     set (n_synthetic = n_majority_train - n_minority_train), matching
     the same balancing target SMOTE used, for a fair comparison.
  4. Post-process synthetic samples: re-discretize one-hot categorical
     blocks (argmax) and round binary flag features — a raw Gaussian
     VAE output for a one-hot block or a 0/1 flag is not a valid record
     otherwise.
  5. Run distributional diagnostics comparing synthetic vs. real
     minority-class training data.
  6. Train Logistic Regression, Random Forest, and XGBoost on:
       (a) the original imbalanced training set (no augmentation)
       (b) the VAE-balanced training set
     and evaluate both on the SAME untouched test set.
  7. Report the comparison honestly — the verdict is whatever the
     numbers say, not a foregone conclusion.

The test set is never used for VAE training, synthetic generation, or
preprocessing fit — it is touched only at final evaluation, exactly
once per model.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

from data_ingestion import load_raw_data
from evaluation import full_metrics
from feature_engineering import engineer_features
from modeling import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor, split_data
from preprocessing import clean_dataset
from vae_model import generate_synthetic_samples, train_vae

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"


def get_feature_names_and_groups(preprocessor) -> tuple[list[str], dict]:
    """Returns full feature name list, plus index groups for the one-hot
    categorical blocks and the binary flag features, needed to
    post-process raw VAE output into valid records."""
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERIC_FEATURES + cat_names

    # one-hot blocks: group column indices by source categorical variable
    onehot_groups = {}
    offset = len(NUMERIC_FEATURES)
    for cat_var, categories in zip(CATEGORICAL_FEATURES, cat_encoder.categories_):
        onehot_groups[cat_var] = list(range(offset, offset + len(categories)))
        offset += len(categories)

    binary_flags = [
        "RECENT_DELINQUENCY", "SEVERE_DELINQUENCY_FLAG",
        "PERSISTENT_DELINQUENCY_FLAG", "UTILIZATION_STRESS_FLAG",
    ]
    binary_flag_idx = [feature_names.index(f) for f in binary_flags if f in feature_names]

    return feature_names, onehot_groups, binary_flag_idx


def postprocess_synthetic(synthetic: np.ndarray, onehot_groups: dict, binary_flag_idx: list[int]) -> np.ndarray:
    out = synthetic.copy()
    # Re-discretize one-hot blocks via argmax -> valid one-hot row
    for cat_var, idxs in onehot_groups.items():
        block = out[:, idxs]
        winner = np.argmax(block, axis=1)
        new_block = np.zeros_like(block)
        new_block[np.arange(len(block)), winner] = 1.0
        out[:, idxs] = new_block
    # Round binary flags to {0, 1}
    for idx in binary_flag_idx:
        out[:, idx] = np.clip(np.round(out[:, idx]), 0, 1)
    return out


def distributional_diagnostics(real: np.ndarray, synthetic: np.ndarray, feature_names: list[str]) -> dict:
    real_mean, synth_mean = real.mean(axis=0), synthetic.mean(axis=0)
    real_std, synth_std = real.std(axis=0), synthetic.std(axis=0)

    # Relative (percent) difference — reported for completeness, but flagged as
    # unstable: it explodes for any feature whose real mean is near zero
    # (several engineered trend/ratio features are), which is a metric
    # artifact, not evidence of a bad synthetic distribution.
    mean_abs_pct_diff = float(np.mean(np.abs((synth_mean - real_mean) / (np.abs(real_mean) + 1e-6))))
    std_abs_pct_diff = float(np.mean(np.abs((synth_std - real_std) / (np.abs(real_std) + 1e-6))))

    # Std-normalized mean difference (in units of the real feature's own std) —
    # the more reliable diagnostic: it answers "how many standard deviations
    # off is the synthetic mean," which behaves sensibly near zero.
    standardized_mean_diff = np.abs((synth_mean - real_mean) / (real_std + 1e-8))
    mean_standardized_diff = float(standardized_mean_diff.mean())

    real_corr = np.corrcoef(real, rowvar=False)
    synth_corr = np.corrcoef(synthetic, rowvar=False)
    real_corr = np.nan_to_num(real_corr)
    synth_corr = np.nan_to_num(synth_corr)
    corr_frobenius_diff = float(np.linalg.norm(real_corr - synth_corr))

    worst_by_pct = sorted(zip(feature_names, np.abs((synth_mean - real_mean) / (np.abs(real_mean) + 1e-6))),
                           key=lambda x: -x[1])[:5]
    worst_by_std = sorted(zip(feature_names, standardized_mean_diff), key=lambda x: -x[1])[:8]

    return {
        "mean_abs_pct_diff_vs_real": mean_abs_pct_diff,
        "std_abs_pct_diff_vs_real": std_abs_pct_diff,
        "mean_standardized_diff_vs_real": mean_standardized_diff,
        "note": (
            "mean_abs_pct_diff is unstable for near-zero-mean features (several "
            "engineered trend/ratio features have real means close to 0, which "
            "inflates a relative-percent metric); mean_standardized_diff (mean "
            "difference in units of the real feature's own std) is the more "
            "reliable summary and is what the verdict below is based on."
        ),
        "correlation_matrix_frobenius_diff": corr_frobenius_diff,
        "worst_features_by_pct_deviation": [{"feature": f, "pct_diff": float(v)} for f, v in worst_by_pct],
        "worst_features_by_standardized_deviation": [{"feature": f, "std_diff": float(v)} for f, v in worst_by_std],
    }


def evaluate_on_data(X_train, y_train, X_test, y_test, label: str) -> dict:
    results = {}
    models = {
        "logistic_regression": LogisticRegression(max_iter=3000, C=0.1, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        ),
    }
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        results[name] = full_metrics(y_test, proba, threshold=0.5)
        print(f"  [{label}] {name}: ROC-AUC={results[name]['roc_auc']:.4f}  "
              f"PR-AUC={results[name]['pr_auc']:.4f}  Recall={results[name]['recall']:.4f}  "
              f"F1={results[name]['f1']:.4f}")
    return results


def main():
    raw = load_raw_data()
    cleaned = clean_dataset(raw)
    featured = engineer_features(cleaned)
    X_train, X_test, y_train, y_test = split_data(featured, test_size=0.25, random_state=42)

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
    if hasattr(X_test_proc, "toarray"):
        X_test_proc = X_test_proc.toarray()

    feature_names, onehot_groups, binary_flag_idx = get_feature_names_and_groups(preprocessor)

    y_train_arr = y_train.values
    minority_mask = y_train_arr == 1
    X_minority = X_train_proc[minority_mask]
    n_majority = int((~minority_mask).sum())
    n_minority = int(minority_mask.sum())
    n_to_generate = n_majority - n_minority

    print(f"Training fold: {n_majority} majority, {n_minority} minority. "
          f"Generating {n_to_generate} synthetic minority samples to fully balance.\n")

    print("Training VAE on minority-class training data only...")
    vae_result = train_vae(X_minority, hidden_dim=64, latent_dim=12, epochs=300, batch_size=64)
    print(f"Final reconstruction loss: {vae_result.final_recon_loss:.4f}, "
          f"final KLD: {vae_result.final_kld_loss:.4f}\n")

    synthetic_raw = generate_synthetic_samples(vae_result, n_to_generate)
    synthetic = postprocess_synthetic(synthetic_raw, onehot_groups, binary_flag_idx)

    diagnostics = distributional_diagnostics(X_minority, synthetic, feature_names)
    print("Distributional diagnostics (synthetic vs. real minority-class training data):")
    print(json.dumps(diagnostics, indent=2))

    # Build the VAE-balanced training set
    X_train_vae_balanced = np.vstack([X_train_proc, synthetic])
    y_train_vae_balanced = np.concatenate([y_train_arr, np.ones(n_to_generate)])
    print(f"\nVAE-balanced training set: {X_train_vae_balanced.shape[0]} rows "
          f"({y_train_vae_balanced.mean()*100:.1f}% positive, vs. {y_train_arr.mean()*100:.1f}% original)")

    print("\n--- Baseline: original imbalanced training data (no resampling) ---")
    baseline_metrics = evaluate_on_data(X_train_proc, y_train_arr, X_test_proc, y_test.values, "baseline")

    print("\n--- VAE-balanced training data ---")
    vae_metrics = evaluate_on_data(X_train_vae_balanced, y_train_vae_balanced, X_test_proc, y_test.values, "vae")

    # Load prior class_weight / SMOTE results for a complete side-by-side comparison
    with open(REPORTS_DIR / "experiment_results.json") as f:
        prior_results = json.load(f)
    prior_imbalance = {r["strategy"]: r for r in prior_results["imbalance_comparison"]}

    comparison_rows = []
    for name, m in baseline_metrics.items():
        comparison_rows.append({"model": name, "strategy": "none (baseline)", **{k: m[k] for k in ["roc_auc", "pr_auc", "recall", "precision", "f1"]}})
    for name, m in vae_metrics.items():
        comparison_rows.append({"model": name, "strategy": "vae_balanced", **{k: m[k] for k in ["roc_auc", "pr_auc", "recall", "precision", "f1"]}})
    # class_weight/SMOTE were only run on logistic regression in the earlier experiment — include for reference
    for strat in ("class_weight", "smote"):
        r = prior_imbalance[strat]
        comparison_rows.append({
            "model": "logistic_regression", "strategy": strat,
            "roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"],
            "recall": r["recall_minority"], "precision": r["precision_minority"], "f1": r["f1_minority"],
        })

    comparison_df = pd.DataFrame(comparison_rows)
    print("\n=== FULL COMPARISON TABLE ===")
    print(comparison_df.to_string(index=False))

    output = {
        "n_majority_train": n_majority,
        "n_minority_train": n_minority,
        "n_synthetic_generated": n_to_generate,
        "vae_training": {
            "final_recon_loss": vae_result.final_recon_loss,
            "final_kld_loss": vae_result.final_kld_loss,
            "epochs": len(vae_result.loss_history),
        },
        "distributional_diagnostics": diagnostics,
        "baseline_metrics": baseline_metrics,
        "vae_balanced_metrics": vae_metrics,
        "full_comparison_table": comparison_df.to_dict(orient="records"),
    }
    with open(REPORTS_DIR / "vae_comparison_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved -> {REPORTS_DIR / 'vae_comparison_results.json'}")


if __name__ == "__main__":
    main()
