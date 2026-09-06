"""
imbalance.py
------------
Investigates whether class-imbalance treatment is needed and, if so,
which approach is justified by evidence — rather than applying SMOTE
or a VAE by default.

All resampling in this module is fit/applied ONLY on the training
fold that is passed in; callers must never pass a combined
train+test set here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_config(X_train, y_train, X_test, y_test, strategy: str) -> dict[str, Any]:
    if strategy == "none":
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)
    elif strategy == "class_weight":
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(X_train, y_train)
    elif strategy == "smote":
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        model = LogisticRegression(max_iter=2000)
        model.fit(X_res, y_res)
    else:
        raise ValueError(strategy)

    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    return {
        "strategy": strategy,
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "precision_minority": float(precision_score(y_test, pred)),
        "recall_minority": float(recall_score(y_test, pred)),
        "f1_minority": float(f1_score(y_test, pred)),
    }


def run_imbalance_comparison(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    results = [
        evaluate_config(X_train, y_train, X_test, y_test, s)
        for s in ["none", "class_weight", "smote"]
    ]
    return pd.DataFrame(results)


# --------------------------------------------------------------------------
# VAE experiment (Section 8 requirement): investigate, do not use by default.
#
# SUPERSEDED: this function is a PCA-based linear approximation of a VAE,
# built when PyTorch could not be installed in this environment. It has
# been replaced by a real VAE (encoder/decoder network, reparameterization
# trick) in src/vae_model.py, run end-to-end and compared against all three
# classifiers in src/run_vae_comparison.py — see reports/vae_comparison_results.json
# and reports/model_validation_report.md §2 for the current, authoritative
# VAE experiment and verdict. This function is kept only as a record of the
# project's history, not as a live part of the pipeline.
# --------------------------------------------------------------------------

def run_vae_experiment(X_train_num: np.ndarray, y_train: np.ndarray, n_synthetic: int = 1500) -> dict[str, Any]:
    """
    DEPRECATED — see module-level note above. Use src/vae_model.py and
    src/run_vae_comparison.py instead.

    Trains a small VAE on the numeric features of the minority
    (default=1) class ONLY, using training data only, then generates
    synthetic minority samples and checks whether they are
    distributionally realistic before ever considering them for
    downstream modeling.

    Returns a verdict dict — this function does NOT decide to use the
    synthetic data; that decision is made by the caller based on the
    returned diagnostics, and is expected (given tabular data of this
    size and the imbalance ratio observed, ~3.5:1) to reject VAE
    augmentation in favor of the simpler class-weighting approach,
    unless the diagnostics show otherwise.
    """
    """
    Note on implementation: a full PyTorch/TensorFlow install was not
    feasible in this sandbox (disk budget too small for the deep-learning
    framework itself — confirmed via a failed `pip install torch`, not
    assumed). Rather than skip the requirement, this implements a
    minimal single-hidden-layer VAE by hand with NumPy (manual forward
    pass, reparameterization trick, and gradient-free optimization via
    a small evolutionary/```CMA-like``` perturbation search is too slow
    for this to be a fair comparison, so this uses a linear-Gaussian VAE
    trained with closed-form-ish PCA-based encoder/decoder as a stand-in
    generative model). This is explicitly a lightweight approximation of
    a VAE (linear encoder/decoder + Gaussian latent sampling), not a deep
    VAE — that limitation is reported alongside the results rather than
    hidden.
    """
    minority = X_train_num[y_train == 1]
    if len(minority) < 50:
        return {"ran": False, "reason": "insufficient minority samples"}

    mu_orig = minority.mean(axis=0)
    std_orig = minority.std(axis=0) + 1e-8
    minority_norm = (minority - mu_orig) / std_orig

    input_dim = minority_norm.shape[1]
    latent_dim = min(8, max(2, input_dim // 3))

    rng = np.random.default_rng(42)
    # Linear encoder via PCA (closed-form) standing in for a learned encoder mean;
    # a fixed latent-noise scale stands in for the learned logvar head.
    U, S, Vt = np.linalg.svd(minority_norm - minority_norm.mean(axis=0), full_matrices=False)
    components = Vt[:latent_dim]  # (latent_dim, input_dim) — decoder basis
    latent_train = (minority_norm - minority_norm.mean(axis=0)) @ components.T
    latent_mean = latent_train.mean(axis=0)
    latent_std = latent_train.std(axis=0) + 1e-8

    recon_train = latent_train @ components + minority_norm.mean(axis=0)
    recon_loss = float(np.mean((recon_train - minority_norm) ** 2))

    z = rng.normal(loc=latent_mean, scale=latent_std, size=(n_synthetic, latent_dim))
    synthetic_norm = z @ components + minority_norm.mean(axis=0)
    synthetic = synthetic_norm * std_orig + mu_orig

    # Diagnostics: compare real vs synthetic minority distributions
    real_mean, synth_mean = minority.mean(axis=0), synthetic.mean(axis=0)
    real_std, synth_std = minority.std(axis=0), synthetic.std(axis=0)
    mean_abs_pct_diff = float(np.mean(np.abs((synth_mean - real_mean) / (np.abs(real_mean) + 1e-6))))
    std_abs_pct_diff = float(np.mean(np.abs((synth_std - real_std) / (np.abs(real_std) + 1e-6))))

    real_corr = np.corrcoef(minority, rowvar=False)
    synth_corr = np.corrcoef(synthetic, rowvar=False)
    corr_frobenius_diff = float(np.linalg.norm(real_corr - synth_corr))

    return {
        "ran": True,
        "final_recon_loss": float(recon_loss),
        "mean_abs_pct_diff_vs_real": mean_abs_pct_diff,
        "std_abs_pct_diff_vs_real": std_abs_pct_diff,
        "correlation_matrix_frobenius_diff": corr_frobenius_diff,
        "n_synthetic": n_synthetic,
        "n_real_minority": int(len(minority)),
    }
