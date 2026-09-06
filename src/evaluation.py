"""
evaluation.py
-------------
Evaluation metrics for a PD model: discrimination, classification, and
probability-quality (calibration). Accuracy is intentionally not the
headline metric — see model_validation_report.md for why (severe class
imbalance means a trivial "always predict no-default" model already
scores ~78% accuracy while being useless).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max separation between the cumulative
    distributions of predicted PD for defaulters vs non-defaulters."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(np.abs(tpr - fpr)))


def full_metrics(y_true, y_proba, threshold: float = 0.5) -> dict[str, Any]:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "ks": ks_statistic(y_true, y_proba),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
        "threshold": threshold,
    }


def calibration_table(y_true: np.ndarray, y_proba: np.ndarray, bins=(0, 0.05, 0.10, 0.20, 0.40, 1.0)) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "pd": y_proba})
    df["band"] = pd.cut(df["pd"], bins=bins, include_lowest=True)
    grouped = df.groupby("band", observed=True).agg(
        customers=("y", "size"),
        avg_predicted_pd=("pd", "mean"),
        actual_default_rate=("y", "mean"),
    ).reset_index()
    return grouped


def reliability_curve(y_true, y_proba, n_bins: int = 10):
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    return mean_pred, frac_pos


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    PSI between an 'expected' (e.g. training/reference) distribution and
    an 'actual' (e.g. test/monitoring) distribution of the same variable
    (typically predicted PD or a single feature).

    PSI < 0.1  : no significant shift
    0.1 - 0.25 : moderate shift, investigate
    > 0.25     : major shift, investigate urgently

    These are the commonly cited industry rules of thumb, not a
    regulatory requirement — treated here as a practical monitoring
    heuristic, not a compliance threshold.
    """
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    breakpoints = np.unique(breakpoints)

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)
