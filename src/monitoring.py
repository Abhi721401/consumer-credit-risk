"""
monitoring.py
-------------
Model-monitoring utilities intended to run periodically against a new
batch of scored data once real outcomes become available.

NOTE: this project has one static snapshot of data (April-Sept 2005),
so there is no true out-of-time monitoring batch. PSI here is computed
between the train and test PD distributions as a methodological
demonstration of the calculation, not a real drift finding — this
limitation is stated explicitly rather than implied away.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from evaluation import full_metrics, population_stability_index


def performance_monitoring_report(y_true, y_proba) -> dict[str, Any]:
    return full_metrics(y_true, y_proba)


def feature_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        if f not in reference_df.columns or f not in current_df.columns:
            continue
        psi = population_stability_index(reference_df[f].dropna().values, current_df[f].dropna().values)
        rows.append({"feature": f, "psi": psi, "flag": _psi_flag(psi)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def _psi_flag(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    elif psi < 0.25:
        return "moderate_shift_investigate"
    else:
        return "major_shift_investigate_urgently"


def prediction_drift_report(reference_pd: np.ndarray, current_pd: np.ndarray) -> dict[str, Any]:
    return {
        "reference_avg_pd": float(np.mean(reference_pd)),
        "current_avg_pd": float(np.mean(current_pd)),
        "psi": population_stability_index(reference_pd, current_pd),
        "reference_pd_std": float(np.std(reference_pd)),
        "current_pd_std": float(np.std(current_pd)),
    }


RETRAINING_TRIGGERS = [
    "PSI on predicted PD exceeds 0.25 vs. the training-time reference distribution",
    "ROC-AUC on a rolling monitoring window drops more than 0.03 below the validated test-set ROC-AUC",
    "Brier score on a rolling monitoring window increases materially vs. the validated test-set Brier score",
    "Calibration table shows a risk band's actual default rate diverging from its average predicted PD "
    "by more than ~30% relative, for two consecutive monitoring periods",
    "A feature used in the model shows PSI > 0.25 versus its training distribution",
    "A material change in underwriting policy, credit-limit strategy, or the macroeconomic environment "
    "that would be expected to shift the underlying relationship between features and default risk",
]
