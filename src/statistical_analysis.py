"""
statistical_analysis.py
------------------------
Formal statistical tests of association between features and the
target. These establish ASSOCIATION, not causation — observational
data of this kind cannot support causal claims, and none are made here
or in any report generated from this module's output.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def chi_square_test(df: pd.DataFrame, cat_col: str, target_col: str) -> dict[str, Any]:
    table = pd.crosstab(df[cat_col], df[target_col])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.sum().sum()
    # Cramer's V effect size
    r, k = table.shape
    cramers_v = float(np.sqrt((chi2 / n) / (min(r - 1, k - 1))))
    return {"variable": cat_col, "chi2": float(chi2), "p_value": float(p), "dof": int(dof), "cramers_v": cramers_v}


def numeric_group_test(df: pd.DataFrame, num_col: str, target_col: str) -> dict[str, Any]:
    """Welch's t-test (unequal variance) between default=0 and default=1 groups, plus Cohen's d."""
    g0 = df.loc[df[target_col] == 0, num_col].dropna()
    g1 = df.loc[df[target_col] == 1, num_col].dropna()
    t_stat, p_val = stats.ttest_ind(g0, g1, equal_var=False)
    pooled_std = np.sqrt(((g0.std() ** 2) + (g1.std() ** 2)) / 2)
    cohens_d = float((g1.mean() - g0.mean()) / pooled_std) if pooled_std > 0 else 0.0
    return {
        "variable": num_col,
        "mean_no_default": float(g0.mean()),
        "mean_default": float(g1.mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": cohens_d,
    }


def run_statistical_suite(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    cat_vars = ["SEX", "EDUCATION", "MARRIAGE"]
    num_vars = [
        "LIMIT_BAL", "AGE", "UTIL_AVG", "UTIL_LATEST", "N_MONTHS_DELINQUENT",
        "MAX_DELINQUENCY", "PAYCOV_AVG", "BILL_AVG", "PAYAMT_AVG",
    ]
    chi_results = [chi_square_test(df, c, target_col) for c in cat_vars if c in df.columns]
    t_results = [numeric_group_test(df, c, target_col) for c in num_vars if c in df.columns]
    corr = df[num_vars + [target_col]].corr(numeric_only=True)[target_col].drop(target_col).sort_values(
        key=lambda s: s.abs(), ascending=False
    )
    return {"chi_square": chi_results, "group_tests": t_results, "correlation_with_target": corr.to_dict()}


if __name__ == "__main__":
    import json

    from data_ingestion import load_raw_data
    from feature_engineering import engineer_features
    from preprocessing import TARGET_CLEAN, clean_dataset

    df = engineer_features(clean_dataset(load_raw_data()))
    results = run_statistical_suite(df, TARGET_CLEAN)
    print(json.dumps(results, indent=2, default=str))
