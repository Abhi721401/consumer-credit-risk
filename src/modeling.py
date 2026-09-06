"""
modeling.py
-----------
Builds the modeling pipeline end to end with an explicit leakage-safe
design:

  1. Stratified train/test split FIRST.
  2. All fitted transforms (scaler, one-hot categories) are fit on the
     TRAINING fold only, then applied to test.
  3. Any resampling (class-weighting is used here, not SMOTE — see
     reports/model_validation_report.md for why) happens only inside
     the training fold.
  4. The test set is touched exactly once, at final evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ID_COL = "ID"
TARGET_COL = "default_next_month"

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]

# PAY_* kept as numeric ordinal codes (documented decision: treated as
# ordinal, scaled like other numerics for the linear model; tree models
# are invariant to monotonic scaling anyway).
NUMERIC_FEATURES = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "UTIL_AVG", "UTIL_MAX", "UTIL_MIN", "UTIL_LATEST", "UTIL_VOLATILITY", "UTIL_TREND",
    "PAYCOV_AVG", "PAYCOV_LATEST", "PAYCOV_TREND",
    "N_MONTHS_DELINQUENT", "MAX_DELINQUENCY", "AVG_DELINQUENCY_SEVERITY",
    "RECENT_DELINQUENCY", "SEVERE_DELINQUENCY_FLAG", "PERSISTENT_DELINQUENCY_FLAG",
    "BILL_AVG", "BILL_LATEST", "BILL_MAX", "BILL_TREND", "BILL_VOLATILITY",
    "PAYAMT_AVG", "PAYAMT_LATEST", "PAYAMT_MAX", "PAYAMT_TREND", "PAYAMT_VOLATILITY",
    "BALANCE_TO_LIMIT_LATEST", "PAYMENT_CAPACITY", "UTILIZATION_STRESS_FLAG",
]


def split_data(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    # NaNs can occur in PAYCOV_* features (undefined when prior bill <= 0);
    # median imputation is fit on train only via the pipeline below.
    from sklearn.impute import SimpleImputer

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()

    log_reg = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1, random_state=random_state)),
    ])

    rf = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )),
    ])

    n_pos = None  # set by caller via scale_pos_weight if desired
    xgb = Pipeline([
        ("prep", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=random_state, n_jobs=-1,
        )),
    ])

    return {"logistic_regression": log_reg, "random_forest": rf, "xgboost": xgb}


def fit_all(models: dict[str, Pipeline], X_train, y_train, scale_pos_weight: float | None = None):
    fitted = {}
    for name, pipe in models.items():
        if name == "xgboost" and scale_pos_weight is not None:
            pipe.set_params(clf__scale_pos_weight=scale_pos_weight)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
    return fitted
