"""
feature_engineering.py
-----------------------
Builds credit-risk features from the cleaned dataset.

Design principle: every feature here has a stated credit-risk rationale
(see comments). Nothing is generated "because more features might help."

Chronology note: BILL_AMT1/PAY_AMT1/PAY_0 are the MOST RECENT month
(September 2005); BILL_AMT6/PAY_AMT6/PAY_6 are the OLDEST (April 2005).
"Trend" features are computed oldest -> newest so a positive slope means
"getting worse/bigger over time," which is the intuitive reading for a
risk analyst.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PAY_COLS_RECENT_TO_OLD = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS_RECENT_TO_OLD = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS_RECENT_TO_OLD = [f"PAY_AMT{i}" for i in range(1, 7)]


def _slope(row_values: np.ndarray) -> float:
    """OLS slope of values against time index (oldest->newest), for trend features."""
    x = np.arange(len(row_values))
    if np.all(row_values == row_values[0]):
        return 0.0
    return float(np.polyfit(x, row_values, 1)[0])


def add_utilization_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Utilization = BILL_AMT / LIMIT_BAL for each month.
    Rationale: utilization is one of the best-known behavioral predictors
    of credit risk — sustained high utilization signals the customer is
    relying on the card as ongoing credit rather than transacting and
    paying off, and reduces headroom before hitting the limit.

    Handling: LIMIT_BAL is always > 0 in this dataset (verified in the
    data-quality report; min = 10,000), so no zero-denominator guard is
    needed for the denominator itself. Negative BILL_AMT values (real
    overpayment states) are allowed to produce negative utilization,
    which is a meaningful value (customer is in credit, not in debt).
    Utilization is clipped only at the top (very large ratios from a
    handful of extreme bill/limit combinations) to avoid a few outliers
    dominating aggregate stats — the clip is documented, not silent.
    """
    out = df.copy()
    util_cols = []
    for i, bill_col in enumerate(BILL_COLS_RECENT_TO_OLD, start=1):
        col = f"UTIL_{i}"
        out[col] = (out[bill_col] / out["LIMIT_BAL"]).clip(-2, 3)
        util_cols.append(col)

    util_matrix = out[util_cols].to_numpy()  # columns ordered recent -> old
    out["UTIL_AVG"] = util_matrix.mean(axis=1)
    out["UTIL_MAX"] = util_matrix.max(axis=1)
    out["UTIL_MIN"] = util_matrix.min(axis=1)
    out["UTIL_LATEST"] = out["UTIL_1"]
    out["UTIL_VOLATILITY"] = util_matrix.std(axis=1)
    out["UTIL_TREND"] = np.apply_along_axis(_slope, 1, util_matrix[:, ::-1])
    return out


def add_payment_coverage_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Payment coverage = PAY_AMT_t / BILL_AMT_{t-1} (this month's payment
    relative to LAST month's bill — the amount the payment was actually
    due against). Rationale: a customer who consistently pays a small
    fraction of what they owed is building risk even if the account
    isn't technically delinquent yet (minimum-payment behavior).

    Handling: when the prior bill is <= 0 (no balance owed, or a credit
    balance), coverage is undefined/not meaningful — set to NaN rather
    than a fabricated 0 or infinity, and NaNs are excluded from the
    aggregate stats (nanmean/nanmax) rather than imputed.
    """
    out = df.copy()
    cov_cols = []
    for i in range(1, 6):
        pay_col = f"PAY_AMT{i}"
        prior_bill_col = f"BILL_AMT{i+1}"
        col = f"PAYCOV_{i}"
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = out[pay_col] / out[prior_bill_col]
        ratio = ratio.where(out[prior_bill_col] > 0, np.nan)
        out[col] = ratio.clip(upper=5)
        cov_cols.append(col)

    cov_matrix = out[cov_cols].to_numpy()
    out["PAYCOV_AVG"] = np.nanmean(cov_matrix, axis=1)
    out["PAYCOV_LATEST"] = out["PAYCOV_1"]
    with np.errstate(all="ignore"):
        out["PAYCOV_TREND"] = np.apply_along_axis(
            lambda r: _slope(r[~np.isnan(r)]) if (~np.isnan(r)).sum() >= 2 else np.nan,
            1,
            cov_matrix[:, ::-1],
        )
    return out


def add_delinquency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    From PAY_0..PAY_6 (repayment status codes). Values > 0 indicate a
    documented number of months delinquent; 0 and -2 are undocumented
    but empirically the most common "not delinquent" states (see data
    quality report); -1 is documented as "pay duly."

    Rationale: raw delinquency codes are the single strongest known
    predictor family for this problem (payment history is the core of
    every real credit bureau score); these features summarize severity,
    persistence, and recency, which is more usable to a model than six
    separate categorical columns with weak monotonic relationships to
    the code value itself.
    """
    out = df.copy()
    pay_matrix = out[PAY_COLS_RECENT_TO_OLD].to_numpy()
    delinquent_flags = (pay_matrix > 0).astype(int)

    out["N_MONTHS_DELINQUENT"] = delinquent_flags.sum(axis=1)
    out["MAX_DELINQUENCY"] = pay_matrix.clip(min=0).max(axis=1)
    out["AVG_DELINQUENCY_SEVERITY"] = pay_matrix.clip(min=0).mean(axis=1)
    out["RECENT_DELINQUENCY"] = (out["PAY_0"] > 0).astype(int)
    out["SEVERE_DELINQUENCY_FLAG"] = (pay_matrix.max(axis=1) >= 3).astype(int)
    out["PERSISTENT_DELINQUENCY_FLAG"] = (delinquent_flags.sum(axis=1) >= 3).astype(int)
    return out


def add_balance_and_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes the raw BILL_AMT*/PAY_AMT* series: level, trend, and
    volatility. Rationale: a rising balance with a flat/falling payment
    amount is a classic early-warning pattern distinct from utilization
    alone (utilization is scaled by limit; these are the raw dollar
    dynamics that a limit increase could otherwise mask).
    """
    out = df.copy()
    bill_matrix = out[BILL_COLS_RECENT_TO_OLD].to_numpy()
    pay_matrix = out[PAYAMT_COLS_RECENT_TO_OLD].to_numpy()

    out["BILL_AVG"] = bill_matrix.mean(axis=1)
    out["BILL_LATEST"] = out["BILL_AMT1"]
    out["BILL_MAX"] = bill_matrix.max(axis=1)
    out["BILL_TREND"] = np.apply_along_axis(_slope, 1, bill_matrix[:, ::-1])
    out["BILL_VOLATILITY"] = bill_matrix.std(axis=1)

    out["PAYAMT_AVG"] = pay_matrix.mean(axis=1)
    out["PAYAMT_LATEST"] = out["PAY_AMT1"]
    out["PAYAMT_MAX"] = pay_matrix.max(axis=1)
    out["PAYAMT_TREND"] = np.apply_along_axis(_slope, 1, pay_matrix[:, ::-1])
    out["PAYAMT_VOLATILITY"] = pay_matrix.std(axis=1)
    return out


def add_exposure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Credit-exposure features relating balance to limit and to observed
    payment capacity. Rationale: a bank cares not just about "is this
    person delinquent" but "how much are we exposed to if they default,"
    and "does their payment behavior look sustainable relative to what
    they owe."
    """
    out = df.copy()
    out["BALANCE_TO_LIMIT_LATEST"] = (out["BILL_AMT1"] / out["LIMIT_BAL"]).clip(-2, 3)
    out["PAYMENT_CAPACITY"] = (out["PAY_AMT1"] / out["LIMIT_BAL"]).clip(0, 2)
    out["UTILIZATION_STRESS_FLAG"] = (
        (out["UTIL_LATEST"] > 0.8) & (out["RECENT_DELINQUENCY"] == 1)
    ).astype(int)
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = add_utilization_features(out)
    out = add_payment_coverage_features(out)
    out = add_delinquency_features(out)
    out = add_balance_and_payment_features(out)
    out = add_exposure_features(out)
    return out


if __name__ == "__main__":
    from data_ingestion import load_raw_data
    from preprocessing import clean_dataset

    raw = load_raw_data()
    cleaned = clean_dataset(raw)
    featured = engineer_features(cleaned)
    print("Shape after feature engineering:", featured.shape)
    print("New columns:", [c for c in featured.columns if c not in cleaned.columns])
    featured.to_csv("../data/processed/featured.csv", index=False)
