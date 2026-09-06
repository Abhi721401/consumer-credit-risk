# Data Quality Report — Default of Credit Card Clients Dataset

*All figures below were computed directly from `data/raw/UCI_Credit_Card.csv` via
`src/data_validation.py`. Nothing here is assumed from the UCI/Kaggle documentation
without being checked against the actual file.*

## 1. Source and Provenance

- Dataset: **Default of Credit Card Clients** (Yeh & Lien, 2009), UCI ML Repository / Kaggle mirror `uciml/default-of-credit-card-clients-dataset`.
- File obtained: `default_of_credit_card_clients.csv`, retrieved from a public GitHub mirror of the UCI file (Kaggle itself was not reachable from this environment's network allowlist). Row/column counts and value distributions below match the values independently reported by multiple public analyses of the Kaggle version, which is the standard cross-check for authenticity.
- Row count: **30,000**
- Column count: **25**

## 2. Schema

| Column | dtype |
|---|---|
| ID | int64 |
| LIMIT_BAL | int64 |
| SEX | int64 |
| EDUCATION | int64 |
| MARRIAGE | int64 |
| AGE | int64 |
| PAY_0, PAY_2–PAY_6 | int64 |
| BILL_AMT1–BILL_AMT6 | int64 |
| PAY_AMT1–PAY_AMT6 | int64 |
| default payment next month | int64 |

**Note:** the target column's literal header in this file is `default payment next month`
(space-separated), not `default.payment.next.month` as sometimes written in
documentation. All downstream code resolves the target column name
programmatically (`data_validation.find_target_column`) rather than hard-coding it,
specifically to avoid this kind of mismatch causing a silent bug.

There is no separate `PAY_1` column — the first repayment-status column is named
`PAY_0`. This is a known quirk of this dataset and is preserved rather than
silently renamed, though it is documented here to avoid confusion (`PAY_0` = repayment
status for the most recent month, September 2005; `PAY_2`...`PAY_6` step backward
month by month — there is no gap in the underlying months, only in the column name).

## 3. Missing Values

**Zero missing values in any of the 25 columns.** Every column has a null count of 0
(0.0%) across all 30,000 rows. This is consistent with the dataset being a cleaned
academic/benchmark release rather than raw operational data — a claim we treat as a
data-quality *observation*, not an assumption, since it was verified column-by-column.

## 4. Duplicates

- Fully duplicated rows: **0**
- Duplicate `ID` values: **0**
- `ID` is a fully unique key, running sequentially from 1 to 30,000 (`id_is_sequential_1_to_n = True`).

**Decision:** `ID` carries no behavioral information — it is a synthetic row index, not
an account-opening sequence number tied to a real onboarding timeline. It will be
excluded from modeling features and retained only as a row identifier for scoring
outputs.

## 5. Categorical Variables — Actual Observed Codes

### SEX
| Code | Count |
|---|---|
| 1 (documented: male) | 11,888 |
| 2 (documented: female) | 18,112 |

No undocumented codes observed.

### EDUCATION
Documented codes: 1=graduate school, 2=university, 3=high school, 4=others.

| Code | Count | Status |
|---|---|---|
| 0 | 14 | **Undocumented** |
| 1 | 10,585 | Documented |
| 2 | 14,030 | Documented |
| 3 | 4,917 | Documented |
| 4 | 123 | Documented ("others") |
| 5 | 280 | **Undocumented** |
| 6 | 51 | **Undocumented** |

Total undocumented EDUCATION rows: **345** (1.15% of the dataset).

**Decision:** Codes 0, 5, and 6 are not silently dropped. They will be grouped with
code 4 into a single `Other/Unknown` category during cleaning, since none of 0/5/6
have a documented meaning and none is large enough (max 280 rows, 0.93%) to justify
inventing a separate treatment for each. This is a judgment call made explicit here
rather than a default `pandas` behavior.

### MARRIAGE
Documented codes: 1=married, 2=single, 3=divorce/others.

| Code | Count | Status |
|---|---|---|
| 0 | 54 | **Undocumented** |
| 1 | 13,659 | Documented |
| 2 | 15,964 | Documented |
| 3 | 323 | Documented |

**Decision:** Code 0 (54 rows, 0.18%) will be grouped into the existing
`Other`-type category (alongside 3) rather than dropped, for the same reason as above.

## 6. Repayment Status Variables (PAY_0, PAY_2–PAY_6)

These are **not** continuous measurements — they are an ordinal/categorical repayment-status
code. Observed values across all six columns: **-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8**.

Example — actual distribution of `PAY_0`:

| Value | Count |
|---|---|
| -2 | 2,759 |
| -1 | 5,686 |
| 0 | 14,737 |
| 1 | 3,688 |
| 2 | 2,667 |
| 3 | 322 |
| 4 | 76 |
| 5 | 26 |
| 6 | 11 |
| 7 | 9 |
| 8 | 19 |

The commonly cited UCI documentation defines -1 = pay duly and 1–9 = months delayed,
but does not clearly define 0 or -2. In practice (and consistent with how this
dataset is treated across independent public analyses), 0 is generally read as "paid
minimum due / no delay recorded" and -2 as "no consumption / balance already settled,"
but this is an **interpretation**, not a documented fact, and will be flagged as such
in the governance document rather than presented as certain.

**Implication for modeling:** these columns must not be treated as ordinary
interval-scaled numeric inputs assumed to have a linear effect. They will be used both
in engineered form (e.g., delinquency-severity, "ever severely delinquent" flags) and,
where kept in a model, treated as ordinal/categorical rather than assumed-linear.

## 7. Bill Amounts — Negative Values

`BILL_AMT1`–`BILL_AMT6` can legitimately be negative (a credit balance / overpayment on
the card), but the counts are non-trivial:

| Column | Negative-value count |
|---|---|
| BILL_AMT1 | 590 |
| BILL_AMT2 | 669 |
| BILL_AMT3 | 655 |
| BILL_AMT4 | 675 |
| BILL_AMT5 | 655 |
| BILL_AMT6 | 688 |

**Decision:** these are **not** treated as invalid/erroneous and are **not** removed —
a negative bill balance is a real, explainable state (overpayment / credit balance) for
a revolving credit product. They do, however, require care in feature engineering:
utilization ( BILL_AMT / LIMIT_BAL ) and payment-coverage ( PAY_AMT / BILL_AMT ) ratios
must be computed with explicit handling for negative and zero denominators/numerators
(see feature engineering stage), rather than being allowed to produce nonsensical
ratios silently.

## 8. Numeric Summary (selected columns; full table generated in `src/data_validation.py` output)

| Column | Mean | Std | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| LIMIT_BAL | 167,484 | 129,748 | 10,000 | 140,000 | 1,000,000 |
| AGE | 35.5 | 9.2 | 21 | 34 | 79 |
| BILL_AMT1 | 51,223 | 73,636 | -165,580 | 22,382 | 964,511 |
| PAY_AMT1 | 5,664 | 16,563 | 0 | 2,100 | 873,552 |

`LIMIT_BAL`, the `BILL_AMT*` and `PAY_AMT*` columns are all heavily right-skewed
(mean well above median, large max relative to the 99th percentile), typical of
monetary variables. This informs the modeling stage: tree-based models are
unaffected, but logistic regression will need scaling and may benefit from
log-style transforms, which will be evaluated (not assumed) during modeling.

## 9. Target Variable

Column: `default payment next month` (0 = no default, 1 = default).

| Class | Count | Rate |
|---|---:|---:|
| 0 (no default) | 23,364 | 77.88% |
| 1 (default) | 6,636 | 22.12% |

This is a moderate — not extreme — class imbalance (imbalance ratio ≈ 3.5:1). It does
not automatically justify synthetic resampling; that decision will be made in the
class-imbalance analysis stage after establishing a class-weighted baseline.

## 10. Leakage Check (preliminary)

No column is a deterministic function of, or was measured strictly after, the target
event in a way that would leak the outcome — all `PAY_*`, `BILL_AMT*`, and `PAY_AMT*`
values are drawn from April–September 2005, and the target is default status in the
month *following* the observation window. `ID` is excluded from modeling (Section 4).
A full leakage check will be repeated after feature engineering, since engineered
features are the more common source of subtle leakage in this kind of project.

## 11. Kaggle vs. UCI Documentation — Discrepancies Found

1. Target column header differs in string form (`default payment next month` vs.
   `default.payment.next.month`) — cosmetic, handled programmatically.
2. `EDUCATION` and `MARRIAGE` both contain undocumented category codes (0 on
   `MARRIAGE`; 0, 5, 6 on `EDUCATION`) not defined in the original UCI variable
   description.
3. `PAY_0` is not renamed to `PAY_1` in the file, despite representing the same
   September-2005 repayment status that would logically be "month 1" in the PAY_1…PAY_6
   sequence used for the other variable families.
4. The documented `PAY_*` scale (-1 = pay duly, 1–9 = months delayed) does not
   document a value of 0 or -2, both of which are heavily populated (0 is in fact the
   single most common value in every `PAY_*` column).

None of these discrepancies affect row count or introduce missing data — they are
category-definition and naming issues, handled explicitly in Section 5–6 rather than
silently.

## 12. Summary of Cleaning Decisions Carried Forward

| Issue | Rows affected | Decision |
|---|---:|---|
| EDUCATION codes 0/5/6 | 345 (1.15%) | Group into `Other/Unknown` |
| MARRIAGE code 0 | 54 (0.18%) | Group into `Other` |
| PAY_* undocumented 0/-2 | majority of rows | Retain; treat as ordinal/categorical, not continuous |
| Negative BILL_AMT* | 590–688 per column | Retain as valid (overpayment state); handle in ratio features |
| ID | all rows | Exclude from modeling; retain as identifier |
| Duplicates | 0 | None found; no action needed |
| Missing values | 0 | None found; no imputation needed |

*No rows have been deleted at this stage. Total dataset size remains 30,000 for the
cleaning and feature-engineering steps that follow.*
