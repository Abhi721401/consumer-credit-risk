# Consumer Credit Risk Analytics & Probability of Default Modeling
## Final Analysis Report

*An independent consumer credit-risk analytics project using a publicly
available dataset. All figures in this report were computed by the code in
`src/` against the real data — see `reports/experiment_results.json` for the
raw machine-readable results this report is built from.*

## Executive Summary

**Business problem:** estimate a consumer credit-card customer's Probability
of Default (PD) to support risk segmentation, early-warning monitoring, and
portfolio-level decision support — the kind of analytical foundation a
consumer-credit risk analytics team relies on.

**Dataset:** UCI/Kaggle "Default of Credit Card Clients" — 30,000 Taiwanese
credit-card accounts, April–September 2005, 25 raw fields, binary target
(22.12% default rate).

**Methodology:** full audit → non-destructive cleaning → 39 engineered
credit-risk features → statistical testing → evidence-based class-imbalance
decision → 3-model comparison → isotonic calibration → SHAP explainability →
risk segmentation → monitoring framework → fairness analysis → illustrative
ECL scenario → SQL layer → Streamlit dashboard with an LLM copilot.

**Best model:** XGBoost — ROC-AUC 0.781, PR-AUC 0.562, KS 0.437, calibrated
Brier score 0.135 (down from 0.178 uncalibrated) on a held-out 7,500-customer
test set.

**Key finding:** repayment/delinquency history dominates every other
predictor family. `N_MONTHS_DELINQUENT` has the strongest correlation with
default (r=0.398) and largest effect size (Cohen's d=0.87) of any feature
tested; SHAP confirms `PAY_0` (most recent repayment status) and
`MAX_DELINQUENCY` as the top two global drivers of the final model.
Demographic variables (sex, education, marital status) are statistically
significant but practically small (Cramer's V 0.03–0.07) — consistent with
how real bureau-style credit scores are built.

**Business implications:** a utilization-and-recent-delinquency "stress"
flag alone separates a 56.2% default-rate population from a 19.5%
baseline — a strong, simple early-warning signal. The 4-tier risk
segmentation concentrates 63.3% of portfolio exposure in the Low-Risk band
and only 3.0% in Very High Risk, with predicted PD tracking actual default
rate within 1.7 points across every band.

## Data Quality

Full detail in `reports/data_quality_report.md`. Headlines: 0 missing
values, 0 duplicate rows/IDs across 30,000 records; `EDUCATION` (345 rows,
1.15%) and `MARRIAGE` (54 rows, 0.18%) contain undocumented category codes,
folded into existing "Other/Unknown" buckets rather than dropped; `PAY_*`
repayment-status columns contain undocumented values 0 and -2 that are in
fact the *most common* values in every column, confirming they cannot be
treated as ordinary continuous inputs.

## EDA — Key Findings

- **Delinquency is the dominant signal.** Default rate rises monotonically
  from 11.71% (0 delinquent months of the last 6) to 70.32% (6 of 6) — see
  `reports/figures/default_rate_by_delinquency.png`.
- **Utilization compounds with delinquency.** The `UTILIZATION_STRESS_FLAG`
  (utilization > 80% AND recent delinquency) population defaults at 56.2%
  vs. 19.5% for everyone else.
- **Age and education show real but modest gradients.** Default rate rises
  from 20.30% (25–34) to 27.19% (<25) and 26.69% (55+) — a mild U-shape;
  education shows a range from 7.05% (Other/Unknown — small, 468-person
  group) to 25.16% (High School).
- **Credit limit is protective.** Average limit for non-defaulters
  ($178,100) is meaningfully higher than for defaulters ($130,110),
  Cohen's d = -0.39 — banks were already pricing risk into limits, which
  the model picks up.

## Feature Engineering

39 engineered features across five families — utilization, payment
coverage, delinquency, balance/payment trend & volatility, and credit
exposure — each with a stated credit-risk rationale (see
`src/feature_engineering.py` docstrings for the full reasoning, including
explicit handling of negative bills and zero/negative denominators).

## Statistical Analysis

Chi-square tests confirm SEX, EDUCATION, and MARRIAGE are all associated
with default at p<0.001, with small effect sizes (Cramer's V 0.031–0.073).
Welch's t-tests on numeric features show `N_MONTHS_DELINQUENT` and
`MAX_DELINQUENCY` with the largest effect sizes (Cohen's d = 0.87 and 0.86
respectively) of any variable tested — by a wide margin the strongest
behavioral signal in the data. All results are reported as **association**,
not causation, throughout.

## Class Imbalance & the VAE Experiment

Moderate imbalance (3.5:1) did not justify synthetic oversampling: SMOTE
improved ROC-AUC by only 0.0004 over simple class-weighting. A real VAE
(encoder/decoder network, reparameterization trick, KL-regularized
training — implemented in `src/vae_model.py`) was trained exclusively on
the training fold's minority class, generated enough synthetic samples to
fully balance the training set, and was evaluated on Logistic Regression,
Random Forest, and XGBoost. It learned realistic per-feature marginals
(synthetic means within 0.059 standard deviations of real minority-class
means, on average) but **fully balancing the training set with VAE-
generated samples did not improve, and for logistic regression actively
hurt, classification performance** compared to simple class-weighting —
see `model_validation_report.md` §2 for the full comparison table. Final
choice: class-weighting.

## Modeling & PD Estimation

Logistic Regression (interpretable baseline), Random Forest, and XGBoost
were compared on a held-out test set never touched during training or
tuning. XGBoost won on the primary criterion (ROC-AUC+PR-AUC)/2 and on
recall (64.4%) and KS (0.437). Full comparison table and confusion matrices
in `model_validation_report.md` §3.

## Model Validation, Calibration & Stability

Isotonic calibration improved Brier score from 0.178 to 0.135 (24%
relative). The resulting calibration table shows predicted PD within 1.7
points of actual default rate in every one of five risk bands — genuinely
strong calibration, not a hedge. PSI (train vs. test PD) = 0.0012,
demonstrating the calculation; a real out-of-time monitoring window would be
needed to assess true production stability (this dataset is a single
snapshot — see Limitations).

## Risk Segmentation

Four bands (Low / Moderate / High / Very High Risk) from the 50th/80th/95th
percentiles of the training PD distribution:

| Band | Test Customers | Avg. Predicted PD | Actual Default Rate | % of Portfolio Exposure |
|---|---:|---:|---:|---:|
| Low Risk | 3,728 | 9.2% | 9.1% | 63.3% |
| Moderate Risk | 2,261 | 21.2% | 20.8% | 23.9% |
| High Risk | 1,152 | 50.7% | 49.9% | 9.9% |
| Very High Risk | 359 | 76.4% | 76.3% | 2.9% |

*(Thresholds are computed from the training-set PD distribution and applied
to the test set — an earlier version of this pipeline used test-set
quantiles, which was circular; see `model_validation_report.md` §5 for the
correction. The fix changed these numbers by well under 1 point, since
train and test are IID splits of the same period.)*

## Early-Warning Indicators

The strongest indicators, in order of demonstrated effect size: recent
repayment status (`PAY_0`), lifetime max delinquency severity, count of
delinquent months (of the last 6), and the combined utilization-stress flag.
These are risk indicators / predictors — not causes of default.

## Model Monitoring

Performance monitoring (ROC-AUC/PR-AUC/KS/Brier/calibration), feature- and
prediction-level PSI, and a documented set of six retraining triggers are
implemented in `src/monitoring.py` and rendered on the dashboard's Model
Monitoring page. See `model_governance.md` §15 for the full trigger list.

## Responsible AI / Fairness

TPR/FPR and calibration were compared across SEX, AGE band, and EDUCATION on
the test set. No stark disparities emerged at this sample size (e.g., a
~2.5-point recall gap by sex), but gaps of this kind warrant ongoing
monitoring in any real deployment, not a one-time check. Demographic fields
were retained specifically so this comparison could be made, per the
project's fairness-analysis requirement. Full tables in
`reports/experiment_results.json` → `fairness`.

## Illustrative Expected Credit Loss (Scenario Analysis)

**This dataset provides no real LGD or EAD outcome data.** The figures below
use current credit limit as an illustrative EAD proxy and treat LGD as a
scenario assumption — this is explicitly a teaching exercise in
EL = PD × LGD × EAD mechanics, not a loss forecast:

| LGD Scenario | Total Illustrative Expected Loss (test population) | Avg. per Account |
|---|---:|---:|
| 45% | $98.5M | $13,135 |
| 60% | $131.4M | $17,514 |
| 75% | $164.2M | $21,892 |

## Business Recommendations (for a hypothetical consumer-credit analytics team)

1. Prioritize behavioral monitoring (delinquency trend, utilization stress)
   over demographic segmentation — behavioral features carry far more signal.
2. The `UTILIZATION_STRESS_FLAG` is simple enough to operationalize as a
   real-time early-warning rule ahead of any model refresh cycle.
3. At the default 0.5 threshold, ~36% of actual defaulters are missed
   (recall 64.4%) — a real deployment should tune the threshold against an
   explicit false-negative/false-positive cost analysis rather than using 0.5.
4. Given the small demographic effect sizes but non-zero fairness gaps
   found, any production use should include ongoing fairness monitoring,
   not a one-time check at model launch.

## Limitations

- **Dataset age & geography:** 2005, Taiwan — cardholder behavior and
  macroeconomic conditions have changed substantially since; results should
  not be assumed to transfer to another market or era without revalidation.
- **Sample characteristics:** credit-card-holder population only, not a
  representative sample of all consumers.
- **No macroeconomic variables** (rates, unemployment, etc.) — the model
  cannot anticipate systemic shocks.
- **No real LGD/EAD** — ECL figures are illustrative scenario analysis only.
- **No longitudinal outcome data** beyond the single next-month target —
  can't assess how PD evolves over a customer's full lifecycle.
- **Public dataset limitations:** undocumented category codes, ambiguous
  `PAY_*` semantics (see data quality report) required judgment calls that
  are disclosed but could differ from the original data generators' intent.
- **Potential fairness/generalization issues:** demographic fairness checked
  only on this dataset's population and time period; would need re-validation
  on any new population before relying on it.
