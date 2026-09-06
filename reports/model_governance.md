# Model Governance Document

**This is a portfolio project inspired by model-risk-management principles.
It does not claim compliance with Basel, SR 11-7, OCC guidance, RBI
regulations, or any other regulatory framework, and was not reviewed by any
regulatory or institutional body.**

## 1. Model Purpose

Estimate a consumer credit-card customer's Probability of Default (PD) on
their next payment, to support portfolio-level risk analytics: segmentation,
early-warning monitoring, and illustrative expected-loss scenario analysis.

## 2. Intended Use

Educational / portfolio demonstration of a PD modeling workflow for a junior
credit-risk / data analyst role. **Not intended for, and must not be used
for, actual lending decisions, credit-limit changes, or any action affecting
a real customer.**

## 3. Data Sources

UCI Machine Learning Repository / Kaggle mirror: "Default of Credit Card
Clients" dataset (Yeh & Lien, 2009). 30,000 Taiwanese credit-card accounts,
April–September 2005. Publicly available, de-identified, static snapshot.

## 4. Target Definition

`default payment next month` (renamed `default_next_month` in this project):
binary indicator, 1 = customer defaulted on payment in the month following
the observed 6-month window, 0 = did not.

## 5. Feature Definitions

See `src/feature_engineering.py` docstrings for the full, code-level
definition and rationale of every engineered feature. Categories:
utilization (6 features), payment coverage (4), delinquency (6), balance/
payment trend & volatility (10), credit exposure (3), plus the 20 raw
LIMIT_BAL/AGE/PAY_*/BILL_AMT*/PAY_AMT* fields and 3 cleaned categorical
fields (SEX, EDUCATION, MARRIAGE).

## 6. Modeling Methodology

- Stratified 75/25 train/test split, `random_state=42`.
- Class-imbalance treatment: class-weighting (`class_weight="balanced"` /
  `scale_pos_weight`), selected over SMOTE and a VAE-based approach after
  evidence-based comparison (see `model_validation_report.md` §2).
- Candidate models: Logistic Regression (interpretable baseline), Random
  Forest, XGBoost. XGBoost selected as final model by (ROC-AUC+PR-AUC)/2.
- Calibration: isotonic regression via 3-fold `CalibratedClassifierCV`,
  selected over raw probabilities by Brier-score improvement.
- Risk segmentation: 4 bands (Low/Moderate/High/Very High) from the 50th,
  80th, and 95th percentiles of the calibrated PD distribution on the
  training population.

## 7. Assumptions

- The 2005 Taiwanese consumer-credit population's default drivers are
  assumed to generalize well enough to illustrate a modeling methodology —
  **not** assumed to generalize to any other geography, era, or product.
- PAY_* codes 0 and -2 (undocumented in the original data dictionary) are
  interpreted as "not delinquent" states based on their empirical position
  in the distribution (they are the most common values), not confirmed by
  original source documentation.
- The illustrative ECL framework assumes fixed LGD scenarios (45%/60%/75%)
  in the absence of real loss data — these are scenario parameters, not
  fitted or estimated values.

## 8. Limitations

See `model_validation_report.md` §9 for the full list. Headline items:
no macroeconomic inputs, no multi-period outcome tracking, no true
out-of-time validation sample, no real LGD/EAD data, and small effect sizes
for demographic variables. (The VAE experiment now uses a real
encoder/decoder network — see §2 — and was rejected on genuine downstream-
performance evidence, not an implementation shortcut.)

## 9. Validation Methodology & Performance

See `model_validation_report.md` in full. Summary: XGBoost, ROC-AUC 0.781,
PR-AUC 0.562, KS 0.437, Brier (calibrated) 0.135 on a held-out 7,500-row
test set untouched until final evaluation.

## 10. Calibration

Isotonic-calibrated PD tracks actual default rate within 1.7 percentage
points across all five reported risk bands on the test set (see calibration
table in `model_validation_report.md` §4).

## 11. Stability

PSI (train PD distribution vs. test PD distribution) = 0.0012 — reported as
a methodology demonstration; this dataset cannot support a genuine
out-of-time stability check (single snapshot, no repeated observation
periods). A production deployment of a model like this would require PSI
monitoring against a true post-deployment scoring population, tracked over
time, not a random train/test split.

## 12. Explainability

Logistic-regression coefficients / odds ratios and XGBoost feature
importances are reported in `model_validation_report.md` §6 and rendered in
the dashboard's "PD Model" page. SHAP values are computed for the selected
model in `src/eda_plots.py` / dashboard integration for local (per-customer)
explanation on the Customer Risk page.

## 13. Fairness Considerations

Default rates, predicted PD, TPR, and FPR were compared across SEX, AGE
band, and EDUCATION on the held-out test set (`model_validation_report.md`
§7). No group showed a stark disparity at this sample size, but the gaps
that do exist (e.g., a ~2.5-point recall gap by sex) are exactly the kind of
finding a real deployment would need to monitor over time and across a
larger population before drawing conclusions. Demographic fields were
**retained** in this exercise (rather than removed pre-emptively) specifically
so that their relationship to model outcomes could be measured and reported,
consistent with the project's fairness-analysis requirement — a real
deployment decision about whether to use them as direct model inputs would
involve legal and compliance review beyond the scope of this project.

## 14. Monitoring Framework

See `src/monitoring.py`. Performance metrics (ROC-AUC, PR-AUC, KS, Brier,
calibration), feature-level PSI, and prediction-level PSI are all
implemented and demonstrated. Rendered in the dashboard's "Model Monitoring"
page.

## 15. Retraining Triggers

1. PSI on predicted PD exceeds 0.25 vs. the training-time reference distribution.
2. Rolling-window ROC-AUC drops more than 0.03 below the validated test-set ROC-AUC.
3. Rolling-window Brier score materially increases vs. the validated test-set Brier score.
4. A risk band's actual default rate diverges from its average predicted PD by
   more than ~30% relative, for two consecutive monitoring periods.
5. Any input feature shows PSI > 0.25 versus its training distribution.
6. A material change in underwriting policy, credit-limit strategy, or the
   macroeconomic environment that would be expected to shift the underlying
   relationship between features and default risk.

## 16. Known Weaknesses

- Dataset age (2005) and single geography (Taiwan) limit generalizability.
- No true production monitoring window exists to validate the monitoring
  framework's practical sensitivity.
- A real VAE (encoder/decoder, reparameterization trick) was trained and
  evaluated for minority-class augmentation and rejected on genuine
  evidence — it learned realistic feature marginals but did not improve,
  and for logistic regression worsened, downstream classification
  performance versus simple class-weighting (see `model_validation_report.md` §2).
- Recall on the minority class, even after class-weighting, tops out at
  64.4% (XGBoost) — meaning over a third of actual defaulters are not
  flagged at the default 0.5 threshold. A real deployment would tune this
  threshold against a business cost-of-error analysis (false negative vs.
  false positive cost), which is outside the scope of this project.
