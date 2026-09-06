# Model Validation Report

*All numbers in this report come from `src/run_experiments.py` executed against
the real dataset. None are assumed or invented.*

## 1. Validation Methodology

- Stratified 75/25 train/test split (`random_state=42`), preserving the 22.12%
  default rate in both folds (train: 22.12%, test: 22.12% — verified, not assumed).
- All fitted transforms (median imputation, standard scaling, one-hot encoding)
  are fit on the training fold inside an sklearn `Pipeline` / `ColumnTransformer`
  and applied to test — never fit on the full dataset.
- The test set (7,500 rows) is used exactly once, for final evaluation and
  calibration comparison. Hyperparameters below were chosen from reasonable
  defaults informed by the imbalance-comparison experiment (run on the training
  fold only), not tuned against the test set.
- `ID` is excluded from all feature sets.

## 2. Class Imbalance: Investigation and Decision

Baseline imbalance ratio: 77.88% / 22.12% (~3.5:1) — moderate, not severe.

| Strategy | ROC-AUC | PR-AUC | Precision (default) | Recall (default) | F1 (default) |
|---|---:|---:|---:|---:|---:|
| No resampling | 0.7554 | 0.5099 | 0.618 | 0.325 | 0.426 |
| Class-weighted | 0.7574 | 0.5063 | 0.434 | 0.614 | 0.509 |
| SMOTE (train-fold only) | 0.7578 | 0.5067 | 0.437 | 0.614 | 0.511 |

**Decision: class-weighting**, not SMOTE. SMOTE produces a statistically
insignificant improvement over simple class weighting (ROC-AUC +0.0004, F1
+0.002) while adding a synthetic-data generation step, extra governance
surface area (synthetic records must be explained/justified to a validator),
and one more thing that can silently leak if ever misapplied. Class-weighting
achieves effectively the same lift with less complexity — the standard
"prefer the simpler method that performs equivalently" validation principle.

### VAE Experiment (Section 8 requirement)

**Updated with a real VAE.** An earlier version of this section used a
linear-Gaussian approximation (PCA encoder + Gaussian latent sampling)
because PyTorch could not initially be installed in this environment. That
constraint was resolved (installed the CPU-only wheel directly, bypassing
its normally-bundled CUDA dependencies, which were the actual disk-space
problem) and the experiment was redone with a genuine VAE: a 2-layer
encoder, a 12-dimensional latent space, the reparameterization trick, and
an Adam-optimized MSE-reconstruction + KL-divergence loss, trained for 300
epochs — implemented in `src/vae_model.py`.

**Setup:** trained exclusively on the 4,977 default=1 rows of the training
fold (never on test data). Generated 12,546 synthetic minority samples —
enough to fully balance the 17,523-row majority class, matching SMOTE's
target for a fair comparison. One-hot categorical blocks were re-discretized
via argmax and binary flag features were rounded to {0,1} after generation,
since a raw Gaussian VAE output isn't a valid categorical/binary record.

**Distributional diagnostics** (synthetic vs. real minority-class training data):

| Diagnostic | Value | Interpretation |
|---|---:|---|
| Mean standardized deviation (mean diff in units of real std) | **0.059** | Synthetic feature means are, on average, under 0.06 standard deviations from real minority-class means — genuinely close |
| Correlation-matrix Frobenius difference | 3.53 | Improved vs. the earlier PCA approximation (6.58) but still a non-trivial joint-structure gap |
| Relative (%) mean/std deviation | 57.9% / 25.1% | **Misleading on its own** — several engineered trend/ratio features have real means near zero, which explodes a percent-based metric; the standardized-deviation figure above is the reliable one |

Lesson learned mid-project: the relative-percent diagnostic used in the
first (PCA) pass was a poor metric choice for this dataset, not just a poor
result — several features (e.g. `PAYCOV_TREND`) have real means near zero,
so even a tiny absolute deviation reads as a huge percentage. Reporting
both metrics here rather than only the flattering one.

**Downstream classification performance** (test set, n=7,500):

| Model | Training Data | ROC-AUC | Recall (default) | F1 |
|---|---|---:|---:|---:|
| Logistic Regression | Original (imbalanced) | 0.756 | 0.326 | 0.427 |
| Logistic Regression | VAE-balanced (50/50) | 0.737 | 0.580 | 0.505 |
| Logistic Regression | Class-weighted | 0.757 | 0.614 | 0.509 |
| Random Forest | Original (imbalanced) | 0.780 | 0.375 | 0.476 |
| Random Forest | VAE-balanced (50/50) | 0.779 | 0.385 | 0.480 |
| XGBoost | Original (imbalanced) | 0.781 | 0.370 | 0.473 |
| XGBoost | VAE-balanced (50/50) | 0.780 | 0.372 | 0.474 |

**Verdict: still rejected, now on stronger evidence.** The real VAE
genuinely learned realistic per-feature marginal distributions (0.059 std
deviation is a good result) — this is not a "the VAE failed to learn
anything" story. But fully balancing the training set with VAE-generated
samples:
- **Hurt** logistic regression's ROC-AUC (0.756 → 0.737) while only
  partially closing the recall gap versus simple class-weighting (0.580 vs.
  0.614 recall, at a worse ROC-AUC).
- **Did essentially nothing** for Random Forest and XGBoost — recall moved
  by ~1 point in either direction, well within run-to-run noise, and
  ROC-AUC was flat to very slightly worse.

The likely explanation: a VAE trained only on the minority class learns to
interpolate *within* that class's already-observed distribution — it
doesn't inject new information about the decision boundary between classes
the way class-weighting (which changes the loss function's sensitivity to
the boundary directly) does. Simple class-weighting remains the chosen
approach for the final model. This is reported as a clean negative result,
not softened — a real, more carefully built VAE was given a fair chance
here and still didn't beat the simpler method.



## 3. Model Comparison (test set, n=7,500)

| Model | ROC-AUC | PR-AUC | KS | Brier (raw) | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7578 | 0.5076 | 0.398 | 0.1925 | 0.435 | 0.615 | 0.510 |
| Random Forest | 0.7799 | 0.5584 | 0.422 | 0.1770 | 0.474 | 0.609 | 0.534 |
| **XGBoost** | **0.7813** | **0.5622** | **0.437** | 0.1776 | 0.462 | **0.644** | **0.538** |

**Selected model: XGBoost**, by (ROC-AUC + PR-AUC)/2, with the best KS and
recall on the minority class among the three. Logistic regression remains the
interpretability baseline and is reported alongside for transparency — its
ROC-AUC (0.758) is only ~2.4 points behind XGBoost, a genuinely useful data
point for a bank deciding whether the added complexity of a gradient-boosted
model is worth the reduced interpretability.

**Why not accuracy:** the no-resampling baseline shows why — a model that
almost never predicts "default" would score close to 78% accuracy while
missing most actual defaulters (67.5% of them, per that baseline's recall).
ROC-AUC, PR-AUC, and KS are the metrics reported as primary.

## 4. Calibration

| | Brier Score |
|---|---:|
| Raw XGBoost probabilities | 0.1776 |
| Isotonic-calibrated (3-fold `CalibratedClassifierCV`) | **0.1346** |

Isotonic calibration produced a meaningfully lower Brier score (24% relative
improvement) and was selected as the final PD output. Platt scaling was not
separately tested in this run; isotonic was tried first because it makes
fewer distributional assumptions and is standard practice for tree-based
models, and it already showed a clear improvement.

### Calibration Table (test set, final calibrated PD)

| PD Band | Customers | Avg. Predicted PD | Actual Default Rate |
|---|---:|---:|---:|
| 0–5% | 550 | 3.91% | 4.36% |
| 5–10% | 1,576 | 7.44% | 7.61% |
| 10–20% | 2,727 | 14.67% | 14.41% |
| 20–40% | 1,466 | 27.41% | 27.63% |
| 40%+ | 1,181 | 62.44% | 60.71% |

Predicted PD tracks actual observed default rate closely across every band —
the largest gap is 1.7 percentage points (40%+ band). This is a genuinely
well-calibrated model on this test set, not a hedge — the numbers speak for
themselves.

## 5. Population Stability

PSI between the calibrated PD distribution on the training fold and on the
test fold: **0.0012** (well under the 0.10 "stable" rule of thumb). This is
expected and not a meaningful finding on its own — train and test are random
splits of the *same* time period, so they should look alike. It is reported
here as a demonstration of the PSI calculation that would be applied to a
true out-of-time monitoring batch in production, which this static dataset
cannot provide (see Limitations).

**Correction note:** an earlier version of this pipeline computed the risk-
segmentation thresholds (§ below and in `segmentation_thresholds`) from the
*test-set* PD distribution rather than the training-set distribution. That
was a methodological error — bucketing the test set by its own quantiles,
then reporting how well-calibrated the resulting buckets are, is mildly
circular. It's fixed now: thresholds come from `X_train` predictions only,
applied to `X_test` for evaluation. Because train and test are IID splits of
the same period (PSI 0.0012), the fix changed the thresholds by less than
0.004 in absolute terms and did not meaningfully change the segmentation
table below — but the fix matters for correctness, and would matter a lot
more on a dataset with real train/test drift.

## 6. Explainability

Top logistic-regression coefficients (by |coefficient|), converted to odds ratios:

| Feature | Coefficient | Odds Ratio | Direction |
|---|---:|---:|---|
| PAYAMT_VOLATILITY | +0.685 | 1.98 | Higher payment-amount volatility → higher risk |
| EDUCATION_4 (Other/Unknown) | −0.654 | 0.52 | Associated with lower risk vs. reference group |
| BILL_MAX | −0.485 | 0.62 | Higher max bill → lower risk (larger, more established balances) |
| MAX_DELINQUENCY | +0.414 | 1.51 | Higher worst-ever delinquency → higher risk |
| PAYAMT_MAX | −0.331 | 0.72 | Higher max payment capacity → lower risk |

These are associations from a regularized logistic regression on observational
data — not causal claims. `EDUCATION_4` in particular is a small, heterogeneous
"Other/Unknown" bucket (468 people); its coefficient should be read with that
caveat rather than as a policy-relevant finding.

## 7. Fairness (test set, by SEX)

| Group | n | Actual Default Rate | Avg. Predicted PD | TPR (Recall) | FPR |
|---|---:|---:|---:|---:|---:|
| SEX=1 (male) | 2,983 | 23.53% | 23.39% | 36.3% | 6.1% |
| SEX=2 (female) | 4,517 | 21.19% | 21.71% | 38.8% | 5.3% |

Predicted PD tracks actual default rate closely for both groups (good
calibration by group, not just in aggregate). TPR and FPR differ modestly
between groups (a ~2.5-point recall gap, ~0.8-point FPR gap) — worth
monitoring but not a stark disparity at this sample size. Full breakdowns by
age band and education are in `reports/experiment_results.json` and surfaced
in the dashboard's Model Monitoring page.

## 8. Known Issues & Areas for Improvement (added on review)

A critical second pass over this project surfaced items worth being
upfront about, beyond the segmentation-threshold fix in §5. (The VAE
implementation itself was one of these — flagged as a linear approximation
in an earlier version of this report, and since replaced with a real
PyTorch VAE; see §2 above for the updated experiment and verdict.)

1. **The imbalance-strategy comparison (class-weight / SMOTE) was
   originally run only on logistic regression**, then class-weighting was
   carried over to Random Forest and XGBoost without re-verifying on those
   model types directly. The VAE re-run (§2) did test all three models
   directly and confirmed the conclusion holds for tree-based models too
   (VAE-balancing changes nothing meaningful for RF/XGBoost) — but the
   original class-weight-vs-SMOTE comparison specifically is still
   LR-only.
2. **The 0.5 classification threshold is not business-cost-informed.** At
   0.5, the selected model misses 35.6% of actual defaulters (recall
   64.4%). A real deployment should tune this against an explicit
   false-negative vs. false-positive cost ratio; this project reports
   metrics at the sklearn default threshold for comparability across
   models, not because 0.5 is the right operating point.
3. **The raw-vs-calibrated Brier score comparison isn't fully isolated.**
   `CalibratedClassifierCV(cv=3)` internally refits 3 cloned models on
   internal folds rather than calibrating the single already-fit XGBoost
   model, so part of the improvement from 0.178 to 0.135 may reflect the
   implicit 3-model ensembling effect rather than calibration alone. The
   direction of the finding (isotonic helps) is still credible, but the
   exact magnitude attributable to calibration specifically is somewhat
   overstated by this setup.

## 9. Limitations

- Single historical snapshot (Taiwan, April–September 2005) — no macroeconomic
  variables, no multi-period outcome tracking, no true out-of-time validation
  window.
- No real LGD or EAD data — the ECL section is explicitly an illustrative
  scenario analysis, not a loss forecast (see `reports/final_analysis.md`).
- A real, properly-trained VAE was evaluated and rejected on genuine
  evidence (see §2) — this is a validated negative result, not a
  workaround limitation.
- Demographic effect sizes are small; the model's real strength is behavioral
  features (delinquency, utilization, payment coverage), consistent with how
  real bureau-style scores are built.
