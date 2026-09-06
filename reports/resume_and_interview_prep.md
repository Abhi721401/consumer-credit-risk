# Resume & Interview Preparation
## Consumer Credit Risk Analytics & Probability of Default Modeling

*All metrics below are taken directly from `reports/experiment_results.json`
and `reports/model_validation_report.md` — nothing here is invented.*

---

## A. Resume Bullets

- Built an end-to-end Probability of Default (PD) modeling pipeline on a
  30,000-account consumer credit dataset, engineering 39 credit-risk
  features (utilization, delinquency, payment coverage, exposure) and
  comparing Logistic Regression, Random Forest, and XGBoost, achieving a
  0.781 ROC-AUC and 0.437 KS statistic on a held-out test set.
- Applied isotonic calibration to the selected model, improving Brier score
  by 24% (0.178 → 0.135) and producing a PD calibration table accurate to
  within 1.7 percentage points across five risk bands.
- Conducted an evidence-based class-imbalance analysis comparing
  no-resampling, class-weighting, and SMOTE, then built and trained a real
  Variational Autoencoder (PyTorch, encoder/decoder, reparameterization
  trick) exclusively on the minority class to generate synthetic
  default-class samples, fully balanced the training set, and rejected the
  approach after it failed to improve — and for logistic regression,
  worsened — downstream classification performance versus simple
  class-weighting, despite the VAE learning realistic per-feature
  distributions (synthetic means within 0.06 standard deviations of real).
- Designed a 4-tier customer risk-segmentation framework from the model's
  PD distribution, concentrating 63.3% of portfolio exposure in the
  lowest-risk segment while isolating a 2.9%-of-portfolio Very-High-Risk
  segment with a 76.3% observed default rate.
- Built a Streamlit risk-analytics dashboard with a Groq-powered "CreditRisk
  AI Copilot" that explains verified model outputs (PD, SHAP drivers,
  monitoring metrics) in natural language without ever computing risk
  itself — architected so the core analytics function fully even if the
  LLM integration is unavailable.

## B. One-Line Project Description

An end-to-end consumer credit-risk analytics project — data audit through
calibrated PD modeling, risk segmentation, monitoring, and an LLM-assisted
Streamlit dashboard — built on the UCI/Kaggle Default of Credit Card
Clients dataset.

## C. Technical Skills Demonstrated

**Statistics:** descriptive statistics, chi-square tests, Welch's t-tests,
Cohen's d / Cramer's V effect sizes, correlation analysis, calibration
theory (Brier score, isotonic regression), population stability index (PSI).

**Machine Learning:** logistic regression, random forests, gradient
boosting (XGBoost), class-imbalance handling (class-weighting, SMOTE),
probability calibration (`CalibratedClassifierCV`), SHAP explainability,
model comparison and selection methodology.

**Credit Risk:** Probability of Default modeling, credit utilization &
payment-coverage feature design, delinquency-severity modeling, risk
segmentation, early-warning indicators, illustrative Expected Credit Loss
(PD × LGD × EAD) scenario analysis, model-risk-management-inspired
governance documentation.

**Data Engineering:** reproducible ingestion/validation/cleaning/
feature-engineering pipeline with documented leakage prevention, SQLite
analytical layer, modular production-style Python (type hints, docstrings,
logging, no hard-coded paths).

**SQL:** portfolio sizing, segment-level default-rate aggregation,
delinquency analysis, high-risk customer identification, exposure
concentration queries against a SQLite database.

**Visualization:** matplotlib/seaborn static EDA and evaluation charts,
Plotly interactive dashboard visuals, SHAP summary plots.

**Model Governance:** a governance document covering purpose, intended use,
assumptions, limitations, validation methodology, calibration, stability,
explainability, fairness, monitoring, and retraining triggers — explicitly
scoped as a portfolio exercise, not a regulatory-compliance claim.

## D. 60-Second Interview Explanation

"I built an end-to-end credit-risk analytics project on the UCI/Kaggle
Default of Credit Card Clients dataset — about 30,000 accounts. I started
with a full data audit rather than jumping into modeling, which turned up
undocumented category codes in EDUCATION and MARRIAGE and confirmed the
repayment-status columns aren't ordinary continuous variables. From there I
engineered 39 features around utilization, delinquency, and payment
coverage, ran statistical tests to confirm which ones actually separate
defaulters from non-defaulters, and compared three models — logistic
regression as an interpretable baseline, random forest, and XGBoost, which
won with a 0.78 ROC-AUC. I calibrated it with isotonic regression, which
cut the Brier score by about a quarter, and built a 4-tier risk
segmentation that's accurate to within about two points of actual default
rate in every band. I also ran a class-imbalance investigation — I tested
SMOTE, and I built a real VAE with PyTorch, trained it only on the
minority class, generated enough synthetic defaulters to fully balance the
training set, and evaluated it against logistic regression, random forest,
and XGBoost. It learned realistic feature distributions but didn't
actually improve classification performance over simple class-weighting —
so I kept class-weighting, since the more complex method didn't earn its
complexity. Finally, I wrapped it all in a Streamlit
dashboard with a Groq-powered assistant that explains the model's outputs
in plain language, but never calculates risk itself — the ML model is
always the source of truth."

## E. Deep-Dive Interview Questions & Answers

**1. Why logistic regression as a baseline instead of jumping straight to XGBoost?**
It's interpretable — coefficients convert directly to odds ratios a
business stakeholder can read (e.g., higher payment-amount volatility →
~2x higher odds of default). It's also a useful sanity check: its 0.758
ROC-AUC is only ~2.4 points behind XGBoost's 0.781, which tells you most of
the signal in this problem is fairly linear/additive, and any complexity
beyond that needs to earn its keep.

**2. Why is accuracy inappropriate here?**
With a 78/22 class split, a model that predicts "no default" for everyone
scores ~78% accuracy while catching zero actual defaulters. I used
ROC-AUC, PR-AUC, and KS instead, which measure ranking/separation quality
independent of the base rate.

**3. Why did you (or didn't you) use a VAE?**
I was asked to investigate it, not to use it by default, so I built a real
one — PyTorch, encoder/decoder, reparameterization trick, trained
exclusively on the training fold's minority class — and generated enough
synthetic defaulters to fully balance the training set. I checked the
synthetic data's quality first: feature means were within about 0.06
standard deviations of the real minority class on average, which is
genuinely good. But when I trained logistic regression, random forest, and
XGBoost on the VAE-balanced data and compared to simple class-weighting,
the VAE version was flat-to-worse across the board — it actually hurt
logistic regression's ROC-AUC (0.756 → 0.737) while giving less recall
improvement than class-weighting gets for free. My read is that a VAE
trained only on the minority class learns to interpolate within what it's
already seen, but doesn't add new information about the decision boundary
the way reweighting the loss function does. So I kept class-weighting —
not because the VAE was poorly built, but because it didn't earn its
complexity even when built properly.

**4. How did you prevent data leakage?**
Split before any fitting; every transform that learns parameters (scaler,
one-hot encoder, imputer) is inside an sklearn Pipeline/ColumnTransformer
fit only on the training fold; SMOTE (when tested) was applied only inside
the training fold, never before the split; the test set was touched exactly
once, at final evaluation.

**5. How did you handle class imbalance?**
Quantified it first (3.5:1, moderate) rather than assuming it needed
treatment, then compared no-resampling, class-weighting, and SMOTE
empirically. Class-weighting won on simplicity for equivalent performance.

**6. What is PD, precisely, in this project?**
The model's calibrated estimated probability that a given customer will
default on their payment in the month immediately following the observed
6-month window — a point-in-time estimate, not a lifetime default
probability.

**7. How do you evaluate calibration?**
I compare raw vs. isotonic-calibrated probabilities by Brier score, then
build a calibration table bucketing customers by predicted PD and checking
whether the actual observed default rate in each bucket matches. In my
final model, predicted and actual are within 1.7 points in every bucket.

**8. What is KS (Kolmogorov-Smirnov) in this context?**
The maximum separation between the cumulative distribution of predicted PD
for actual defaulters vs. non-defaulters. My best model's KS is 0.437 —
higher KS means the model separates the two populations better at some
threshold.

**9. What is PSI, and what did you find?**
Population Stability Index — measures how much a distribution (a feature,
or the PD score itself) has shifted between two periods. I computed it
between my train and test PD distributions (0.0012, negligible, as
expected for a random split of the same period) mainly to demonstrate the
calculation; a real deployment needs a true out-of-time batch to make this
meaningful, which this static dataset doesn't provide.

**10. Why does model monitoring matter for a PD model?**
Credit behavior and macroeconomic conditions drift over time; a model
validated on 2005 data could silently degrade if redeployed later without
monitoring. I built out performance monitoring (AUC/KS/Brier/calibration),
feature- and prediction-level PSI, and six documented retraining triggers.

**11. How would you handle model drift in production?**
Watch the six triggers I documented (PSI thresholds, AUC/Brier degradation,
calibration-table divergence, feature-level PSI, policy/macro changes);
when triggered, investigate root cause before automatically retraining,
since drift can reflect a genuine population shift the model should learn
or a data-pipeline bug that retraining would just memorize.

**12. Why are PAY_0 through PAY_6 important, and how did you treat them?**
They're repayment-status history — the single strongest predictor family in
the data (SHAP confirms PAY_0 and MAX_DELINQUENCY as the top two global
drivers). I treated them as ordinal/categorical, not naively continuous,
and built explicit engineered summaries (months delinquent, max severity,
persistence flags) on top of the raw codes.

**13. What are the limitations of this dataset?**
Single 2005 Taiwanese snapshot, no macroeconomic variables, no real LGD/EAD
outcome data, no multi-period outcome tracking, and a handful of
undocumented category codes that required judgment calls I've disclosed
rather than hidden.

**14. How would you implement this in production?**
Wrap the scoring pipeline (`src/scoring.py`) behind a service boundary that
takes raw customer records and returns PD/risk band, with the monitoring
job running on a schedule against real outcomes as they arrive, and a
documented retraining/approval workflow rather than silent auto-retraining.

**15. How would you handle LGD/EAD, which this dataset doesn't provide?**
I was explicit that this project doesn't have real LGD/EAD — I used current
credit limit as an illustrative EAD proxy and treated LGD as a scenario
parameter (tested at 45/60/75%) rather than pretending to have modeled
either one. A real implementation would need actual charge-off/recovery
data to fit LGD and a real exposure-at-default definition tied to the
product's terms.

**16. How would you explain a high-risk customer to a business stakeholder?**
I'd point to the SHAP breakdown for that specific customer — for example,
"this customer's estimated default probability is elevated primarily
because of six consecutive months of delinquency and a recent repayment
status showing they're currently behind, not because of any demographic
factor" — and I'd be explicit that this is a model estimate, not a
certainty or a lending decision.

**17. What would you do differently with more time/data?**
Get a genuine out-of-time sample to validate stability properly, tune the
classification threshold against a real cost-of-error framework instead of
using 0.5, and re-run the imbalance comparison (class-weight vs. SMOTE, not
just the VAE) directly on the tree-based models rather than only on
logistic regression.

**18. Why exclude ID from the model?**
It's a synthetic sequential row identifier (verified 1-to-30,000, no
duplicates) with no behavioral meaning — including it risks the model
picking up a spurious artifact of how the dataset was assembled rather than
a real credit-risk signal.

**19. How did you decide on the risk-segmentation thresholds?**
From the training-set PD distribution's 50th/80th/95th percentiles, not
round numbers — so segment sizes reflect the actual shape of the risk
distribution, and the resulting bands are still well-calibrated against
actual outcomes on the test set.

**20. What's the architecture of the AI Copilot, and why build it that way?**
DATA → ANALYTICS/ML → VERIFIED METRICS → LLM EXPLANATION. The LLM only ever
sees precomputed, verified numbers as JSON context and is explicitly
instructed not to calculate risk or make lending decisions itself — that
separation matters because an LLM hallucinating a risk score would be a
much worse failure mode than an LLM giving a slightly awkward explanation
of a correct one.

**21. What happens if the Groq API is down or the key is missing?**
The copilot page shows a clear "currently unavailable" message and every
other page of the dashboard — the actual risk analytics — keeps working
without interruption. The LLM is additive, never load-bearing.

**22. Tell me about a mistake you found in this project and how you handled it.**
On a self-review pass, I found that my risk-segmentation thresholds (the
50th/80th/95th percentile cutoffs defining Low/Moderate/High/Very-High
Risk) were computed from the test set's own predicted-PD distribution
instead of the training set's. That's circular — bucketing the test set
using quantiles derived from that same test set, then reporting how
well-calibrated the buckets are, flatters the result. I fixed it to compute
thresholds from training-set predictions only and re-ran the full pipeline.
In this case the fix changed the thresholds by less than half a percentage
point, because my train/test split is IID from the same time period (PSI
0.0012) — but I documented the bug, the fix, and why the impact happened to
be small here rather than quietly correcting it and moving on, because the
same mistake on a dataset with real train/test drift would matter a lot
more. I also flagged two other things on that same review: my
class-imbalance comparison only tested logistic regression, not the final
XGBoost model (I later closed part of this gap by re-running the VAE
comparison directly on all three models); and my calibration comparison
partly conflates isotonic calibration's benefit with the implicit
ensembling effect of `CalibratedClassifierCV`'s internal cross-fitting.
Separately, I'd initially implemented my "VAE" as a PCA-based linear
approximation because I couldn't fit PyTorch's default install in my
environment's disk budget — I didn't let that stand, found a way to
install a CPU-only build without its normally-bundled CUDA dependencies,
and rebuilt the experiment with a real encoder/decoder VAE. The corrected
version actually changed the finding's strength, not just its honesty: the
real VAE learned realistic feature distributions (much better than the
PCA version), which makes the "still doesn't beat class-weighting" verdict
more convincing, not less. None of these break the project's core
conclusions, but I'd rather surface them than have someone else find them
first.
