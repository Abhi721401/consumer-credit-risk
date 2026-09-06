"""
Consumer Credit Risk Analytics — Streamlit Dashboard
=====================================================
Portfolio project. Not a production lending system — see the
disclaimer in the sidebar and on the Customer Risk page.

Run with:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from llm.groq_client import CreditRiskCopilot  # noqa: E402

st.set_page_config(page_title="Consumer Credit Risk Analytics", layout="wide", page_icon="💳")


# ----------------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------------
@st.cache_data
def load_results() -> dict:
    with open(PROJECT_ROOT / "reports" / "experiment_results.json") as f:
        return json.load(f)


@st.cache_data
def load_scored() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "data" / "processed" / "scored_test_set.csv")


@st.cache_data
def load_featured() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "data" / "processed" / "featured.csv")


results = load_results()
scored = load_scored()
featured = load_featured()

FIG_DIR = PROJECT_ROOT / "reports" / "figures"

# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.title("💳 Credit Risk Analytics")
page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Risk Overview",
        "Portfolio Analytics",
        "Risk Segmentation",
        "PD Model",
        "Class Imbalance / VAE",
        "Model Monitoring",
        "Customer Risk",
        "🤖 CreditRisk AI Copilot",
        "Methodology",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ **Portfolio / educational project** using a public dataset "
    "(UCI Default of Credit Card Clients, Taiwan, 2005). Not real Bank of "
    "America data, not a production system, and not a lending decision tool."
)

# ----------------------------------------------------------------------------
# Page 1 — Executive Risk Overview
# ----------------------------------------------------------------------------
if page == "Executive Risk Overview":
    st.title("Executive Risk Overview")

    total_customers = len(featured)
    total_exposure = featured["LIMIT_BAL"].sum()
    default_rate = featured["default_next_month"].mean() * 100
    avg_pd = scored["predicted_pd"].mean() * 100
    high_risk_pct = (scored["risk_band"].isin(["High Risk", "Very High Risk"])).mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Total Credit Exposure", f"${total_exposure/1e6:,.1f}M")
    c3.metric("Historical Default Rate", f"{default_rate:.2f}%")
    c4.metric("Avg. Predicted PD (test set)", f"{avg_pd:.2f}%")
    c5.metric("High/Very High Risk Population", f"{high_risk_pct:.1f}%")

    st.markdown("### Portfolio Risk Distribution")
    band_counts = scored["risk_band"].value_counts().reindex(
        ["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"]
    )
    fig = px.bar(
        x=band_counts.index, y=band_counts.values,
        labels={"x": "Risk Band", "y": "Customers"},
        color=band_counts.index,
        color_discrete_map={
            "Low Risk": "#2c7fb8", "Moderate Risk": "#41b6c4",
            "High Risk": "#feb24c", "Very High Risk": "#d95f0e",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"The best-performing model ({results['best_model'].replace('_', ' ').title()}) "
        f"achieves an ROC-AUC of {results['model_metrics'][results['best_model']]['roc_auc']:.3f} "
        f"on a held-out test set, with isotonic calibration applied "
        f"(Brier score improved from {results['calibration']['raw_brier']:.4f} to "
        f"{results['calibration']['isotonic_brier']:.4f})."
    )

# ----------------------------------------------------------------------------
# Page 2 — Portfolio Analytics
# ----------------------------------------------------------------------------
elif page == "Portfolio Analytics":
    st.title("Portfolio Analytics")
    tabs = st.tabs(["By Age", "By Education", "By Utilization", "By Delinquency"])

    with tabs[0]:
        st.image(str(FIG_DIR / "default_rate_by_age.png"), use_container_width=True)
    with tabs[1]:
        st.image(str(FIG_DIR / "default_rate_by_education.png"), use_container_width=True)
    with tabs[2]:
        st.image(str(FIG_DIR / "default_rate_by_utilization.png"), use_container_width=True)
    with tabs[3]:
        st.image(str(FIG_DIR / "default_rate_by_delinquency.png"), use_container_width=True)

    st.markdown("### Statistical Tests (association, not causation)")
    stats_path = PROJECT_ROOT / "reports" / "experiment_results.json"
    st.caption(
        "Chi-square tests confirm SEX, EDUCATION, and MARRIAGE are all statistically "
        "associated with default at p < 0.001, though effect sizes (Cramer's V) are "
        "small (0.03–0.07) — demographic variables are weak predictors compared to "
        "behavioral ones like delinquency history."
    )

# ----------------------------------------------------------------------------
# Page 3 — Risk Segmentation
# ----------------------------------------------------------------------------
elif page == "Risk Segmentation":
    st.title("Customer Risk Segmentation")
    st.caption(
        "Segments are defined from the PD distribution's 50th, 80th, and 95th "
        "percentiles on the training population — not arbitrary round numbers."
    )

    seg_summary = scored.groupby("risk_band", observed=True).agg(
        customers=("ID", "count"),
        avg_predicted_pd=("predicted_pd", "mean"),
        actual_default_rate=("actual_default", "mean"),
        total_exposure=("LIMIT_BAL", "sum"),
    ).reindex(["Low Risk", "Moderate Risk", "High Risk", "Very High Risk"])
    st.dataframe(
        seg_summary.style.format({
            "avg_predicted_pd": "{:.2%}", "actual_default_rate": "{:.2%}",
            "total_exposure": "${:,.0f}",
        }),
        use_container_width=True,
    )

    fig = px.pie(seg_summary.reset_index(), names="risk_band", values="total_exposure",
                 title="Exposure Concentration by Risk Band")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Page 4 — PD Model
# ----------------------------------------------------------------------------
elif page == "PD Model":
    st.title("Probability of Default Model")
    best = results["best_model"]
    m = results["model_metrics"][best]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    c2.metric("PR-AUC", f"{m['pr_auc']:.3f}")
    c3.metric("KS Statistic", f"{m['ks']:.3f}")
    c4.metric("Brier Score (calibrated)", f"{results['calibration']['isotonic_brier']:.4f}")

    st.markdown("### Model Comparison")
    comp_df = pd.DataFrame(results["model_metrics"]).T[["roc_auc", "pr_auc", "ks", "brier_score", "f1"]]
    st.dataframe(comp_df.style.format("{:.4f}"), use_container_width=True)
    st.caption(
        "Accuracy is intentionally not shown as a headline metric — with a 78/22 class "
        "split, a trivial 'always predict no default' model already scores ~78% "
        "accuracy while catching zero defaulters."
    )

    st.markdown("### ROC & Precision-Recall Curves")
    st.image(str(FIG_DIR / "roc_pr_curves.png"), use_container_width=True)

    st.markdown("### Calibration")
    st.image(str(FIG_DIR / "calibration_plot.png"), use_container_width=True)
    cal_df = pd.DataFrame(results["calibration_table"])
    st.dataframe(cal_df, use_container_width=True)

    st.markdown("### Top Risk Drivers")
    st.image(str(FIG_DIR / "feature_importance.png"), use_container_width=True)

    shap_summary_path = PROJECT_ROOT / "reports" / "shap_summary.json"
    shap_fig_path = FIG_DIR / "shap_summary.png"
    if shap_summary_path.exists():
        st.markdown("### SHAP Global Feature Importance")
        st.image(str(shap_fig_path), use_container_width=True)
        with open(shap_summary_path) as f:
            shap_summary = json.load(f)
        st.caption(
            f"Computed on a random sample of {shap_summary['sample_size']} test-set "
            f"customers using SHAP TreeExplainer on the {shap_summary['model']} model."
        )

# ----------------------------------------------------------------------------
# Page — Class Imbalance / VAE Experiment
# ----------------------------------------------------------------------------
elif page == "Class Imbalance / VAE":
    st.title("Class Imbalance & the VAE Experiment")
    vae_path = PROJECT_ROOT / "reports" / "vae_comparison_results.json"
    if not vae_path.exists():
        st.info("Run `src/run_vae_comparison.py` to generate this page's data.")
    else:
        with open(vae_path) as f:
            vae_results = json.load(f)

        st.markdown(
            f"A real VAE (PyTorch — encoder/decoder network, reparameterization "
            f"trick, KL-regularized training) was trained **exclusively on the "
            f"{vae_results['n_minority_train']:,} minority-class (default) rows "
            f"of the training fold**, then used to generate "
            f"{vae_results['n_synthetic_generated']:,} synthetic samples — enough "
            f"to fully balance the training set against its "
            f"{vae_results['n_majority_train']:,} majority-class rows."
        )

        c1, c2, c3 = st.columns(3)
        diag = vae_results["distributional_diagnostics"]
        c1.metric("Mean standardized deviation", f"{diag['mean_standardized_diff_vs_real']:.3f} std",
                   help="Synthetic vs. real minority-class feature means, in units of the real std. Lower is better.")
        c2.metric("Correlation Frobenius diff", f"{diag['correlation_matrix_frobenius_diff']:.2f}",
                   help="How well the synthetic data preserves real feature-to-feature correlations. Lower is better.")
        c3.metric("VAE training epochs", vae_results["vae_training"]["epochs"])

        st.caption(diag["note"])

        st.markdown("### Does it actually improve the model?")
        st.image(str(FIG_DIR / "vae_comparison.png"), use_container_width=True)

        comp_df = pd.DataFrame(vae_results["full_comparison_table"])
        st.dataframe(
            comp_df[comp_df["strategy"].isin(["none (baseline)", "vae_balanced"])]
            .style.format({"roc_auc": "{:.4f}", "pr_auc": "{:.4f}", "recall": "{:.4f}",
                            "precision": "{:.4f}", "f1": "{:.4f}"}),
            use_container_width=True,
        )

        st.error(
            "**Verdict: rejected.** The VAE learned realistic per-feature marginal "
            "distributions, but fully balancing the training set with its synthetic "
            "samples did not improve — and for Logistic Regression, actively "
            "worsened — classification performance versus simple class-weighting. "
            "Class-weighting remains the model's actual imbalance-handling strategy. "
            "This page reports a real, evaluated negative result rather than "
            "assuming synthetic augmentation must help."
        )

# ----------------------------------------------------------------------------
# Page 5 — Model Monitoring
# ----------------------------------------------------------------------------
elif page == "Model Monitoring":
    st.title("Model Monitoring")
    st.warning(
        "This dataset is a single historical snapshot (April–September 2005). "
        "There is no true out-of-time batch to monitor drift against. The PSI "
        "below compares the train-set and test-set PD distributions as a "
        "**methodology demonstration**, not a real drift finding."
    )
    st.metric("PSI (train PD vs. test PD)", f"{results['psi_train_vs_test_pd']:.4f}", help="< 0.10 = stable")

    st.markdown("### Retraining Triggers (documented policy)")
    from monitoring import RETRAINING_TRIGGERS
    for t in RETRAINING_TRIGGERS:
        st.markdown(f"- {t}")

# ----------------------------------------------------------------------------
# Page 6 — Customer Risk
# ----------------------------------------------------------------------------
elif page == "Customer Risk":
    st.title("Customer-Level Risk Assessment")
    st.error(
        "⚠️ This is a model-based risk estimate from a portfolio/educational project, "
        "**not a real lending decision.**"
    )

    customer_id = st.selectbox("Select a customer (test set)", scored["ID"].tolist())
    row = scored[scored["ID"] == customer_id].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted PD", f"{row['predicted_pd']:.2%}")
    c2.metric("Risk Band", row["risk_band"])
    c3.metric("Credit Limit", f"${row['LIMIT_BAL']:,.0f}")
    c4.metric("Latest Utilization", f"{row['UTIL_LATEST']:.1%}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Months Delinquent (of 6)", int(row["N_MONTHS_DELINQUENT"]))
    c6.metric("Max Delinquency Severity", int(row["MAX_DELINQUENCY"]))
    c7.metric("Actual Outcome (held-out)", "Defaulted" if row["actual_default"] == 1 else "No Default")

    shap_path = PROJECT_ROOT / "data" / "processed" / "shap_values_sample.csv"
    if shap_path.exists():
        shap_df = pd.read_csv(shap_path)
        if customer_id in shap_df["ID"].values:
            st.markdown("### Why this customer's PD is what it is (SHAP)")
            row_shap = shap_df[shap_df["ID"] == customer_id].iloc[0].drop("ID")
            top = row_shap.abs().sort_values(ascending=False).head(8)
            plot_df = pd.DataFrame({
                "feature": [c.replace("shap_", "") for c in top.index],
                "shap_value": [row_shap[c] for c in top.index],
            })
            fig = px.bar(plot_df, x="shap_value", y="feature", orientation="h",
                         color="shap_value", color_continuous_scale="RdBu_r",
                         title="Top contributing features (SHAP, +pushes PD up / -pushes PD down)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("SHAP explanation not available for this customer (outside the sampled 1,000).")

    st.session_state["selected_customer"] = row.to_dict()

# ----------------------------------------------------------------------------
# Page 7 — CreditRisk AI Copilot
# ----------------------------------------------------------------------------
elif page == "🤖 CreditRisk AI Copilot":
    st.title("🤖 CreditRisk AI Copilot")
    st.caption(
        "Explains verified analytical outputs from this project. It does not "
        "calculate risk itself and does not make lending decisions."
    )

    copilot = CreditRiskCopilot()
    if not copilot.is_available:
        st.warning(
            "AI Copilot is currently unavailable (GROQ_API_KEY not set). "
            "Core risk analytics above remain fully available without it. "
            "See `.env.example` / README → 'AI Copilot' for setup."
        )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    col_a, col_b = st.columns([3, 1])
    with col_b:
        st.markdown("**Example questions**")
        examples = [
            "Summarize the portfolio risk.",
            "What are the strongest default predictors?",
            "Is the PD model well calibrated?",
            "Why is this customer high risk?",
            "What early-warning indicators should we monitor?",
            "How stable is the model?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex}"):
                st.session_state["pending_question"] = ex

        if st.button("📋 Generate Executive Risk Summary"):
            st.session_state["pending_question"] = (
                "Generate an executive risk summary covering: portfolio health, "
                "default risk, high-risk population, major risk drivers, "
                "early-warning indicators, model performance, monitoring status, "
                "business recommendations, and limitations."
            )

        if st.button("🧹 Clear Conversation"):
            st.session_state["chat_history"] = []

    def build_context() -> dict:
        best = results["best_model"]
        ctx = {
            "portfolio": {
                "total_customers": int(len(featured)),
                "default_rate_pct": round(float(featured["default_next_month"].mean() * 100), 2),
                "total_exposure": float(featured["LIMIT_BAL"].sum()),
            },
            "model_performance": {"best_model": best, **results["model_metrics"][best],
                                   "calibration": results["calibration"]},
            "risk_segments": scored.groupby("risk_band", observed=True)["predicted_pd"].agg(
                ["count", "mean"]).to_dict(),
            "top_risk_drivers": results["explainability"],
            "monitoring": {"psi_train_vs_test_pd": results["psi_train_vs_test_pd"]},
            "fairness": results.get("fairness", {}),
            "data_quality": {
                "rows": 30000, "missing_values": 0, "duplicate_rows": 0,
                "undocumented_education_codes_pct": 1.15, "undocumented_marriage_code_pct": 0.18,
            },
        }
        if "selected_customer" in st.session_state:
            ctx["selected_customer"] = st.session_state["selected_customer"]
        return ctx

    with col_a:
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        pending = st.session_state.pop("pending_question", None)
        user_input = st.chat_input("Ask about the portfolio, the model, or a customer...")
        question = pending or user_input

        if question:
            st.session_state["chat_history"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                if not copilot.is_available:
                    st.markdown(
                        "AI Copilot is currently unavailable. Core risk analytics "
                        "remain available in the other tabs."
                    )
                else:
                    with st.spinner("Thinking..."):
                        result = copilot.chat(
                            question, build_context(),
                            history=st.session_state["chat_history"][:-1],
                        )
                    if result["success"]:
                        st.markdown(result["content"])
                        st.session_state["chat_history"].append(
                            {"role": "assistant", "content": result["content"]}
                        )
                    else:
                        st.error(f"AI Copilot request failed: {result['error']}")

# ----------------------------------------------------------------------------
# Page 8 — Methodology
# ----------------------------------------------------------------------------
elif page == "Methodology":
    st.title("Methodology")
    report_path = PROJECT_ROOT / "reports" / "final_analysis.md"
    if report_path.exists():
        st.markdown(report_path.read_text())
    else:
        st.info("Run the full pipeline to generate reports/final_analysis.md.")
