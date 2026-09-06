"""
eda_plots.py
------------
Generates a small set of EDA/evaluation visualizations — each one tied
to a specific business question, not a generic "plot everything" sweep.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay, precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})


def plot_target_distribution(df: pd.DataFrame, target_col: str):
    counts = df[target_col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["No Default", "Default"], counts.values, color=["#2c7fb8", "#d95f0e"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 200, f"{v:,}\n({v/counts.sum()*100:.1f}%)", ha="center")
    ax.set_title("Target Class Distribution")
    ax.set_ylabel("Number of Customers")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "target_distribution.png")
    plt.close(fig)


def plot_default_rate_by_age(df: pd.DataFrame, target_col: str):
    df = df.copy()
    df["age_band"] = pd.cut(df["AGE"], bins=[20, 25, 30, 35, 40, 45, 50, 55, 60, 80])
    rate = df.groupby("age_band", observed=True)[target_col].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    rate.plot(kind="bar", ax=ax, color="#2c7fb8")
    ax.set_ylabel("Default Rate (%)")
    ax.set_title("Default Rate by Age Band")
    ax.set_xlabel("Age Band")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "default_rate_by_age.png")
    plt.close(fig)


def plot_default_rate_by_delinquency(df: pd.DataFrame, target_col: str):
    rate = df.groupby("N_MONTHS_DELINQUENT", observed=True)[target_col].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    rate.plot(kind="bar", ax=ax, color="#d95f0e")
    ax.set_ylabel("Default Rate (%)")
    ax.set_xlabel("Number of Months Delinquent (of last 6)")
    ax.set_title("Default Rate vs. Delinquency History")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "default_rate_by_delinquency.png")
    plt.close(fig)


def plot_default_rate_by_utilization(df: pd.DataFrame, target_col: str):
    df = df.copy()
    df["util_band"] = pd.cut(df["UTIL_AVG"], bins=[-2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 3])
    rate = df.groupby("util_band", observed=True)[target_col].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    rate.plot(kind="bar", ax=ax, color="#31a354")
    ax.set_ylabel("Default Rate (%)")
    ax.set_xlabel("Average Utilization Band")
    ax.set_title("Default Rate vs. Average Credit Utilization")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "default_rate_by_utilization.png")
    plt.close(fig)


def plot_default_rate_by_education(df: pd.DataFrame, target_col: str):
    labels = {1: "Graduate", 2: "University", 3: "High School", 4: "Other/Unknown"}
    df = df.copy()
    df["edu_label"] = df["EDUCATION"].map(labels)
    rate = df.groupby("edu_label", observed=True)[target_col].mean() * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    rate.plot(kind="bar", ax=ax, color="#756bb1")
    ax.set_ylabel("Default Rate (%)")
    ax.set_title("Default Rate by Education Level")
    plt.xticks(rotation=30)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "default_rate_by_education.png")
    plt.close(fig)


def plot_roc_and_pr(y_true, y_proba, model_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=axes[0])
    axes[0].set_title(f"ROC Curve — {model_name}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    axes[1].plot(recall, precision)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve — {model_name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_pr_curves.png")
    plt.close(fig)


def plot_calibration(cal_table_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 5))
    x = cal_table_df["avg_predicted_pd"]
    y = cal_table_df["actual_default_rate"]
    ax.plot([0, x.max() * 1.1], [0, x.max() * 1.1], "k--", label="Perfect calibration")
    ax.scatter(x, y, s=cal_table_df["customers"] / 10, color="#d95f0e", label="Risk bands")
    ax.set_xlabel("Average Predicted PD")
    ax.set_ylabel("Actual Default Rate")
    ax.set_title("Calibration: Predicted vs. Actual Default Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration_plot.png")
    plt.close(fig)


def plot_feature_importance(explain: dict, model_name: str):
    key = f"{model_name}_top_features"
    if key not in explain:
        return
    data = explain[key][:12]
    features = [d["feature"] for d in data][::-1]
    values = [d["importance"] for d in data][::-1]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(features, values, color="#2c7fb8")
    ax.set_title(f"Top Feature Importances — {model_name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png")
    plt.close(fig)


if __name__ == "__main__":
    from data_ingestion import load_raw_data
    from feature_engineering import engineer_features
    from preprocessing import TARGET_CLEAN, clean_dataset

    df = engineer_features(clean_dataset(load_raw_data()))
    plot_target_distribution(df, TARGET_CLEAN)
    plot_default_rate_by_age(df, TARGET_CLEAN)
    plot_default_rate_by_delinquency(df, TARGET_CLEAN)
    plot_default_rate_by_utilization(df, TARGET_CLEAN)
    plot_default_rate_by_education(df, TARGET_CLEAN)

    with open(PROJECT_ROOT / "reports" / "experiment_results.json") as f:
        results = json.load(f)

    scored = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "scored_test_set.csv")
    plot_roc_and_pr(scored["actual_default"], scored["predicted_pd"], results["best_model"])

    cal_df = pd.DataFrame(results["calibration_table"])
    plot_calibration(cal_df)
    plot_feature_importance(results["explainability"], results["best_model"])
    print("Saved figures to", FIG_DIR)
