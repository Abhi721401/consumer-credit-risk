"""
main.py
-------
FastAPI service exposing the trained, calibrated Consumer Credit Risk
PD model. This is the single production-facing entry point for scoring
new customer records — it calls the exact same `src/scoring.py` pipeline
used by the batch scoring job and the Streamlit dashboard, so there is
one source of truth for "how a raw record becomes a PD," not three.

Run locally:
    uvicorn api.main:app --reload --port 8000

Run via Docker:
    docker build -f Dockerfile.api -t credit-risk-api .
    docker run -p 8000:8000 credit-risk-api

Docs: http://localhost:8000/docs (Swagger UI, auto-generated)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from scoring import score_customers  # noqa: E402

from api.schemas import (  # noqa: E402
    BatchPredictionResponse,
    BatchRequest,
    CustomerRecord,
    HealthResponse,
    ModelInfo,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Consumer Credit Risk — PD Scoring API",
    description=(
        "Serves the trained, isotonic-calibrated XGBoost Probability-of-Default "
        "model from the Consumer Credit Risk Analytics portfolio project. "
        "**Portfolio/educational project — not a real lending system.**"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relax for local/demo use; restrict in a real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: dict = {"results": None, "thresholds": None, "model_ready": False}


@app.on_event("startup")
def load_artifacts():
    """Load model metadata once at startup rather than per-request."""
    try:
        with open(PROJECT_ROOT / "reports" / "experiment_results.json") as f:
            results = json.load(f)
        _state["results"] = results
        _state["thresholds"] = results["segmentation_thresholds"]
        # Confirms the model artifact itself is present and loadable, without
        # holding it in a global (scoring.py loads it per call — see note below).
        from scoring import load_scoring_model

        load_scoring_model()
        _state["model_ready"] = True
        logger.info("Model artifacts loaded successfully. Best model: %s", results["best_model"])
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to load model artifacts: %s", e)
        _state["model_ready"] = False


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "Consumer Credit Risk PD Scoring API",
        "docs": "/docs",
        "health": "/health",
        "disclaimer": "Portfolio/educational project. Not a real lending decision system.",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(
        status="ok" if _state["model_ready"] else "degraded",
        model_loaded=_state["model_ready"],
        model_name=_state["results"]["best_model"] if _state["results"] else None,
    )


@app.get("/model/info", response_model=ModelInfo, tags=["meta"])
def model_info():
    if not _state["model_ready"]:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    r = _state["results"]
    m = r["model_metrics"][r["best_model"]]
    return ModelInfo(
        model_name=r["best_model"],
        calibration_method=r["calibration"]["chosen"],
        roc_auc=m["roc_auc"],
        pr_auc=m["pr_auc"],
        ks_statistic=m["ks"],
        brier_score=r["calibration"]["isotonic_brier"] if r["calibration"]["chosen"] == "isotonic" else m["brier_score"],
        segmentation_thresholds=r["segmentation_thresholds"],
        imbalance_strategy="class_weight (SMOTE and a real VAE were evaluated and rejected — see model_validation_report.md)",
        trained_on_rows=r["split"]["n_train"],
        notes="Portfolio/educational project using the UCI/Kaggle Default of Credit Card Clients dataset.",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(customer: CustomerRecord):
    if not _state["model_ready"]:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    start = time.time()
    try:
        raw_df = pd.DataFrame([customer.model_dump()])
        scored = score_customers(raw_df, _state["thresholds"])
        row = scored.iloc[0]
    except Exception as e:  # noqa: BLE001
        logger.error("Scoring failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Scoring failed: {e}") from e

    logger.info("Scored customer %s in %.3fs -> PD=%.4f", customer.ID, time.time() - start, row["predicted_pd"])
    return PredictionResponse(
        ID=int(row["ID"]),
        predicted_pd=float(row["predicted_pd"]),
        risk_band=str(row["risk_band"]),
        predicted_class=int(row["predicted_class"]),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["scoring"])
def predict_batch(batch: BatchRequest):
    if not _state["model_ready"]:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    if not batch.customers:
        raise HTTPException(status_code=400, detail="No customers provided.")

    start = time.time()
    try:
        raw_df = pd.DataFrame([c.model_dump() for c in batch.customers])
        scored = score_customers(raw_df, _state["thresholds"])
    except Exception as e:  # noqa: BLE001
        logger.error("Batch scoring failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Batch scoring failed: {e}") from e

    predictions = [
        PredictionResponse(
            ID=int(row["ID"]),
            predicted_pd=float(row["predicted_pd"]),
            risk_band=str(row["risk_band"]),
            predicted_class=int(row["predicted_class"]),
        )
        for _, row in scored.iterrows()
    ]
    logger.info("Scored batch of %d customers in %.3fs", len(predictions), time.time() - start)
    return BatchPredictionResponse(predictions=predictions, n_customers=len(predictions))
