"""
schemas.py
----------
Pydantic models for the Consumer Credit Risk API. Field names match the
raw dataset schema exactly so a caller can pass through source-system
fields without renaming.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, conint


class CustomerRecord(BaseModel):
    ID: int = Field(..., description="Customer/account identifier (not used as a model feature)")
    LIMIT_BAL: float = Field(..., gt=0, description="Credit limit (NT dollar)")
    SEX: conint(ge=1, le=2) = Field(..., description="1 = male, 2 = female")
    EDUCATION: conint(ge=0, le=6) = Field(..., description="1=grad school, 2=university, 3=high school, 4=other (0/5/6 undocumented, folded into 'other')")
    MARRIAGE: conint(ge=0, le=3) = Field(..., description="1=married, 2=single, 3=other (0 undocumented, folded into 'other')")
    AGE: conint(ge=18, le=100)
    PAY_0: int = Field(..., description="Repayment status, most recent month")
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(..., ge=0)
    PAY_AMT2: float = Field(..., ge=0)
    PAY_AMT3: float = Field(..., ge=0)
    PAY_AMT4: float = Field(..., ge=0)
    PAY_AMT5: float = Field(..., ge=0)
    PAY_AMT6: float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "ID": 1, "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
                "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
                "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0,
                "BILL_AMT5": 0, "BILL_AMT6": 0,
                "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0,
            }
        }


class BatchRequest(BaseModel):
    customers: list[CustomerRecord]


class PredictionResponse(BaseModel):
    ID: int
    predicted_pd: float = Field(..., description="Calibrated probability of default (0-1)")
    risk_band: str
    predicted_class: int = Field(..., description="1 = predicted default at 0.5 threshold, 0 = otherwise")
    disclaimer: str = (
        "Model-based estimate from a portfolio/educational project. Not a real "
        "lending decision."
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    n_customers: int


class ModelInfo(BaseModel):
    model_name: str
    calibration_method: str
    roc_auc: float
    pr_auc: float
    ks_statistic: float
    brier_score: float
    segmentation_thresholds: dict
    imbalance_strategy: str
    trained_on_rows: int
    notes: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
