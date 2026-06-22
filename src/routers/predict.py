from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import mlflow.pyfunc
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

_model: mlflow.pyfunc.PyFuncModel | None = None

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "MLP_PyTorch")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "challenger")


def _load_model() -> mlflow.pyfunc.PyFuncModel:
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")


@asynccontextmanager
async def lifespan(app: Any):  # noqa: ANN001
    global _model
    _model = _load_model()
    yield


router = APIRouter(prefix="/predict", tags=["predict"])


class CustomerFeatures(BaseModel):
    """Campos do dataset Telco Customer Churn após remoção do customerID."""

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):
    churn: int
    probability: float


@router.post("/", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado")

    import pandas as pd

    data = pd.DataFrame([customer.model_dump()])
    raw = _model.predict(data)

    # suporta modelos que retornam array de probabilidades ou logits
    prob = float(raw[0]) if hasattr(raw, "__len__") else float(raw)
    if prob > 1.0 or prob < 0.0:
        import torch

        prob = float(torch.sigmoid(torch.tensor(prob)))

    return PredictionResponse(churn=int(prob >= 0.5), probability=round(prob, 4))
