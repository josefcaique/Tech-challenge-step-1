from __future__ import annotations

import os
from threading import Lock
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd

from src.api.schemas import ChurnPredictionRequest


class ChurnModelService:
    def __init__(self) -> None:
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        self.model_uri = os.getenv("MODEL_URI", "models:/churn_prediction_pipeline/latest")
        self.threshold = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
        self._model: Any | None = None
        self._lock = Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return

            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            self._model = mlflow.sklearn.load_model(self.model_uri)

    def predict(self, payload: ChurnPredictionRequest) -> tuple[int, float | None]:
        if self._model is None:
            self.load()

        data = pd.DataFrame([payload.model_dump()])

        probability = self._predict_probability(data)
        if probability is None:
            prediction = int(self._model.predict(data)[0])
        else:
            prediction = int(probability >= self.threshold)

        return prediction, probability

    def _predict_probability(self, data: pd.DataFrame) -> float | None:
        if self._model is None or not hasattr(self._model, "predict_proba"):
            return None

        probabilities = self._model.predict_proba(data)
        return float(probabilities[0][1])


model_service = ChurnModelService()
