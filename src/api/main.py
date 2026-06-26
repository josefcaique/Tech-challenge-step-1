from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from mangum import Mangum

from src.api.model import model_service
from src.api.schemas import ChurnPredictionRequest, ChurnPredictionResponse, ModelInfoResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load()
    yield


app = FastAPI(
    title="Churn Prediction API",
    version="0.1.0",
    description="API para predição de churn usando o pipeline registrado no MLflow.",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Churn Prediction API"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        status="loaded" if model_service.is_loaded else "not_loaded",
        model_uri=model_service.model_uri,
        tracking_uri=model_service.tracking_uri,
        threshold=model_service.threshold,
    )


@app.post("/predict", response_model=ChurnPredictionResponse)
def predict(payload: ChurnPredictionRequest) -> ChurnPredictionResponse:
    try:
        prediction, probability = model_service.predict(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return ChurnPredictionResponse(
        prediction=prediction,
        churn_probability=probability,
        threshold=model_service.threshold,
        model_uri=model_service.model_uri,
    )


handler = Mangum(app)
