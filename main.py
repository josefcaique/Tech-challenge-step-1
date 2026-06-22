from fastapi import FastAPI

from src.routers import predict

app = FastAPI(
    title="Telco Churn API",
    description="API de predição de churn — Tech Challenge Etapa 2",
    version="0.1.0",
    lifespan=predict.lifespan,
)

app.include_router(predict.router)


@app.get("/health", tags=["status"])
def health() -> dict:
    return {"status": "ok"}
