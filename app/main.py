from fastapi import FastAPI, HTTPException
from app.model_service import ModelService
from app.schemas import CustomerInput
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Customer Churn ML API",
    description="REST API serving the trained customer churn Random Forest model.",
    version="1.0.0",
)

service = ModelService()
service.load()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

@app.get("/")
def home():
    return {
        "message": "Welcome to Customer Churn ML API",
        "status": "running",
        "model": "Random Forest Classifier",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health",
        "metadata": "/metadata",
        "prediction_endpoint": "/predict"
    }

@app.get("/health")
def health():
    return {
        "status": "ok" if service.ready else "degraded",
        "model_loaded": service.ready,
        "service": "customer-churn-ml-api",
    }


@app.get("/metadata")
def metadata():
    if not service.ready:
        raise HTTPException(status_code=503, detail=service.load_error)

    features = list(service.model.feature_names_in_) if hasattr(
        service.model, "feature_names_in_"
    ) else []

    return {
        "model_type": type(service.model).__name__,
        "feature_count": len(features),
        "features": features,
        "encoder_count": len(service.encoders),
    }


@app.post("/predict")
def predict(customer: CustomerInput):
    if not service.ready:
        raise HTTPException(
            status_code=503,
            detail=service.load_error or "Model artifacts are unavailable.",
        )

    try:
        return service.predict(customer.model_dump(by_alias=False))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
