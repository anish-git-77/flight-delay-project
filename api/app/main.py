"""
Flight Delay Risk API.

Serves the production gradient-boosting model trained in
src/build_production_model.py. See README.md in the project root for the
full methodology and the honest performance numbers (ROC-AUC 0.652,
PR-AUC 0.316 on a temporal holdout).

Run locally:
    uvicorn app.main:app --reload --port 8000
Then open http://localhost:8000/docs for interactive API docs.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.features import load_artifacts, predict
from app.schemas import HealthResponse, PredictRequest, PredictResponse

app = FastAPI(
    title="Flight Delay Risk API",
    description=(
        "Estimates the probability that a flight's departure will be delayed "
        "more than 15 minutes, based on schedule, route, carrier, and weather "
        "history. Trained on NYC-area departures (EWR/JFK/LGA), 2013."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    # Fail fast and loudly if artifacts are missing, rather than on first request.
    load_artifacts()


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "Flight Delay Risk API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    try:
        _, _, meta, _ = load_artifacts()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            trained_on=meta["trained_on"],
            holdout_roc_auc=meta["holdout_roc_auc"],
            holdout_pr_auc=meta["holdout_pr_auc"],
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")


@app.get("/metadata", tags=["meta"])
def metadata():
    _, _, meta, _ = load_artifacts()
    return meta


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict_delay(req: PredictRequest):
    """
    Predict departure-delay risk for a single flight.

    Only carrier, origin, dest, and scheduled_departure are required.
    Everything else (distance, weather) is optional - if omitted, the API
    estimates it from historical route and climatology data, and tells you
    which values were estimated in `feature_sources`.
    """
    try:
        return predict(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not score this flight: {e}")
