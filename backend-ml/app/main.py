"""
SkyGuardAI backend-ml — FastAPI service.

Endpoints:
  GET  /health                 liveness check
  GET  /predict/live           fetch Open-Meteo -> run model + SHAP -> explain
  POST /predict/manual         same pipeline, but on a hand-entered observation
                                (fill this in via Swagger UI at /docs)

Run:
  python train_model.py        # once, to build artifacts/ from dublin.csv
  uvicorn app.main:app --reload
  open http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import STATION_LAT, STATION_LON
from app.explain import build_explanation
from app.model_service import ModelNotTrainedError, get_model_service
from app.schemas import AnomalyResponse, WeatherInput
from app.weather_client import fetch_live_weather

logger = logging.getLogger("skyguardai")

app = FastAPI(
    title="SkyGuardAI Anomaly Detection API",
    description="Fetches live weather (Open-Meteo), scores it with an "
                "IsolationForest anomaly detector, and explains the result "
                "with SHAP in plain language.",
    version="1.0.0",
)

# CORS_ALLOWED_ORIGINS="https://your-frontend.com,https://staging.your-frontend.com"
# Defaults to "*" for local dev — set the env var before deploying publicly.
_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "*")
_allow_origins = ["*"] if _origins_env == "*" else [o.strip() for o in _origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_model():
    try:
        get_model_service()
        logger.info("Model artifacts loaded successfully.")
    except ModelNotTrainedError as e:
        # Don't crash the whole app — surface a clear error on the
        # prediction endpoints instead, so /docs still loads.
        logger.error(str(e))


@app.get("/health")
def health():
    try:
        get_model_service()
        return {"status": "ok", "model_loaded": True}
    except ModelNotTrainedError as e:
        return {"status": "degraded", "model_loaded": False, "detail": str(e)}


def _respond(values: dict, observed_at, estimated_fields: list[str], source: str) -> AnomalyResponse:
    try:
        service = get_model_service()
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    result = service.explain(values, observed_at)
    text = build_explanation(result, service.feature_stats)

    return AnomalyResponse(
        is_anomaly=result["is_anomaly"],
        anomaly_score=result["anomaly_score"],
        severity_level=text["severity_level"],
        severity_text=text["severity_text"],
        what=text["what"],
        root_cause=text["root_cause"],
        top_contributions=result["top_contributions"],
        observed_at=observed_at,
        raw_values=values,
        estimated_fields=estimated_fields,
        source=source,
    )


@app.get("/predict/live", response_model=AnomalyResponse, tags=["prediction"])
async def predict_live(
    lat: float = Query(STATION_LAT, description="Latitude (defaults to Dublin Airport, matching training data)"),
    lon: float = Query(STATION_LON, description="Longitude (defaults to Dublin Airport, matching training data)"),
):
    """
    Fetches the current hour's weather from Open-Meteo for the given
    coordinates, runs it through the anomaly model, and returns a
    SHAP-explained result. This is the endpoint your frontend should call
    to trigger the full live flow.
    """
    try:
        weather = await fetch_live_weather(lat, lon)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch Open-Meteo data: {e}")

    return _respond(
        weather["values"], weather["observed_at"],
        weather["estimated_fields"], weather["source"],
    )


@app.post("/predict/manual", response_model=AnomalyResponse, tags=["prediction"])
def predict_manual(payload: WeatherInput):
    """
    Run the anomaly model on a hand-entered observation. Use this from the
    Swagger UI (/docs) to test specific scenarios without needing live data
    — e.g. plug in extreme values and confirm the model flags them.
    """
    import datetime as dt
    observed_at = payload.observed_at or dt.datetime.utcnow()
    values = payload.model_dump(exclude={"observed_at"})
    return _respond(values, observed_at, estimated_fields=[], source="manual-input")
