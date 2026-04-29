"""FastAPI application for carbon intensity forecasting."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI

import os
from src.features.store import load_features
from src.models.forecasting.artefacts import load_latest_artefacts

_FEATURES_CACHE_TTL_MINUTES = int(os.getenv("FEATURES_CACHE_TTL_MINUTES", "30"))


def get_features_with_ttl(app: FastAPI) -> object:
    """Get features from app.state, reloading if cache TTL expired."""
    features_loaded_at = getattr(app.state, "features_loaded_at", None)
    if features_loaded_at is None:
        app.state.features = load_features()
        app.state.features_loaded_at = datetime.now(timezone.utc)
        return app.state.features

    age = datetime.now(timezone.utc) - features_loaded_at
    if age > timedelta(minutes=_FEATURES_CACHE_TTL_MINUTES):
        app.state.features = load_features()
        app.state.features_loaded_at = datetime.now(timezone.utc)

    return app.state.features


@asynccontextmanager
def lifespan(app: FastAPI):
    """Load models and features on startup, store in app.state, unload on shutdown."""
    app.state.models = load_latest_artefacts()
    app.state.features = load_features()
    app.state.features_loaded_at = datetime.now(timezone.utc)
    yield
    del app.state.models
    del app.state.features


app = FastAPI(
    title="Carbon Intensity Forecast API",
    description="EV Charging Demand Optimisation — Forecast endpoint",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Health check."""
    return {"status": "healthy"}


@app.get("/metrics")
def metrics() -> str:
    """Prometheus metrics stub."""
    return "# HELP forecast_requests_total Total forecast requests\n# TYPE forecast_requests_total counter\nforecast_requests_total 0"


from src.api.routes import forecast

app.include_router(forecast.router)