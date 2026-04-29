"""FastAPI application for carbon intensity forecasting."""
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI

from src.api.cache import get_features_with_ttl
from src.features.store import load_features
from src.models.forecasting.artefacts import load_latest_artefacts, load_latest_artefacts_from_gcs

_MODEL_BUCKET = os.getenv("MODEL_BUCKET", "")


@asynccontextmanager
def lifespan(app: FastAPI):
    """Load models from GCS and features on startup, store in app.state, unload on shutdown."""
    if _MODEL_BUCKET:
        app.state.models = load_latest_artefacts_from_gcs(_MODEL_BUCKET)
    else:
        from src.models.forecasting.artefacts import load_latest_artefacts

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