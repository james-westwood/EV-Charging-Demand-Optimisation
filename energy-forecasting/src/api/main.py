"""FastAPI application for carbon intensity forecasting."""
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.cache import get_features_with_ttl
from src.features.store import load_features, load_features_from_gcs
from src.models.forecasting.artefacts import load_latest_artefacts, load_latest_artefacts_from_gcs

_MODEL_BUCKET = os.getenv("MODEL_BUCKET", "")
_FEATURES_BUCKET = os.getenv("FEATURES_BUCKET", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and features from GCS (if configured) on startup, store in app.state, unload on shutdown."""
    import logging

    logger = logging.getLogger(__name__)

    # Load models
    try:
        if _MODEL_BUCKET:
            app.state.models = load_latest_artefacts_from_gcs(_MODEL_BUCKET)
        else:
            app.state.models = load_latest_artefacts()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        app.state.models = None

    # Load features
    try:
        if _FEATURES_BUCKET:
            app.state.features = load_features_from_gcs(_FEATURES_BUCKET)
        else:
            app.state.features = load_features()
        logger.info("Features loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load features: {e}")
        app.state.features = None

    app.state.features_loaded_at = datetime.now(timezone.utc)
    yield
    if hasattr(app.state, "models"):
        del app.state.models
    if hasattr(app.state, "features"):
        del app.state.features


app = FastAPI(
    title="Carbon Intensity Forecast API",
    description="EV Charging Demand Optimisation — Forecast endpoint",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: allow the Netlify frontend and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://profound-parfait-d0e282.netlify.app",
        "https://james-westwood.dev",
        "http://localhost:4321",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
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