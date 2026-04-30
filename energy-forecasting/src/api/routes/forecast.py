"""Forecast routes for the FastAPI application."""
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/forecast", tags=["forecast"])


class ForecastRequest(BaseModel):
    """Request body for /forecast endpoint."""
    horizon: int = Field(ge=1, le=336, description="Number of 30-min periods (max 7 days)")


class ForecastResponse(BaseModel):
    """Response body for /forecast endpoint."""
    forecasts: list[dict]


@router.post("", response_model=ForecastResponse)
def forecast(
    request: ForecastRequest,
    http_request: Request,
) -> ForecastResponse:
    """Generate carbon intensity forecasts for N periods ahead.

    Returns {timestamp, p10, p50, p90} for each settlement period.
    """
    from src.features.inference import generate_inference_features
    from src.api.cache import get_features_with_ttl

    horizon = request.horizon

    start = datetime.now(timezone.utc)

    features = generate_inference_features(horizon, cached_df=get_features_with_ttl(http_request.app.state), start=start)

    models = http_request.app.state.models
    p10_model = models["p10"]
    p50_model = models["p50"]
    p90_model = models["p90"]

    p10_preds = p10_model.predict(features)
    p50_preds = p50_model.predict(features)
    p90_preds = p90_model.predict(features)

    forecasts = []
    for i in range(horizon):
        ts = (start + i * timedelta(minutes=30)).isoformat()
        forecasts.append({
            "timestamp": ts,
            "p10": float(p10_preds[i]),
            "p50": float(p50_preds[i]),
            "p90": float(p90_preds[i]),
        })

    return ForecastResponse(forecasts=forecasts)