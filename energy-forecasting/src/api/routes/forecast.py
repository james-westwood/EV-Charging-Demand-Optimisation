"""Forecast routes for the FastAPI application."""
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/forecast", tags=["forecast"])


class ForecastRequest(BaseModel):
    """Request body for /forecast endpoint."""
    horizon: int = Field(ge=1, le=336, description="Number of 30-min periods (max 7 days)")


class ForecastResponse(BaseModel):
    """Response body for /forecast endpoint."""
    forecasts: list[dict]


def _generate_forecasts(
    horizon: int,
    app_state,
) -> list[dict]:
    """Shared forecast generation logic."""
    from fastapi import HTTPException
    from src.features.inference import generate_inference_features
    from src.api.cache import get_features_with_ttl

    if app_state.models is None or app_state.features is None:
        raise HTTPException(
            status_code=503,
            detail="Model or feature data not loaded. Service temporarily unavailable.",
        )

    start = datetime.now(timezone.utc)

    features = generate_inference_features(
        horizon,
        cached_df=get_features_with_ttl(app_state),
        start=start,
    )

    models = app_state.models
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

    return forecasts


@router.post("", response_model=ForecastResponse)
def forecast_post(
    request: ForecastRequest,
    http_request: Request,
) -> ForecastResponse:
    """Generate carbon intensity forecasts for N periods ahead (POST)."""
    forecasts = _generate_forecasts(request.horizon, http_request.app.state)
    return ForecastResponse(forecasts=forecasts)


@router.get("")
def forecast_get(
    http_request: Request,
    horizon: int = Query(default=48, ge=1, le=336, description="Number of 30-min periods to forecast"),
) -> dict:
    """Generate carbon intensity forecasts for N periods ahead (GET).

    Returns the public API format consumed by the frontend widget:
    { generated_at, horizon_hours, settlement_period_minutes, model_version, points: [{t, p10, p50, p90}] }
    """
    forecasts = _generate_forecasts(horizon, http_request.app.state)

    points = [
        {
            "t": f["timestamp"],
            "p10": round(f["p10"], 1),
            "p50": round(f["p50"], 1),
            "p90": round(f["p90"], 1),
        }
        for f in forecasts
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "horizon_hours": horizon * 0.5,
        "settlement_period_minutes": 30,
        "model_version": "gb-ci-2026.05.15",
        "points": points,
    }