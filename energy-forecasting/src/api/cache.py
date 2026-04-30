"""Feature cache with TTL for FastAPI."""
from datetime import datetime, timezone, timedelta

import os

from src.features.store import load_features, load_features_from_gcs

_FEATURES_CACHE_TTL_MINUTES = int(os.getenv("FEATURES_CACHE_TTL_MINUTES", "30"))
_FEATURES_BUCKET = os.getenv("FEATURES_BUCKET", "")


def get_features_with_ttl(app_state) -> object:
    """Get features from app.state, reloading if cache TTL expired."""
    features_loaded_at = getattr(app_state, "features_loaded_at", None)
    if features_loaded_at is None:
        if _FEATURES_BUCKET:
            app_state.features = load_features_from_gcs(_FEATURES_BUCKET)
        else:
            app_state.features = load_features()
        app_state.features_loaded_at = datetime.now(timezone.utc)
        return app_state.features

    age = datetime.now(timezone.utc) - features_loaded_at
    if age > timedelta(minutes=_FEATURES_CACHE_TTL_MINUTES):
        if _FEATURES_BUCKET:
            app_state.features = load_features_from_gcs(_FEATURES_BUCKET)
        else:
            app_state.features = load_features()
        app_state.features_loaded_at = datetime.now(timezone.utc)

    return app_state.features