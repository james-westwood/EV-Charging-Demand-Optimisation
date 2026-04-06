from datetime import date
from pathlib import Path

import joblib
from lightgbm import LGBMRegressor


def save_artefacts(model_dict: dict[str, LGBMRegressor], date_of_model: str | None = None, base_path = Path("saved_models")) -> Path:
    """Save P10/P50/P90 models to saved_models/YYYY-MM-DD/."""
    if date_of_model is None:
        date_of_model = date.today().isoformat()

    model_root = base_path / date_of_model
    model_root.mkdir(parents=True, exist_ok=True)

    for q_name, model in model_dict.items():
        joblib.dump(model, model_root / f"{q_name}.joblib")

    return model_root


def load_latest_artefacts(base_path = Path("saved_models")) -> dict[str, LGBMRegressor]:
    """Load the most recently saved P10/P50/P90 models."""
    date_dirs = sorted(base_path.iterdir())
    if not date_dirs:
        raise FileNotFoundError(f"No saved models found in {base_path}")

    latest = date_dirs[-1]
    return {q: joblib.load(latest / f"{q}.joblib") for q in ["p10", "p50", "p90"]}
