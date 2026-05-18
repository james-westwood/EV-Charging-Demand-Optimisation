from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import google.cloud.storage
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
    """Load the most recently saved P10/P50/P90 models from local disk."""
    date_dirs = sorted(base_path.iterdir())
    if not date_dirs:
        raise FileNotFoundError(f"No saved models found in {base_path}")

    latest = date_dirs[-1]
    return {q: joblib.load(latest / f"{q}.joblib") for q in ["p10", "p50", "p90"]}


def load_latest_artefacts_from_gcs(bucket_name: str, prefix: str = "saved_models/") -> dict[str, LGBMRegressor]:
    """Load the most recent models from Google Cloud Storage.

    Downloads to temp directory and loads with joblib.
    Models are expected at gs://{bucket}/{prefix}{YYYY-MM-DD}/p{10,50,90}.joblib

    Args:
        bucket_name: GCS bucket name (without gs:// prefix)
        prefix: path prefix within bucket

    Returns:
        dict of {"p10": model, "p50": model, "p90": model}
    """
    client = google.cloud.storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        raise FileNotFoundError(f"No models found in gs://{bucket_name}/{prefix}")

    date_dirs = sorted(set(Path(b.name).parent.name for b in blobs if b.name.endswith(".joblib")))
    if not date_dirs:
        raise FileNotFoundError(f"No model directories in gs://{bucket_name}/{prefix}")

    latest_date = date_dirs[-1]
    model_keys = [f"{prefix}{latest_date}/{q}.joblib" for q in ["p10", "p50", "p90"]]

    with TemporaryDirectory() as tmpdir:
        models = {}
        for key in model_keys:
            blob = bucket.blob(key)
            if not blob.exists():
                raise FileNotFoundError(f"Model not found: gs://{bucket_name}/{key}")

            local_path = Path(tmpdir) / Path(key).name
            blob.download_to_filename(local_path)
            q_name = Path(key).stem
            models[q_name] = joblib.load(local_path)

    return models
