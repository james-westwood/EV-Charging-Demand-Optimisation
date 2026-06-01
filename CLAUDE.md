# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working directory

All commands run from `energy-forecasting/`. The repo root contains notebooks, docs, and `prd.json` — the ML pipeline lives entirely under `energy-forecasting/`.

## Commands

```bash
# Install (uses uv)
cd energy-forecasting && uv sync --group dev

# Run all tests
uv run pytest tests/

# Run a single test file
uv run pytest tests/features/test_lags.py -v

# Run a single test by name
uv run pytest tests/features/test_lags.py::test_add_lag_features -v

# Lint
uv run ruff check src

# Type check
uv run ty check src

# Start the API locally
uv run uvicorn src.api.main:app --reload --port 8080
```

## Architecture

The pipeline has four sequential stages, each with a run script:

```
src/data/run_data_collection_pipeline.py     → DuckDB (data/ev_charging.duckdb)
src/features/run_feature_pipeline.py         → Parquet (data/features/features_YYYY-MM-DD.parquet)
src/models/forecasting/run_training_pipeline.py  → joblib (saved_models/YYYY-MM-DD/{p10,p50,p90}.joblib)
src/api/main.py                              → FastAPI serving /forecast and /optimise
```

### Data flow

Raw API data (carbon intensity, generation mix, Open-Meteo weather) lands in DuckDB via `src/data/collectors/`. The feature pipeline joins and engineers these into 30-min settlement period rows via `src/features/`. Three LightGBM quantile models (P10/P50/P90) are trained on this feature store and saved as joblib artefacts. At inference time, `src/features/inference.py` reconstructs features without DuckDB using only the last 7 days cached from the feature store.

### API startup

`src/api/main.py` uses a FastAPI lifespan context to load models and features into `app.state` on startup. Two env vars control whether these load from local disk or GCS:
- `MODEL_BUCKET` — if set, loads models from GCS instead of `saved_models/`
- `FEATURES_BUCKET` — if set, loads features from GCS instead of `data/features/`

Feature cache TTL is controlled by `FEATURES_CACHE_TTL_MINUTES` (default 30). The cache logic lives in `src/api/cache.py` and is called from routes via `get_features_with_ttl(app_state)`.

### Optimiser

`src/optimiser/` contains the LP charging schedule optimizer. The key orchestration function is `optimise_session()` in `orchestrator.py`, which accepts a carbon forecast array and session parameters and returns an optimal schedule plus a dumb-charging baseline for comparison. The API endpoint (`src/api/routes/optimise.py`) internally calls `_generate_forecasts()` from the forecast route to get the P50 carbon forecast for the departure window, then passes it to the optimiser.

### Local vs cloud

The same ML logic runs locally (pandas + DuckDB) and on Databricks (PySpark + Delta tables). The mapping is 1:1: collectors → Bronze, feature transforms → Silver, feature store → Gold, `artefacts.py` → MLflow Model Registry.

## PRD and status

`prd.json` at the repo root is the source of truth for task status. Tasks are grouped by epic (EPIC 0–8). Mark tasks `"completed": true` when done.

`STATUS.md` at the repo root is read by the MasterVault librarian to generate cross-project overviews. Update it after significant work sessions: status, progress %, and next actions.

## Deployment

The API is containerised via `energy-forecasting/Dockerfile` and deployed to Google Cloud Run. CI runs on PRs to `main`: ruff → ty → pytest.
