# AGENTS.md — EV Charging Demand Optimisation

This file is read by the AI agent at the start of each Ralph loop iteration to orient itself without re-reading all source files. Keep it up to date.

---

## Project Overview

Local ML pipeline for EV charging demand optimisation. Pulls historical grid data from free public APIs, engineers features, trains LightGBM quantile models (P10/P50/P90), models EV session behaviour with a GMM, and uses Linear Programming to schedule charging to minimise carbon emissions. Served via FastAPI.

**Target**: Carbon intensity (gCO2/kWh) and wind generation (MW) as separate LightGBM model sets.

---

## Directory Structure

```
energy-forecasting/          ← project root (cd here first)
├── src/
│   ├── data/
│   │   ├── collectors/      ← API clients + storage
│   │   └── validators/      ← schema validation + exceptions
│   ├── features/            ← feature engineering pipeline
│   ├── models/
│   │   ├── forecasting/     ← LightGBM training + inference
│   │   └── ev_behaviour/    ← GMM session model
│   ├── optimiser/           ← LP charging scheduler
│   └── api/
│       └── routes/          ← FastAPI routers
├── tests/
│   ├── fixtures/            ← sample Parquet + JSON fixtures
│   ├── data/
│   ├── features/
│   ├── models/
│   └── optimiser/
├── data/
│   ├── raw/
│   │   ├── carbon_intensity/
│   │   ├── generation_mix/
│   │   ├── weather/
│   │   └── ev_sessions/
│   └── features/
├── saved_models/            ← trained artefacts versioned by YYYY-MM-DD
├── prd.json                 ← task list for Ralph loop
├── progress.txt             ← updated each iteration
├── AGENTS.md                ← this file
├── pyproject.toml
└── pytest.ini
```

---

## Key Conventions

### Python
- **Package manager**: `uv` (not pip). Run `uv run pytest` or `uv run python`.
- **Python version**: 3.11+
- **All src/ subdirectories**: must have `__init__.py`
- **Imports**: absolute from `src.*` (e.g. `from src.data.collectors.carbon_intensity import fetch_carbon_intensity`)
- **Logging**: use `get_logger(__name__)` from `src.logging_config`, never `print()`
- **No type: ignore** — fix type issues properly

### Data
- **Settlement periods**: datetime, 30-min intervals, UTC
- **Parquet schema**: always use pyarrow engine
- **Raw data path**: `data/raw/{source}/{YYYY-MM-DD}.parquet`
- **Feature path**: `data/features/features_{YYYY-MM-DD}.parquet`
- **Model path**: `saved_models/{YYYY-MM-DD}/` (p10_model.txt, p50_model.txt, p90_model.txt, metadata.json)
- **GMM path**: `saved_models/gmm_{YYYY-MM-DD}.pkl`

### Testing
- **Framework**: pytest
- **Run command**: `cd energy-forecasting && uv run pytest tests/ -v`
- **Fixtures**: loaded from `tests/conftest.py` which reads `tests/fixtures/*.parquet`
- **HTTP mocking**: use `unittest.mock.patch` or `pytest-httpx` — do NOT make real HTTP calls in tests
- **No real file I/O in unit tests**: use `tmp_path` pytest fixture for any file writes
- **One test file per source module**: `src/data/collectors/carbon_intensity.py` → `tests/data/test_carbon_intensity.py`

### LightGBM
- **Objective**: `quantile` for all three models
- **Alpha values**: 0.1 (p10), 0.5 (p50), 0.9 (p90)
- **CV**: TimeSeriesSplit, 5 folds, gap=48 periods
- **Save format**: `model.save_model(path)` / `lgb.Booster(model_file=path)`

### Optimiser
- **LP library**: PuLP with CBC solver
- **Slot duration**: 0.5 hours (30 min)
- **Energy per slot**: charge_rate_kw * 0.5 = kWh delivered
- **Carbon formula**: kWh * gCO2/kWh = gCO2
- **Agile prices**: loaded from `tests/fixtures/agile_prices.json` (static, 48 values, 4–35 p/kWh)

### API
- **Framework**: FastAPI with Pydantic v2
- **Lifespan**: models loaded on startup, stored in `app.state`
- **Test client**: `from fastapi.testclient import TestClient`

---

## External APIs

| API | Base URL | Auth |
|-----|----------|------|
| Carbon Intensity GB | `https://api.carbonintensity.org.uk` | None |
| Open-Meteo Archive | `https://archive-api.open-meteo.com/v1/archive` | None |
| ACN-Data | Manual download from `https://ev.caltech.edu/dataset` | Free account |

---

## Dependencies (pyproject.toml)

lightgbm, duckdb, pandas, pyarrow, scikit-learn, shap, fastapi, uvicorn, httpx, pytest, pytest-asyncio, workalendar, pulp, scipy, numpy

---

## Ralph Loop Instructions

1. Read `prd.json` to find the next incomplete task (lowest `id` where `completed: false`)
2. Read `progress.txt` for context on prior work
3. Implement exactly one task — no more
4. Run `cd energy-forecasting && uv run pytest tests/ -v` — fix any failures before committing
5. Update `prd.json`: set `"completed": true` for the finished task
6. Append a one-line summary to `progress.txt`
7. Commit with message: `[{task_id}] {task_title}: {one line summary}`
8. Exit

**Do not implement multiple tasks in one iteration.**
**Do not skip the test run.**
**Do not commit if tests fail.**
