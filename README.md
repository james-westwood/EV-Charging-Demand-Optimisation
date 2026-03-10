# EV Charging Demand Optimisation

Forecast grid carbon intensity and EV charging demand, then optimise charge schedules to minimise carbon emissions and cost. Built as a deliberately over-engineered local MVP — the architecture mirrors what a production system at Kaluza/Flex scale would look like, even though a single laptop is enough to run it.

> **Cloud-native version:** see [`EV_Charging_Cloud_Native_Architecture_Brief.md`](./EV_Charging_Cloud_Native_Architecture_Brief.md) for the full UpCloud + GCP design with Kafka, BigQuery, Cloud Run, and Dataflow.

---

## What this repo is

A **local-only ML MVP** that runs end-to-end on a single machine:

1. Pulls live grid and weather data from public APIs
2. Validates and engineers features into 30-minute settlement period windows
3. Trains LightGBM quantile models (P10/P50/P90) to forecast carbon intensity
4. Models EV session behaviour using a Gaussian Mixture Model
5. Optimises individual charging schedules with linear programming
6. Exposes everything through a local FastAPI

The production cloud version replaces Parquet files with BigQuery, the local scheduler with Cloud Scheduler, and the FastAPI with Cloud Run microservices — but the ML logic, feature pipeline, and LP formulation are identical.

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ managed with `uv` |
| Data collection | `httpx` — Carbon Intensity API, Open-Meteo, ACN-Data |
| Storage | Parquet (local), `pyarrow` |
| Feature engineering | `pandas`, `numpy` |
| ML forecasting | `lightgbm` (quantile regression), `shap` |
| EV behaviour model | `scikit-learn` GaussianMixture |
| Optimiser | `PuLP` / `scipy` (linear programming) |
| API | `FastAPI` + `uvicorn` |
| Testing | `pytest`, `httpx` mock transport |

---

## Project structure

```
energy-forecasting/
├── src/
│   ├── data/
│   │   ├── collectors/        # API clients: carbon intensity, generation mix, weather, EV sessions
│   │   └── validators/        # Schema and range validation for each data source
│   ├── features/              # Feature engineering pipeline
│   │   ├── alignment.py       # Align all sources to 30-min settlement periods
│   │   ├── weather_join.py    # Interpolate weather onto the grid index
│   │   ├── rolling.py         # 7-day rolling averages
│   │   ├── lags.py            # Lag features (t-1, t-2, t-48, t-336)
│   │   ├── calendar.py        # Hour, day-of-week, bank holidays
│   │   ├── penetration.py     # Wind/solar penetration %
│   │   └── pipeline.py        # Compose all steps into one call
│   ├── models/
│   │   ├── forecasting/       # LightGBM quantile trainer, CV, metrics, SHAP, artefacts
│   │   └── ev_behaviour/      # GMM session model
│   ├── optimiser/             # LP charge scheduler
│   ├── api/                   # FastAPI app and endpoint schemas
│   └── logging_config.py      # Structured JSON logging
├── tests/                     # Mirrors src/ structure, all HTTP mocked
├── data/
│   ├── raw/                   # Downloaded Parquet files by source
│   └── features/              # Feature store output
└── saved_models/              # Versioned model artefacts by date (YYYY-MM-DD/)
```

---

## Epics

| Epic | Status | Description |
|---|---|---|
| 0 — Project Setup | ✅ Complete | Scaffold, dependencies, logging, test fixtures |
| 1 — Data Acquisition | ✅ Complete | API clients, retry logic, incremental fetch, raw Parquet save |
| 2 — Data Validation | ✅ Complete | Schema checks, range validation, validation report |
| 3 — Feature Engineering | ✅ Complete | Full pipeline: alignment → weather → rolling → lags → calendar |
| 4 — ML Model Training | 🔨 In progress | Time-series CV, LightGBM quantile, baselines, SHAP, artefacts |
| 5 — EV Behaviour Model | ⏳ Pending | GMM fit on ACN session data, session sampler |
| 6 — Charging Optimiser | ⏳ Pending | LP formulation, carbon/cost saving vs dumb charging baseline |
| 7 — Local Forecast API | ⏳ Pending | FastAPI wrapping the trained models and optimiser |

---

## Development method

This project is built using **Ralph Loops** — an autonomous multi-agent development pattern where an AI agent reads a PRD, implements one task at a time, runs tests, commits, and iterates until the PRD is complete.

Each task follows this flow:

```
prd.json (task spec)
    │
    ▼
Feature branch created
    │
    ├── CODER (Claude or Gemini, randomly assigned)
    │     implements in atomic commits:
    │     [task-id] title: implement
    │     [task-id] title: add tests
    │     [task-id] title: mark complete
    │
    ▼
PR opened on GitHub
    │
    ├── REVIEWER (the other AI)
    │     reads gh pr diff, posts review comment
    │     ends with APPROVED or CHANGES REQUESTED
    │
    ▼
Auto-merged → main
```

Claude and Gemini are randomly assigned coder/reviewer roles per task, so each PR has a cross-model review. The human developer (James) owns Epics 4–6 (the ML and optimisation work) — those tasks are marked `"owner": "human"` in `prd.json` and the loop stops automatically when it reaches them.

To run the loop:

```bash
./ralph-loop.sh --max 10
```

To run a single iteration:

```bash
./ralph-once.sh
# or force a specific task:
./ralph-once.sh --task 1.4
```

---

## Quickstart

```bash
cd energy-forecasting
uv sync                          # install dependencies
uv run pytest tests/ -v          # run all tests (~60 passing)
```

To fetch live data:

```bash
uv run python -c "
from src.data.collectors.carbon_intensity import fetch_carbon_intensity
from datetime import datetime, timedelta, timezone
end = datetime.now(timezone.utc)
start = end - timedelta(days=7)
df = fetch_carbon_intensity(start, end)
print(df.head())
"
```

---

## Interview talking points

- **Quantile regression over point estimates:** P10/P50/P90 forecasts give uncertainty bounds — standard in energy dispatch where knowing the range matters as much as the median.
- **LP over RL for the optimiser:** the single-vehicle charge scheduling problem has known constraints and a fixed horizon. LP is exact, fast, and fully interpretable. RL becomes relevant at fleet scale with live grid feedback.
- **Time-series CV:** random CV would leak future data into training. TimeSeriesSplit with a 48-period gap (1 day) ensures validation always follows training chronologically.
- **DuckDB vs PySpark (cloud version):** at this data volume DuckDB is faster and simpler; the architecture isolates that choice to one service, so swapping Dataproc in at scale changes nothing else.
- **Ralph Loops + multi-agent review:** autonomous AI-driven development with cross-model code review (Claude ↔ Gemini) reflects where engineering practice is heading — and produced 20+ reviewed PRs with atomic commits.
