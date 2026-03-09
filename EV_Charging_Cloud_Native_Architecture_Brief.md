# EV Charging Demand Optimisation — Cloud-Native Architecture Brief

**Status:** Planning  
**Cloud Providers:** UpCloud (Kafka broker) + Google Cloud Platform (everything else)  
**Development Method:** Claude Code with Ralph Loops

---

## Cloud Provider Strategy

The infrastructure is split across two providers deliberately:

**UpCloud** hosts a self-managed Kafka broker. You already have €250 credit — a suitable VM costs ~€10–12/month, meaning the Kafka layer is essentially free for the duration of development and the interview process. GCP's Managed Kafka service would cost £50–80/month and is the single biggest cost driver in an all-GCP setup.

**GCP** hosts everything else — BigQuery, Cloud Run microservices, Dataflow, GCS, Secret Manager. This directly mirrors Kaluza's Flex production stack and is the part worth showcasing.

The production migration path is trivial to articulate in an interview: "In production this would move to GCP Managed Kafka — one Secret Manager value changes, nothing else does." That framing demonstrates cost awareness without architectural compromise.

---

## Architecture

```
  External APIs                    UPCLOUD VM (~€10/month)
  ─────────────                   ┌──────────────────────────────────┐
  Carbon Intensity API ──────────▶│  Apache Kafka (self-hosted)       │
  NESO Data Portal     ──────────▶│                                   │
  Open-Meteo API       ──────────▶│  Topics:                          │
  UK Power Networks    ──────────▶│  - grid.generation.halfhourly     │
                                  │  - grid.carbon.intensity          │
                                  │  - weather.forecast               │
                                  │  - ev.sessions.raw                │
                                  │  - grid.generation.forecast       │
                                  │  - charging.schedules.optimised   │
                                  └────────────────┬──────────────────┘
                                                   │ SASL/SSL
                                  ┌────────────────▼──────────────────────────┐
                                  │               GCP PROJECT                  │
                                  │                                            │
                                  │  Go Ingestor Services (Cloud Run)          │
                                  │  ├── Grid Data Ingestor  (every 30 min)    │
                                  │  └── EV Session Ingestor (scheduled batch) │
                                  │                  │                         │
                                  │  Dataflow (Kafka → BigQuery sink)          │
                                  │                  │                         │
                                  │  BigQuery  ev_charging dataset             │
                                  │  ├── raw_generation                        │
                                  │  ├── raw_ev_sessions                       │
                                  │  ├── generation_features                   │
                                  │  └── optimised_schedules                   │
                                  │                  │                         │
                                  │  Python Feature Service (Cloud Run)        │
                                  │  Kafka consumer → DuckDB → BigQuery        │
                                  │                  │                         │
                                  │  Python ML Service (Cloud Run)             │
                                  │  LightGBM forecast + GMM EV behaviour      │
                                  │                  │                         │
                                  │  Go Optimiser (Cloud Run)                  │
                                  │  LP charging scheduler                     │
                                  │                  │                         │
                                  │  Streamlit Dashboard (Cloud Run)           │
                                  └────────────────────────────────────────────┘
```

---

## GCP Services

| Service | Role |
|---------|------|
| **Cloud Run** | All microservice hosting — serverless, pay per invocation, zero idle cost |
| **BigQuery** | Data warehouse — serverless SQL, native Kafka sink via Dataflow, Kaluza's stack |
| **GCS** | Raw data landing zone, Parquet staging, trained model artefacts versioned by date |
| **Dataflow** | Managed Kafka→BigQuery pipeline. Connects to external Kafka via bootstrap server config — no code change needed vs managed Kafka |
| **Cloud Scheduler** | Cron triggers for ingestor services every 30 minutes |
| **Secret Manager** | Kafka bootstrap server + SASL credentials, external API keys |
| **Artifact Registry** | Docker images for all Cloud Run services |
| **Cloud Monitoring** | Cloud Run and Kafka consumer lag metrics |

---

## Microservices

### Service 1 — Grid Data Ingestor (Go, Cloud Run)
Polls Carbon Intensity API, NESO Data Portal, and Open-Meteo every 30 minutes. Publishes half-hourly generation mix, carbon intensity, and weather to Kafka. Triggered by Cloud Scheduler. Go chosen for fast startup time (Cloud Run bills per 100ms) and lightweight goroutines for concurrent API polling — directly mirrors Kaluza Flex's Go services.

### Service 2 — EV Session Ingestor (Go, Cloud Run)
Reads UK Power Networks / ACN-Data CSV files from GCS, validates, normalises, and publishes session events to `ev.sessions.raw`. In a production extension this would consume from a live smart charger API stream.

### Service 3 — Feature Engineering Consumer (Python, Cloud Run)
Long-running Kafka consumer. Joins generation, carbon intensity, weather, and EV session streams. Uses DuckDB for in-memory window functions, rolling statistics, and lag feature computation before writing to BigQuery.

DuckDB is chosen over PySpark deliberately: at this data volume (millions of rows, not billions), DuckDB is faster, simpler, and an order of magnitude cheaper than Dataproc. The architecture isolates this choice to Service 3 — swapping PySpark in at scale requires no changes to any other service. Articulating that trade-off is an interview talking point.

### Service 4 — ML Forecasting Service (Python, Cloud Run)
Exposes `POST /forecast`. Loads LightGBM model from GCS at startup. Reads features from BigQuery, runs inference, returns P10/P50/P90 quantile forecasts and SHAP values for the top 5 features. Publishes forecasts to Kafka for the optimiser to consume.

### Service 5 — Charging Optimiser (Go, Cloud Run)
Consumes from `grid.generation.forecast` and `ev.sessions.raw`. For each session, runs a linear programming solve: minimise gCO2 during charging, subject to energy delivered ≥ required kWh before departure and charge rate ≤ hardware maximum. LP over RL because the single-vehicle problem has known constraints and a fixed horizon — LP is exact, fast, and fully interpretable.

---

## Claude Code + Ralph Loop Setup Strategy

The full infrastructure and all services are built using Claude Code running Ralph Loops. The workflow is: write a PRD with explicit, programmatically verifiable completion criteria, feed it to a Ralph loop, and come back to working infrastructure.

Each PRD ends with a concrete verification step and outputs `PHASE_COMPLETE` on success — giving Ralph an unambiguous exit condition rather than asking it to self-assess.

```bash
/ralph-loop "$(cat prds/prd-name.md)" --max-iterations 30 --completion-promise "PHASE_COMPLETE"
```

Once Kafka and GCP infrastructure are up, services can be developed in parallel using git worktrees with separate Ralph loops running simultaneously.

---

### Phase 1 — UpCloud Kafka VM

**PRD for Claude Code:**

Provision a VM on UpCloud using upcloud-cli: Ubuntu 24.04, 2 vCPU, 4GB RAM, 80GB SSD, UK-LON1 region. Assign a static IP.

Install OpenJDK 21 and Apache Kafka 3.7. Configure `server.properties` with `SASL_SSL` listener on port 9092, log retention 7 days, 3 partitions default. Generate a self-signed SSL certificate. Configure SASL/PLAIN authentication with a single user `kafkauser`. Write a systemd service file, enable and start it.

Create six topics with 3 partitions each: `grid.generation.halfhourly`, `grid.carbon.intensity`, `weather.forecast`, `ev.sessions.raw`, `grid.generation.forecast`, `charging.schedules.optimised`.

Configure UpCloud firewall to allow port 9092 inbound and port 22 from your IP only.

Write a Go test script that connects to the broker using SASL/PLAIN + SSL, produces 5 messages to `test.topic`, consumes them back, and exits 0 on success. Run the test script from a GCP Cloud Shell instance to verify cross-provider connectivity.

Output `PHASE_COMPLETE` on success.

---

### Phase 2 — GCP Infrastructure

**PRD for Claude Code:**

Create a GCP project and enable APIs: Cloud Run, BigQuery, Cloud Storage, Dataflow, Secret Manager, Artifact Registry, Cloud Scheduler, Cloud Build.

In Secret Manager, create placeholders for: `kafka/bootstrap-server`, `kafka/username`, `kafka/password`, `kafka/ssl-cert`.

Create BigQuery dataset `ev_charging` in europe-west2. Create four tables partitioned by `DATE(settlement_period)`: `raw_generation`, `raw_ev_sessions`, `generation_features`, `optimised_schedules` with appropriate schemas.

Create two GCS buckets in europe-west2: `ev-charging-raw-data` (with folders `/generation/`, `/ev-sessions/`, `/weather/`, `/errors/`) and `ev-charging-models`.

Create an Artifact Registry Docker repository `ev-charging-services` in europe-west2.

Capture everything in a `main.tf` Terraform file using the Google Cloud provider. Run `terraform plan` and confirm zero errors.

Output `PHASE_COMPLETE` on success.

---

### Phase 3 — Go Grid Data Ingestor

**PRD for Claude Code:**

Write a Go microservice that, on each invocation: fetches current and 48h forecast from the Carbon Intensity API; fetches half-hourly generation mix from NESO Data Portal; fetches 48h weather forecast from Open-Meteo for London, Manchester, and Edinburgh. For each record, publishes a JSON message to the appropriate Kafka topic. Reads Kafka credentials from environment variables populated by Secret Manager. Retries failed API calls up to 3 times with exponential backoff. Writes malformed responses to GCS `/generation/errors/` as dead letter.

Write a Dockerfile using a distroless base image. Build and push to Artifact Registry. Deploy to Cloud Run in europe-west2 with min instances 0, memory 256Mi, timeout 300s, secrets mounted from Secret Manager. Create a Cloud Scheduler job to trigger every 30 minutes.

Verify by triggering the Cloud Run service manually with `gcloud run jobs execute` and confirming messages appear in the Kafka topic using `kafka-console-consumer` against the UpCloud broker.

Output `PHASE_COMPLETE` on success.

---

### Phase 4 — Dataflow Kafka→BigQuery Pipeline

**PRD for Claude Code:**

Deploy the official Google "Apache Kafka to BigQuery" Dataflow flex template for three topics: `grid.generation.halfhourly` → `ev_charging.raw_generation`, `grid.carbon.intensity` → `ev_charging.raw_carbon_intensity`, `ev.sessions.raw` → `ev_charging.raw_ev_sessions`. Configure each job with the UpCloud bootstrap server and SASL credentials from Secret Manager, JSON message format, and a dead letter table for malformed messages. Region: europe-west2.

Verify by producing 10 test messages to `grid.generation.halfhourly` from the Go test script, then querying `SELECT COUNT(*) FROM ev_charging.raw_generation` and confirming the count increases by 10.

Output `PHASE_COMPLETE` on success.

---

### Phase 5 — Python Feature Engineering Service

**PRD for Claude Code:**

Write a long-running Python Kafka consumer that consumes from `grid.generation.halfhourly`, `grid.carbon.intensity`, and `weather.forecast`. Buffer records in a DuckDB in-memory database. Every 30 minutes, compute a feature table using DuckDB SQL: rolling 7-day averages for wind and solar generation; rolling 24h average carbon intensity; lag features at t-1, t-2, t-48, t-336; calendar features including hour of day, day of week, month, is_weekend, is_bank_holiday (UK bank holidays via `workalendar`); weather features from the joined weather stream. Write the computed features to `ev_charging.generation_features` in BigQuery using the Storage Write API. Commit Kafka offsets only after successful BigQuery write. Log consumer lag as a custom Cloud Monitoring metric.

Deploy to Cloud Run with min instances 1 (long-running consumer), memory 1Gi.

Verify with BigQuery SQL assertions: records exist for the last 24h, no NULLs in rolling average columns, lag_t48 values match raw_generation records from 24h prior.

Output `PHASE_COMPLETE` on success.

---

### Phase 6 — Python ML Forecasting Service

**PRD for Claude Code:**

Write a model training script that queries `ev_charging.generation_features` from BigQuery, trains three LightGBM quantile models (alpha=0.1, 0.5, 0.9) with TimeSeriesSplit (5 folds), evaluates against persistence and naive seasonal baselines, and saves model artefacts to `gs://ev-charging-models/lgbm/{date}/`. Run training locally first.

Write a Cloud Run service that exposes `POST /forecast`. On startup, load the latest model from GCS. On request, query BigQuery for the requested window plus 2 weeks of history for lag computation, run inference for all three quantiles, compute SHAP values for the top 5 features, publish the P50 forecast to `grid.generation.forecast` Kafka topic, and return the full response as JSON.

Deploy to Cloud Run with min instances 1, memory 2Gi, CPU 2.

Verify by calling the endpoint with curl and confirming the response includes P10/P50/P90 values and SHAP importances. Confirm messages appear in the Kafka topic.

Output `PHASE_COMPLETE` on success.

---

### Phase 7 — Go Charging Optimiser

**PRD for Claude Code:**

Write a Go Kafka consumer that consumes from `ev.sessions.raw` and `grid.generation.forecast`. For each EV session, formulate and solve a linear programming problem using gonum: decision variables are charge rate per 30-minute slot; objective is to minimise the sum of (charge_rate[t] × carbon_intensity[t]); constraints are total energy delivered ≥ requested kWh, charge rate per slot within hardware limits, charge rate zero outside the plug-in window. Also fetch real-time Agile tariff prices from the Octopus Energy public API to compute cost saving alongside carbon saving. Compute the dumb charging baseline (constant rate over the window) for comparison. Publish the optimised schedule to `charging.schedules.optimised` and write results to BigQuery `optimised_schedules`.

Deploy to Cloud Run, min instances 0.

Verify by producing 100 simulated EV sessions to `ev.sessions.raw` and querying `SELECT AVG(carbon_saving_g) FROM optimised_schedules` to confirm a positive average carbon saving.

Output `PHASE_COMPLETE` on success.

---

### Phase 8 — Streamlit Dashboard

**PRD for Claude Code:**

Build a Streamlit dashboard with four views, deployed to Cloud Run.

**Live Grid:** Pull current carbon intensity directly from the Carbon Intensity API. Show gCO2/kWh with green/amber/red colour coding, generation mix as a donut chart, and 24h trend line. Auto-refresh every 5 minutes.

**Generation Forecast:** Call the ML service `/forecast` endpoint for the next 48h. Show wind generation P10/P50/P90 as a fan chart, carbon intensity P50 as a line, and SHAP top 5 feature importances as a horizontal bar chart alongside MAE vs the persistence baseline.

**Single Vehicle Optimiser:** User inputs plug-in time, departure time, energy needed (kWh), and max charge rate (kW). On submit, call the optimiser service. Show the optimised charging schedule as a step chart overlaid with carbon intensity forecast. Display carbon saving (gCO2) and cost saving (£) vs dumb charging. Highlight the greenest 4-hour window.

**Fleet Simulation:** Simulate 100 EV sessions using the GMM behaviour model. Show aggregate demand curves (optimised vs unmanaged), total fleet carbon saving, and percentage of demand shifted to low-carbon windows.

Deploy to Cloud Run with min instances 1. Verify by navigating all four views and confirming the single-vehicle optimiser returns a result.

Output `PHASE_COMPLETE` on success.

---

## Parallelisation Plan

Phases 1 and 2 must run sequentially (infrastructure must exist before services). Once complete, Phases 3 and 4 can run in parallel worktrees, and once data is flowing, Phases 5, 6, and 7 can also run in parallel:

```bash
# After phases 1-2:
git worktree add ../ev-ingestor -b feature/ingestor
git worktree add ../ev-dataflow -b feature/dataflow

# Terminal 1
cd ../ev-ingestor && /ralph-loop "$(cat prds/prd3.md)" --max-iterations 25 --completion-promise "PHASE_COMPLETE"

# Terminal 2
cd ../ev-dataflow && /ralph-loop "$(cat prds/prd4.md)" --max-iterations 15 --completion-promise "PHASE_COMPLETE"
```

---

## Cost Estimate

| Service | Provider | Monthly cost |
|---------|----------|-------------|
| Kafka VM (2 vCPU, 4GB RAM) | UpCloud | ~€10–12 (covered by €250 credit) |
| Cloud Run (all services) | GCP | ~£5–15 |
| BigQuery | GCP | ~£5–10 |
| GCS | GCP | ~£1–2 |
| Dataflow (3 streaming jobs) | GCP | ~£10–20 |
| Scheduler, Secret Manager | GCP | <£2 |
| **Total** | | **~£22–50/month** |

Self-hosting Kafka on UpCloud saves £50–80/month vs GCP Managed Kafka. The UpCloud credit covers approximately two years of the VM.

---

## Interview Talking Points

- **Hybrid cloud pragmatism:** Kafka on UpCloud with a trivial migration path to GCP Managed Kafka shows cost awareness without architectural compromise — the bootstrap server URL in Secret Manager is the only thing that changes.
- **Ralph Loops for infrastructure:** Autonomous Claude Code loops driven by PRDs with verifiable completion criteria reflect modern AI-augmented engineering practice, increasingly relevant at scale.
- **Go/Python boundary:** Go for latency-sensitive ingestors and the LP optimiser; Python where the ML and analytics ecosystem matters more than raw performance. Justifying the boundary matters more than the choice itself.
- **DuckDB vs PySpark:** Right tool for the data volume, architecture designed so the switch to Dataproc is isolated to one service.
- **LP over RL:** Exact and interpretable for the constrained single-vehicle problem. RL becomes appropriate at fleet scale with live grid feedback.
- **Kafka as contract boundary:** Each microservice only knows about its topics — independently deployable, independently testable.
- **IaC from day one:** Terraform makes the system reproducible and teardownable. Important for cost control and good production discipline.

---

## Relationship to Other Projects

This is deliberately over-engineered for the data volume — that's the point. A single laptop could run this entire pipeline. The cloud-native microservices design demonstrates how you would build this at Kaluza's production scale, not just that you can train a LightGBM model. That framing should be explicit in both the README and any interview discussion.

The NHS A&E Forecasting project uses the same core ML approach without the distributed systems complexity — building that first provides a solid ML foundation before tackling the infrastructure here.
