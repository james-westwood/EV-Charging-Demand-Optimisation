# EV Charging Demand Optimisation
**Status:** active
**Priority:** critical
**Progress:** ~70%
**Last updated:** 2026-06-16
**Repo:** `~/Programming/personal/EV-Charging-Demand-Optimisation`

## What it does
MLOps end-to-end pipeline for EV charging demand forecasting; primary Zoomcamp learning vehicle.

## Why it matters
Primary SDG learning project — directly builds toward ML Engineer roles at energy companies. Covers the full MLOps stack: data ingestion, feature engineering, model training, deployment, monitoring.

## Next actions
- [x] Complete feature engineering pipeline
- [x] Set up MLflow experiment tracking
- [x] Build model serving layer (FastAPI on Cloud Run)
- [ ] Deploy EV Charging case-study page to Cloudflare Pages (preview ready)
- [ ] Productionise LP optimiser endpoint
- [ ] Add session model (GMM)

## Notes
### 2026-06-16 — API CORS fix + redeployment
- **Problem:** CORS blocked on Cloud Run API — the frontend widget at `james-westwood.dev` couldn't fetch live forecasts
- **Fix:** Expanded `allow_origins` in `energy-forecasting/src/api/main.py` to include:
  - `https://www.james-westwood.dev`
  - `http://localhost:4322` (Astro dev)
  - `http://192.168.68.245:8080` (local network)
  - `http://127.0.0.1:8080`
- **Commit:** `0efa095` on `feat/epic7-charging-optimizer`
- **Status:** API redeployed, live forecast now working in the LP scheduler widget
- **Widget:** `https://james-westwood.dev/projects/linear-programming-scheduler`

### 2026-06-02 — EV charging case-study page (preview)
- Portfolio widget page (`/projects/ev-charging`) built in preview on james-westwood.dev — not yet deployed
- Case-study page includes: animated SVG scene, React widget, architecture flow, stack & source
- SMIL dot animation restored after CSS experiments failed cross-browser
