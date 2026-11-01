# DEPLOYMENT.md

## Deploy target reality

This project ships as **research code first**; of the two "live" surfaces, only the API is containerized. The Streamlit dashboard is intentionally local-only today (roadmap: containerize it).

## API (v2) in Docker — the supported path

```bash
docker build -t power-price-predictor .
docker run -p 8000:8000 power-price-predictor
```

- Image: `python:3.11-slim`, `pip install -e .`, copies `src/ data/ results/ models/` into the image, serves `src.api_v2:app` on :8000 with a `/health` check.
- `docker compose up -d` (single service `api`): mounts `./data`, `./results`, `./models`, `restart: unless-stopped`, healthcheck via curl-homebrew (`/health`).

Endpoints (compliant with `src/api_v2.py`): `GET /health`, `POST /predict` (point forecast + conformal intervals; model selected per family), `GET /metrics/{model}`.

Serving notes:
- The image carries the **committed** artifacts (models/, data/, results/) — a fresh pull = the committed experiment's state. This is intentional: deployment == the paper's numbers.
- If you regenerate artifacts locally and rebuild, you're deploying a *new experiment*; change OpenAPI `description`/Release notes rather than silently swapping weights.
- CORS is wildcard; front a reverse proxy + auth before any multi-user exposure.

## Streamlit dashboard (local ops)

```bash
pip install streamlit
streamlit run app.py    # uses committed models/ + predictions/ for Scorecard/Ballparks
```

Data Management page shells out to pipeline scripts — it's a local operator tool; do not expose publicly.

## Power BI (BI consumers)

```bash
python src/powerbi_exporter.py       # writes powerbi_data/*.csv (8 exports)
python src/powerbi_live_sync.py      # refresh helper
```

Setup + build guide: `docs/POWERBI_SETUP.md`, `docs/POWERBI_DASHBOARD_BUILD_GUIDE.md`.

## CI

Lint (ruff format --check + ruff check) and pytest on Python 3.10/3.11/3.12; no image-build or deploy job yet (roadmap).
