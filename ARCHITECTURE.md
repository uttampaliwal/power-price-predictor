# ARCHITECTURE.md

## What this is

An end-to-end **price-cap forecasting + uncertainty research project** for the Indian Energy Exchange Day-Ahead Market (IEX DAM): raw market data in, per-model probabilistic forecasts out, with conformal-prediction intervals that have been backtested for coverage and tradability, a Streamlit ops dashboard, and a serving API.

Stack: **Python (pandas/sklearn/XGBoost/LightGBM)**, optional **torch** (LSTM), **Streamlit** dashboard (`app.py`), **FastAPI** serving (`src/api_v2.py`), **ruff/pytest/CI**, optional MLflow. License: MIT.

## Data flow (one direction, no loops)

```
IEX "Market Snapshot" HTML (public page)
        │  src/fetch_data.py (urllib + pd.read_html + UA header; NO Playwright)
        ▼
data/raw/training/dam_2020..2024.csv   data/raw/holdout/dam_2025..2026.csv
        │              + src/fetch_weather.py → Open-Meteo archive (Delhi & Mumbai proxies)
        ▼
src/preprocess.py → time features, lags/rolls, weather features, holiday flags
        ▼
data/processed/{training,holdout}_features[_no_weather].parquet
        │
        ├──► src/models/*_model.py        (5 planned families — naive, ridge, random_forest,
        │                                   xgboost, lightgbm — each writes models/<fam>/ +
        │                                   predictions/<fam>/test_predictions.csv + metrics CSVs)
        │
        └──► src/models/<name>_model.py --no-weather  (same families, weather feature ablated
                                              → models_no_weather/, <name>_no_weather artifacts)
        ▼
src/evaluate.py (metrics) · src/evaluate_by_segment (by hour/season)
        ▼
src/benchmark.py → predictions/benchmark_metrics.csv, drift_by_month.csv, PNG figures
        ▼
src/run_conformal.py → src/probabilistic.py methods (SplitConformal, AdaptiveConformal,
                        CQR, QuantileForecaster, EnbPI, SPCI)
        ▼
results/conformal/*.csv|json   → src/hourly_coverage.py → src/trading_simulation.py
                                                      → src/backtest.py (alternative strategy)
```

Time-based split (no leakage): train 2020–2024, holdout 2025+ (`src/config.py::HOLDOUT_START_DATE = 2025-01-01`). Means/medians for IEX cap estimation are a research sensitivity, not the headline.

## Surfaces

| Surface | File | Runs as |
|---|---|---|
| Dashboard | `app.py` (Streamlit, 4 pages: Live Tracker / Forecast Sandbox / Model Scorecard / Data Management) | `streamlit run app.py` |
| Serving API (v2, current) | `src/api_v2.py` (FastAPI; `/health`, `/predict` with conformal intervals, `/metrics/{model}`) | `uvicorn src.api_v2:app --port 8000` / Docker |
| Serving API (v1, legacy) | `src/api.py` | maintained, not deployed |
| Power BI export | `src/powerbi_exporter.py` (+ `powerbi_live_sync.py`) → `powerbi_data/` CSVs (generated; see docs/POWERBI_SETUP.md) | python script |

## Key design decisions

- **Everything generatable is deterministic:** seeds fixed at 42, yearly raw CSVs committed, so committed metric snapshots (`predictions/benchmark_metrics.csv`, `results/conformal/`) are checkable at clone time against code.
- **Ablation axis built into the tree:** each trainer's `--no-weather` mode answers "do weather proxies add value?" with identical seeds/protocols.
- **Results live in git** as CSV snapshots (experiments registry substitute). MLflow exists as a dep but registries aren't committed yet.
- **Two evaluation frameworks:** conformal-interval trading (`src/trading_simulation.py`, quoted in README/paper) vs block-shift backtest (`src/backtest.py`). Their P&Ls differ by design; keep them separated by file and docs.

## Debts (see ROADMAP.md for full queue)

The ablation axis is parameterized into the four gradient/baseline trainers (merged 2026-11), root `app.py` is a 1,300-line monolith mixing pages+subprocess orchestration, v1 API overlaps v2, and no lockfile pins pandas/xgboost — all accepted for now, prioritized in ROADMAP.md.
