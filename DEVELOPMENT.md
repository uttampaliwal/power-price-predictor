# DEVELOPMENT.md

## Prerequisites

Python >= 3.9 (CI matrix: 3.10/3.11/3.12; Docker uses 3.11-slim). `pip` is fine; `uv` works too.

```bash
pip install -e .            # core: pandas, sklearn, xgboost, lightgbm, fastapi, uvicorn, matplotlib, seaborn
pip install -e ".[dev]"     # + pytest, pytest-cov
pip install streamlit plotly # dashboard deps (requirements.txt lists them as core)
```

## Daily commands

```bash
# research pipeline (in order)
python src/fetch_data.py              # IEX DAM scrape (appends current year; data refresh!)
python src/fetch_weather.py           # Open-Meteo weather pull
python src/preprocess.py              # → data/processed/*.parquet
python src/models/xgboost_model.py    # train one family (writes models/ + predictions/)
python src/benchmark.py               # compare + drift figures
python src/run_conformal.py           # interval methods → results/conformal/
python src/hourly_coverage.py
python src/trading_simulation.py
python src/backtest.py                # alternative strategy set (separate narrative!)

# surfaces
streamlit run app.py                  # 4-page ops dashboard
uvicorn src.api_v2:app --port 8000    # serving API (Docker path)
python src/powerbi_exporter.py        # → powerbi_data/

# quality
python -m pytest tests/ -v --tb=short
ruff format src/ tests/ && ruff check src/ tests/
```

Full pipeline chain is also documented in `docs/OPERATIONS_GUIDE.md` (note: its `daily/` predictions layout and `pipeline_report.html` refer to an older iteration — trust the file list above).

## Repository map

```
app.py                  Streamlit dashboard (Live Tracker / Sandbox / Scorecard / Data Mgmt)
app/ — (not yet)        single-file monolith today; see ROADMAP.md
src/
  config.py             paths, FEATURE_COLS, TARGET_COL, HOLDOUT_START_DATE, MODEL_LIST
  fetch_data.py         IEX "Market Snapshot" scraper (urllib + read_html)
  fetch_weather.py      Open-Meteo archive fetch (Delhi & Mumbai proxies)
  preprocess.py         feature engineering → parquet
  probabilistic.py      SCP / ACP / CQR / EnbPI / SPCI / QuantileForecaster
  run_conformal.py      conformal sweep → results/conformal/
  hourly_coverage.py    coverage by hour slice
  trading_simulation.py interval-trade P&L (README/paper numbers)
  backtest.py           block-shift strategy (alternative numbers)
  benchmark.py          benchmark table + drift figures (writes predictions/*.png)
  evaluate.py           metrics shared across models/evals
  models/               one trainer file per family (+ *_no_weather twins)
  api.py (v1, legacy)   FastAPI v1 (feature-parity serving)
  api_v2.py             FastAPI v2 (conformal intervals; Dockerized)
  powerbi_exporter.py   Power BI CSV writer; powerbi_live_sync.py
data/                   raw (committed) + processed parquet (committed)
models/ models_no_weather/   trained artifacts + metrics CSVs (committed)
predictions/            per-model holdout predictions + backtest/ + benchmark CSVs
results/conformal/      interval sweep outputs (committed snapshots)
paper/                  LaTeX + figures + as-submitted materials
blog/                   research write-ups
docs/                   POWERBI / learning-summary docs
tests/                  pytest suite
```

## Module boundaries

- `src/config.py` is the single source of paths/features/dates — extend it, don't shadow it.
- Model trainers are flat scripts by design (one file = one family); their docstring states what they write.
- `app.py` owns UI + subprocess orchestration only; pipeline logic stays in `src/`.
- Anything touching "money" (P&L/Sharpe claims) lives in `trading_simulation` or `backtest` — never blend them.

## Data model & conventions

- Raw = `data/raw/{training,holdout}/dam_YYYY.csv` (96 blocks/day; schema documented in `data/README.md`).
- Holdout split: `date(2025,1,1)` onward, never train on it.
- Seeds: 42 everywhere; dataset refresh = new experiment (see AGENTS.md rule 4).
