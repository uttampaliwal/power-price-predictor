# Conformal Prediction for Indian Electricity Price Forecasting

Production-grade uncertainty quantification for day-ahead electricity prices on the Indian Energy Exchange (IEX) at 15-minute resolution.

**Problem:** Point forecasts alone are insufficient for trading and risk management. A forecast of Rs3,000/MWh with a 90% interval of [Rs2,500, Rs3,500] is fundamentally different from [Rs1,000, Rs5,000]. Electricity markets need calibrated uncertainty estimates.

**Approach:** Wrap gradient-boosted tree models with conformal prediction to produce distribution-free prediction intervals with guaranteed marginal coverage (target: 90%).

**Key Results:**
- SCP + LightGBM achieves 91.4% coverage (gap: +1.4%) on 58K holdout points
- EnbPI achieves 90.6% coverage across all base learners (gap: +0.61%)
- Coverage varies 62-98% by hour — evening peaks under-cover when stakes are highest

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                             │
├──────────────────────────────────────────────────────────────┤
│  IEX DAM (urllib scrape)──┐                                  │
│  Open-Meteo Weather    ──┼──▶ Feature Engineering (18 feats)│
│  Calendar/Holidays     ──┘         │                         │
│                                    ▼                         │
│                    ┌─────────────────────────────┐          │
│                    │   Train/Calibration/Holdout │          │
│                    │   175K / 35K / 58K rows     │          │
│                    └─────────────┬───────────────┘          │
│                                  │                           │
├──────────────────────────────────┼──────────────────────────┤
│                    MODEL LAYER   │                           │
│                                  ▼                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ XGBoost  │  │ LightGBM │  │  Ridge   │                  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                  │
│       │              │              │                        │
│       ▼              ▼              ▼                        │
│  ┌──────────────────────────────────────────┐               │
│  │     Conformal Prediction Layer           │               │
│  │  SCP │ ACP │ CQR │ EnbPI │ SPCI         │               │
│  └──────────────────┬───────────────────────┘               │
│                     │                                        │
├─────────────────────┼───────────────────────────────────────┤
│                 DEPLOYMENT                                   │
│                     ▼                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ FastAPI  │  │  Docker  │  │  MLflow  │                  │
│  │ :8000    │  │          │  │ (SQLite) │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
└──────────────────────────────────────────────────────────────┘
```

## Results

### Conformal Prediction Coverage (58K holdout points, target: 90%)

| Base Model | Method | PICP | Gap | PINAW | Winkler |
|------------|--------|------|-----|-------|---------|
| XGBoost | SCP | 89.97% | -0.03% | 94.03 | 5960 |
| XGBoost | EnbPI | 90.61% | +0.61% | 76.06 | 6366 |
| LightGBM | SCP | 91.39% | +1.39% | 102.60 | 6014 |
| LightGBM | SPCI | 90.37% | +0.37% | 85.41 | 5675 |
| Ridge | SCP | 88.97% | -1.03% | 66.84 | 6375 |

### Coverage by Hour (LightGBM + SCP)

| Period | Hours | Coverage | Mean Price |
|--------|-------|----------|------------|
| Evening peak | 18-22h | 62-76% | Rs6,000-7,700/MWh |
| Midday trough | 10-15h | 96-98% | Rs1,600-2,500/MWh |
| Morning ramp | 06-09h | 73-89% | Rs2,600-4,900/MWh |

**Finding:** Coverage correlates with price level (r = -0.87). The method under-covers precisely when stakes are highest.

### Trading Simulation

| Strategy | Sharpe | Win Rate | P&L |
|----------|--------|----------|-----|
| Interval Arbitrage | 19.3 | 97.7% | Rs3,519M |
| Confidence-Weighted | 14.1 | 80.3% | Rs2,180M |

## Quick Start

```bash
# Install
pip install -e .

# Run full evaluation pipeline
python src/run_conformal.py

# Run trading simulation
python src/trading_simulation.py

# Run hourly coverage analysis
python src/hourly_coverage.py

# Start API server
uvicorn src.api_v2:app --port 8000

# Run the operator dashboard
pip install streamlit
streamlit run app.py

# Docker
docker build -t power-price-predictor .
docker run -p 8000:8000 power-price-predictor
```

## Project Structure

```
power-price-predictor/
├── app.py                    # Streamlit dashboard (Live Tracker / Sandbox / Scorecard / Data Mgmt)
├── src/
│   ├── config.py             # Constants, paths, feature lists, HOLDOUT_START_DATE
│   ├── fetch_data.py         # IEX DAM scraper (urllib + read_html)
│   ├── fetch_weather.py      # Open-Meteo API client
│   ├── preprocess.py         # Feature engineering (18 features)
│   ├── probabilistic.py      # SCP, ACP, CQR, EnbPI, SPCI
│   ├── run_conformal.py      # Conformal sweep → results/conformal/
│   ├── hourly_coverage.py    # Hourly coverage analysis
│   ├── trading_simulation.py # Interval-trading strategies (paper numbers)
│   ├── backtest.py           # Block-shift strategy (alternative narrative)
│   ├── benchmark.py          # Model comparison + drift figures
│   ├── tracking_cp.py        # MLflow experiment tracking (runs not committed)
│   ├── powerbi_exporter.py   # Power BI export (→ powerbi_data/)
│   ├── api.py                # FastAPI v1 (legacy, not deployed)
│   ├── api_v2.py             # FastAPI v2 serving (conformal intervals; Dockerized)
│   └── models/
│       ├── naive_model.py / ridge / random_forest / xgboost / lightgbm
│       ├── *_no_weather.py   # ablation twins → models_no_weather/
│       └── (experimental: lstm / prophet / tft — no trained artifacts)
├── data/                     # raw yearly CSVs + processed parquet (committed snapshot)
├── models/  models_no_weather/ # trained artifacts + metrics CSVs (committed)
├── predictions/              # holdout predictions, benchmark CSVs, backtest/
├── results/conformal/        # interval sweep snapshots (committed yardsticks)
├── paper/                    # LaTeX paper + figures/ + submission materials
├── blog/                     # research write-ups
├── docs/                     # POWERBI guides, learning summary
└── tests/                    # pytest suite (25 tests)
```

## Data

- **Market:** IEX Day-Ahead Market, 15-minute blocks (96/day)
- **Period:** Jan 2020 - Aug 2026 (233K rows)
- **Training:** Jan 2020 - Dec 2024 (175K rows)
- **Holdout:** Jan 2025 - Aug 2026 (58K rows, untouched)
- **Features:** 18 (calendar, autoregressive lags, rolling stats, weather)
- **Source:** [IEX Market Snapshot](https://www.iexindia.com/market-data/day-ahead-market/market-snapshot)

## Experiments

MLflow experiment tracking is available (runs write to `mlruns/`, not committed):

```bash
pip install mlflow
python src/tracking_cp.py
mlflow ui  # View at http://localhost:5000
```

Committed metric snapshots (`predictions/`, `results/conformal/`) are the registry of record — see ROADMAP.md for the planned manifest-based registry.

## Tests

```bash
python -m pytest tests/ -v  # 25 tests
```

## License

MIT
