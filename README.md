# ⚡ IEX Power Price Predictor

### Machine Learning–Based Forecasting of Indian Electricity Prices (Day-Ahead Market) at 15-Minute Granularity

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0+-red.svg)
![Power BI](https://img.shields.io/badge/Power%20BI-Integration-yellow.svg)

---

## 📌 Project Overview

**Problem:** Indian electricity prices in the Day-Ahead Market (DAM) are highly volatile — ranging from ₹2,000 to ₹10,000/MWh within hours based on demand, supply, weather, and grid conditions. Accurate price forecasting helps power traders, producers, and DISCOMs optimize bidding strategies and reduce costs.

**Solution:** An end-to-end ML pipeline that predicts IEX DAM prices for all 96 daily time-blocks (15-minute intervals) with **87% accuracy (R² = 0.871)**, achieving a **62% improvement over naive baseline**.

**Impact:** Enables data-driven bidding decisions, cost optimization, and real-time market intelligence for energy stakeholders.

---

## 🏆 Results Summary

| Model | R² Score | WAPE | RMSE | AUC-ROC | Best For |
|-------|----------|------|------|---------|----------|
| **XGBoost** | **0.871** | 14.1% | 1030 | 0.606 | Price forecasting |
| **LightGBM** | **0.872** | 14.0% | 1030 | 0.602 | Price forecasting |
| **Random Forest** | 0.864 | 14.4% | 1060 | 0.593 | Robust predictions |
| **Ridge** | 0.830 | 16.3% | 1183 | 0.511 | Linear baseline |
| **Naive Baseline** | 0.539 | 25.2% | 1951 | 0.633 | Comparison |

**Key Achievements:**
- ✅ 62% improvement over naive baseline (R²: 0.539 → 0.871)
- ✅ 14.1% average error (WAPE) — operationally acceptable
- ✅ Time-based train/test split (no data leakage)
- ✅ Validated on 20 months of unseen holdout data (Jan 2025 – Aug 2026)

---

## 📊 Model Performance Visualization

### Actual vs Predicted Prices (Sample Holdout Period)

![Model Comparison](images/model_comparison.png)

*XGBoost and LightGBM consistently track actual prices with <15% average error across all 96 daily blocks.*

### Price Prediction Drift Analysis (Month-wise)

![Drift Plot](images/drift_plot.png)

*Model maintains stable performance over 20 months of unseen holdout data (Jan 2025 – Aug 2026).*

---

## 🧠 Modeling Approach

### Algorithms Tried

| Algorithm | Type | R² | Status |
|-----------|------|-----|--------|
| Naive Baseline | Statistical | 0.539 | ✅ Baseline |
| Ridge Regression | Linear ML | 0.830 | ✅ Active |
| Random Forest | Bagging | 0.864 | ✅ Active |
| XGBoost | Gradient Boosting | **0.871** | ✅ Champion |
| LightGBM | Gradient Boosting | **0.872** | ✅ Active |
| LSTM | Deep Learning | — | 🔲 Stub |

### Why XGBoost Won

1. **Non-linear patterns**: Captures complex interactions between time, demand, supply, and weather
2. **Feature importance**: Interpretable model drivers (hour, day-of-week, season)
3. **Robustness**: Handles price cap anomalies (₹10,000/MWh) better than linear models
4. **Speed**: Fast training and inference for daily production runs

### Train/Test Split Strategy

- **Training**: Jan 2020 – Dec 2024 (~175K rows)
- **Holdout**: Jan 2025 – Aug 2026 (20 months, ~58K rows, unseen data)
- **Method**: Time-based split (NOT random) to simulate real-world deployment
- **Rationale**: Random splits cause data leakage in time-series forecasting

---

## 🔄 End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW DIAGRAM                                   │
└─────────────────────────────────────────────────────────────────────────────┘

1. DATA INGESTION
   ┌─────────────────┐         ┌─────────────────┐
   │  IEX Website   │         │  Open-Meteo     │
   │  (Playwright)   │         │  Weather API    │
   └────────┬────────┘         └────────┬────────┘
            │                        │
            ▼                        ▼
   ┌─────────────────────────────────────────────┐
   │  data/raw/ (training/ & holdout/)          │
   └────────────────────┬────────────────────────┘

2. FEATURE ENGINEERING (src/preprocess.py)
   ┌─────────────────────────────────────────────┐
   │  • Time features: hour, day, month, season  │
   │  • Lag features: 1-day, 7-day (per block)  │
   │  • Rolling stats: 7d & 30d mean/std        │
   │  • Weather: Delhi & Mumbai temperatures   │
   │  • Calendar: holidays, peak hours          │
   └────────────────────┬────────────────────────┘

3. MODEL TRAINING (src/models/*.py)
   ┌─────────────────────────────────────────────┐
   │  • XGBoost / LightGBM / Random Forest       │
   │  • Ridge / Naive / LSTM                     │
   │  • Per-model metrics by season/hour        │
   └────────────────────┬────────────────────────┘

4. PREDICTION & EVALUATION (src/predict.py)
   ┌─────────────────────────────────────────────┐
   │  • 96-block forecast for any date           │
   │  • Regression metrics: R², RMSE, WAPE      │
   │  • Classification: AUC-ROC, F1 (direction) │
   └────────────────────┬────────────────────────┘

5. POWER BI EXPORT (src/powerbi_exporter.py)
   ┌─────────────────────────────────────────────┐
   │  powerbi_data/ (17 CSV files)              │
   └────────────────────┬────────────────────────┘
```

---

## 📈 Data

### Source

- **Exchange**: Indian Energy Exchange (IEX)
- **Market**: Day-Ahead Market (DAM)
- **URL**: https://www.iexindia.com/market-data/day-ahead-market/market-snapshot

### Granularity

- **Time blocks**: 96 per day (15-minute intervals)
- **Example blocks**: 00:00-00:15, 00:15-00:30, ... 23:45-00:00

### Features Used

| Category | Features | Count |
|----------|----------|-------|
| **Calendar/Time** | block, hour, day-of-week, day-of-year, month, year, weekend, holiday, season, hour-bucket | 10 |
| **Price Lags** | 1-day lag, 7-day lag (per block) | 2 |
| **Rolling Stats** | 7-day & 30-day mean/std (per block, with 1-day shift) | 4 |
| **Weather** | Delhi apparent temperature, Mumbai apparent temperature | 2 |
| **Total** | | **18** |

### Key Insight: Time Features > Weather

> Weather adds only **+0.3%** to R² improvement. Time-based features (hour, day-of-week, season) explain the majority of price variance because IEX is a short-term market where temporal patterns dominate.

---

## 🖥️ Dashboard

### Streamlit Dashboard (app.py)

Launch with: `streamlit run app.py`

| Page | Features |
|------|----------|
| **Live Tracker** | Historical MCP prices, volatility indicators, 7-day trends |
| **Forecast Sandbox** | Generate predictions, compare models, block inspector |
| **Model Scorecard** | Side-by-side model comparison, feature importance, weather impact |
| **Data Management** | Fetch data, refresh features, train models, run benchmarks |

### Power BI Integration

The pipeline auto-exports 17 CSV files to `powerbi_data/` for Power BI dashboards:

```
powerbi_data/
├── predictions.csv              # All model predictions
├── model_metrics.csv            # Performance metrics
├── daily_prices.csv             # Historical MCP
├── feature_importance.csv       # Model drivers
├── weather_impact_comparison.csv
├── monthly_drift.csv
└── forecast_YYYY-MM-DD_*.csv    # Daily forecasts
```

See [POWERBI_SETUP.md](POWERBI_SETUP.md) for step-by-step Power BI integration guide.

---

## 🚀 How to Run

### 1. Installation

```bash
# Clone repository
git clone https://github.com/uttampaliwal/power-price-predictor.git
cd power-price-predictor

# Install dependencies
pip install -r requirements.txt

# Install browser for web scraping
playwright install chromium
```

### 2. Data Collection

```bash
# Fetch 5 years of historical prices (Jan 2020 - Dec 2024)
python src/fetch_data.py --start 2020-01-01 --end 2024-12-31 --split training

# Fetch holdout data (Jan 2025 - Aug 2026)
python src/fetch_data.py --start 2025-01-01 --end 2026-08-31 --split holdout

# Fetch weather data
python src/fetch_weather.py --start 2020-01-01 --end 2026-08-31
```

### 3. Preprocessing & Training

```bash
# Generate feature parquet files
python src/preprocess.py --split training
python src/preprocess.py --split holdout

# Train all models
python src/models/xgboost_model.py
python src/models/lightgbm_model.py
python src/models/random_forest_model.py
python src/models/ridge_model.py
python src/models/naive_model.py
```

### 4. Make Predictions

```bash
# Predict for a specific date
python src/predict.py --model xgboost --date 2026-04-15

# Compare multiple models
python src/predict.py --model lightgbm --date 2026-04-15
```

### 5. Launch Dashboard

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
power-price-predictor/
├── src/
│   ├── config.py                 # Directory paths & constants
│   ├── fetch_data.py             # IEX web scraper
│   ├── fetch_weather.py           # Open-Meteo API scraper
│   ├── preprocess.py             # Feature engineering
│   ├── predict.py                # 96-block predictions
│   ├── evaluate.py               # Metrics computation
│   ├── benchmark.py              # Model comparison
│   ├── backtest.py               # Backtesting & P&L simulation
│   ├── api.py                    # FastAPI REST endpoint
│   ├── tracking.py               # MLflow experiment tracking
│   ├── visualize_pipeline.py     # HTML report generator
│   ├── powerbi_exporter.py       # CSV export for Power BI
│   └── models/
│       ├── naive_model.py
│       ├── ridge_model.py
│       ├── random_forest_model.py
│       ├── xgboost_model.py
│       ├── lightgbm_model.py
│       └── lstm_model.py
├── data/                        # Raw & processed data (committed)
│   ├── raw/training/            # DAM prices 2020-2024
│   ├── raw/holdout/             # DAM prices 2025-2026
│   └── processed/               # Feature parquets
├── models/                      # Trained models (committed, RF .pkl excluded)
├── predictions/                 # Model predictions + backtest results
├── powerbi_data/                # Gitignored (exported CSVs)
├── app.py                       # Streamlit dashboard
├── requirements.txt
├── README.md
├── OPERATIONS_GUIDE.md
├── MODEL_EXPLANATION.md
└── blog_post.md
```

---

## 🔌 Key Technical Decisions

### Why Time-Based Split?
Random train/test splits cause **data leakage** in time-series. Using chronological splits (2020-2024 training, 2025+ holdout) simulates real-world deployment where you predict future prices.

### Why WAPE Instead of MAPE?
MAPE becomes unstable when prices hit caps (₹10,000/MWh). **WAPE (Weighted Absolute Percentage Error)** normalizes by total actual values, making it more robust for energy markets.

### Why Weather Features Optional?
Weather adds only +0.3% R² improvement. Time features (hour, day-of-week, season) are far more important because IEX is a short-term market. Models are trained with and without weather for comparison.

---

## 📌 Future Improvements

- [ ] **LSTM/Transformer models** for sequence modeling
- [ ] **Real-time API integration** with IEX
- [ ] **Automated retraining pipeline** (CI/CD)
- [ ] **Weather derivatives** incorporation
- [ ] **Anomaly detection** for price spikes

---

## 📈 Backtesting

Three trading strategies were backtested on 20 months of holdout data (Jan 2025 – Aug 2026):

| Strategy | Total P&L | Sharpe | Win Rate | Capacity |
|----------|-----------|--------|----------|----------|
| Block-Shift Arbitrage | ₹404M | 56.7 | 100% | 100 MW |
| Directional Trading | ₹1,277M | 7.6 | 59.6% | 100 MW |
| Peak Shaving | ₹216M | 63.0 | 100% | 50 MW |

**⚠️ Important Disclaimers:**
- Results are **simulated**, not live trading performance
- No transaction costs, exchange fees, or slippage included
- Fixed capacity (100 MW / 50 MW) with no market-price impact modeling
- Perfect execution assumed at actual MCP (no bid-ask spread)
- High Sharpe ratios (56.7, 63.0) arise from idealized assumptions
- Past performance does not guarantee future results

Run backtesting:
```bash
python src/backtest.py
```

---

## 🔌 API

FastAPI endpoint for real-time predictions:

```bash
# Start server
uvicorn src.api:app --port 8000

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_name": "xgboost", "target_date": "2026-04-15"}'

# Compare models
curl -X POST http://localhost:8000/compare \
  -d '{"model_names": ["xgboost", "lightgbm"], "target_date": "2026-04-15"}'
```

---

## 📊 MLflow Tracking

```bash
# Log all model experiments
python src/tracking.py --log-all

# View MLflow UI
mlflow ui --port 5000
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file — project overview & quick start |
| [OPERATIONS_GUIDE.md](OPERATIONS_GUIDE.md) | Step-by-step daily operations |
| [MODEL_EXPLANATION.md](MODEL_EXPLANATION.md) | Model architecture & feature details |
| [POWERBI_SETUP.md](POWERBI_SETUP.md) | Power BI dashboard setup |
| [POWERBI_DASHBOARD_BUILD_GUIDE.md](POWERBI_DASHBOARD_BUILD_GUIDE.md) | Advanced Power BI tips |
| [PROJECT_LEARNING_SUMMARY.md](PROJECT_LEARNING_SUMMARY.md) | Key learnings & insights |
| [blog_post.md](blog_post.md) | Blog post draft for Medium |

---

## 🙏 Credits

- **Data Source**: [Indian Energy Exchange (IEX)](https://www.iexindia.com)
- **Weather Data**: [Open-Meteo API](https://open-meteo.com)
- **Built with**: Python, scikit-learn, XGBoost, LightGBM, Streamlit, Plotly, PyTorch

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

*Built with ❤️ for the Indian power market*