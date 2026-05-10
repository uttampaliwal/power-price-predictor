# IEX Power Price Predictor

A production-ready machine learning pipeline for forecasting Indian electricity Day-Ahead Market (DAM) prices from **IEX India**. Predicts 96 time-blocks (15-minute intervals) across a full day with ~87% accuracy using XGBoost/LightGBM ensemble models.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Data Pipeline](#data-pipeline)
- [Models](#models)
- [Features](#features)
- [Evaluation Metrics](#evaluation-metrics)
- [Dashboard](#dashboard)
- [Power BI Integration](#power-bi-integration)
- [Results Summary](#results-summary)
- [Key Learnings](#key-learnings)
- [Documentation](#documentation)

---

## Overview

### What This Project Does

Predicts electricity prices (Market Clearing Price - MCP) for the IEX Day-Ahead Market at 15-minute granularity (96 blocks per day) to help energy traders and operators make better bidding decisions.

### Key Metrics

| Metric | Value | Model |
|--------|-------|-------|
| R² Score | **0.867** | XGBoost |
| WAPE | **13.8%** | LightGBM |
| AUC-ROC | **0.79** | Ridge (for direction) |
| F1 Score | **0.70** | Ridge (for direction) |

### Improvement Over Baseline

- **Naive Baseline R²**: 0.555
- **XGBoost R²**: 0.867
- **Improvement**: 56% better than baseline

---

## Project Structure

```
power_price/
├── src/
│   ├── config.py                 # All directory paths, constants, model list
│   ├── fetch_data.py             # Web scraper for IEX DAM prices
│   ├── fetch_weather.py          # Open-Meteo API scraper for weather data
│   ├── preprocess.py             # Feature engineering pipeline
│   ├── predict.py                # 96-block prediction for any date
│   ├── evaluate.py               # Metrics computation (RMSE, MAE, MAPE, R², AUC, F1)
│   ├── benchmark.py              # Compare all models on holdout
│   ├── visualize_pipeline.py     # Generate HTML report with charts
│   ├── powerbi_exporter.py       # Export data to CSV for Power BI
│   ├── powerbi_live_sync.py      # Auto-sync live predictions
│   └── models/
│       ├── naive_model.py        # Baseline (last-week same block)
│       ├── ridge_model.py        # Linear ML baseline
│       ├── random_forest_model.py
│       ├── xgboost_model.py
│       ├── lightgbm_model.py
│       ├── lstm_model.py         # Deep learning (PyTorch)
│       ├── prophet_model.py      # Stub (Facebook Prophet)
│       └── tft_model.py          # Stub (Temporal Fusion Transformer)
├── data/
│   ├── raw/
│   │   ├── training/             # 2020-01-01 to 2024-12-31 CSVs
│   │   ├── holdout/             # 2025-01-01 onwards CSVs (untouched)
│   │   ├── weather_training.csv
│   │   └── weather_holdout.csv
│   └── processed/
│       ├── training_features.parquet
│       └── holdout_features.parquet
├── models/                       # Models trained WITH weather features
│   ├── xgboost/
│   ├── lightgbm/
│   ├── random_forest/
│   └── ridge/
├── models_no_weather/            # Models trained WITHOUT weather features
├── predictions/                   # Test predictions for each model
├── powerbi_data/                 # CSV exports for Power BI dashboards
├── .streamlit/config.toml        # Streamlit theme config
├── app.py                        # Streamlit dashboard
├── requirements.txt
├── README.md
├── OPERATIONS_GUIDE.md
├── MODEL_EXPLANATION.md
└── WEATHER_IMPACT_ANALYSIS.md
```

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
cd power_price

# Install dependencies
pip install -r requirements.txt

# Install Playwright browser (required for web scraping)
playwright install chromium
```

### 2. Data Collection

```bash
# Fetch training data (Jan 2020 - Dec 2024)
python src/fetch_data.py --start 2020-01-01 --end 2024-12-31 --split training

# Fetch holdout data (2025 onwards - kept separate for fair evaluation)
python src/fetch_data.py --start 2025-01-01 --end 2026-04-08 --split holdout

# Fetch weather data (Delhi & Mumbai temperatures)
python src/fetch_weather.py --start 2020-01-01 --end 2026-04-08 --split training
python src/fetch_weather.py --start 2025-01-01 --end 2026-04-08 --split holdout
```

### 3. Preprocessing

```bash
# Generate feature parquet files
python src/preprocess.py --split training
python src/preprocess.py --split holdout
```

### 4. Model Training

```bash
# Train all models (each script is independent)
python src/models/naive_model.py
python src/models/ridge_model.py
python src/models/random_forest_model.py
python src/models/xgboost_model.py
python src/models/lightgbm_model.py
```

### 5. Benchmarking

```bash
# Compare all models and generate visualizations
python src/benchmark.py

# Generate interactive HTML report
python src/visualize_pipeline.py
```

### 6. Making Predictions

```bash
# Predict for a specific date using XGBoost
python src/predict.py --model xgboost --date 2026-04-15

# Compare with LightGBM
python src/predict.py --model lightgbm --date 2026-04-15
```

### 7. Launch Dashboard

```bash
streamlit run app.py
```

---

## Data Pipeline

### Data Source

**IEX India - Day-Ahead Market**  
URL: `https://www.iexindia.com/market-data/day-ahead-market/market-snapshot`

### Data Fields

| Field | Description |
|-------|-------------|
| `date` | Date of the time block |
| `block` | Block number (1-96, 15-min intervals) |
| `time_block` | Human-readable time (HH:MM-HH:MM) |
| `purchase_bid_mw` | Purchase bid in MW |
| `sell_bid_mw` | Sell bid in MW |
| `mcv_mw` | Market Clearing Volume in MW |
| `mcp_rs_per_mwh` | Market Clearing Price (Rs/MWh) - **TARGET** |

### Train/Holdout Split

- **Training**: January 2020 - December 2024 (5 years)
- **Holdout**: January 2025 - April 2026 (unseen data for evaluation)
- **Split Method**: Time-based (not random) to avoid data leakage

### Weather Data

- **Source**: Open-Meteo API (free, no API key required)
- **Cities**: Delhi (28.61°N, 77.21°E) and Mumbai (19.08°N, 72.88°E)
- **Features**: Actual temperature, Apparent (feels-like) temperature

---

## Models

### Active Models

| Model | Type | R² | WAPE | Best Use Case |
|-------|------|-----|------|---------------|
| **XGBoost** | Gradient Boosting | **0.867** | 13.8% | Price forecasting |
| **LightGBM** | Gradient Boosting | 0.867 | 13.8% | Price forecasting |
| **Random Forest** | Bagging | 0.862 | 14.0% | Robust predictions |
| **Ridge** | Linear | 0.817 | 16.4% | Direction trading |
| **Naive** | Baseline | 0.555 | 23.4% | Baseline comparison |
| **LSTM** | Deep Learning | - | - | Sequence modeling |

### Stub Models (Ready to Implement)

| Model | Type | Status |
|-------|------|--------|
| Prophet | Statistical | 🔲 Stub |
| TFT | Deep Learning | 🔲 Stub |

### Weather vs No-Weather Analysis

Weather features (temperature) add only **0.1%** improvement to R². Time-based features (hour, day-of-week, season) are far more important for short-term electricity pricing.

| Configuration | R² | WAPE |
|--------------|-----|------|
| With Weather | 0.867 | 13.8% |
| Without Weather | 0.866 | 13.9% |

---

## Features

### Time-Based Features (Most Important)

| Feature | Description |
|---------|-------------|
| `hour` | Hour of day (0-23) |
| `block` | 15-minute block number (1-96) |
| `day_of_week` | Day index (0=Monday, 6=Sunday) |
| `is_weekend` | Binary flag for Saturday/Sunday |
| `month` | Month (1-12) |
| `season` | Season category (Winter, Summer, Monsoon, Autumn) |
| `quarter` | Quarter of year (Q1-Q4) |
| `day_of_year` | Day index (1-365) |

### Demand-Supply Features

| Feature | Description |
|---------|-------------|
| `purchase_bid_mw` | Total purchase bid |
| `sell_bid_mw` | Total sell bid |
| `demand_supply_ratio` | Purchase/Sell ratio |
| `net_demand` | Purchase - Sell |

### Lag Features

| Feature | Description |
|---------|-------------|
| `mcp_lag_1` | MCP from previous block |
| `mcp_lag_4` | MCP from 1 hour ago |
| `mcp_lag_96` | MCP from yesterday same block |
| `mcp_lag_672` | MCP from same block 1 week ago |
| `mcp_rolling_mean_4` | Rolling average of last 4 blocks |

### Weather Features

| Feature | Description |
|---------|-------------|
| `delhi_weather` | Delhi temperature |
| `mumbai_weather` | Mumbai temperature |
| `delhi_apparent_temp` | Delhi feels-like temperature |
| `mumbai_apparent_temp` | Mumbai feels-like temperature |

### Calendar Features

| Feature | Description |
|---------|-------------|
| `is_holiday` | Indian holiday flag |
| `holiday_name` | Name of holiday (if any) |
| `ispeak_hour` | Peak demand hour (17-22) |

---

## Evaluation Metrics

### Regression Metrics

| Metric | Description | Best Value |
|--------|-------------|------------|
| **RMSE** | Root Mean Squared Error | Lower is better |
| **MAE** | Mean Absolute Error | Lower is better |
| **MAPE** | Mean Absolute Percentage Error | Lower is better |
| **WAPE** | Weighted Absolute Percentage Error | Lower is better |
| **R²** | Coefficient of Determination | Higher is better (0-1) |

### Classification Metrics (Price Direction)

| Metric | Description | Best Value |
|--------|-------------|------------|
| **AUC-ROC** | Area Under ROC Curve | Higher is better (0-1) |
| **F1 Score** | Harmonic mean of Precision/Recall | Higher is better (0-1) |
| **Precision** | True Positive / Predicted Positive | Higher is better |
| **Recall** | True Positive / Actual Positive | Higher is better |
| **Accuracy** | Correct predictions / Total | Higher is better |

### Segmented Analysis

All metrics are computed for:
- **By Season**: Winter, Summer, Monsoon, Autumn
- **By Hour Bucket**: Morning (7-10), Day (11-17), Evening (18-23), Night (0-6)
- **By Month**: January-December

---

## Dashboard

### Streamlit Dashboard (app.py)

Launch with: `streamlit run app.py`

#### Page 1: Live Tracker

- Browse historical MCP prices by date
- View 96-block price curve for any day
- Compare with previous day
- Volatility indicators
- 7-day average price trend

#### Page 2: Forecast Sandbox

- Generate predictions for any future date
- Compare multiple models simultaneously
- View confidence bands (±10%)
- Block-by-block price inspector
- Download predictions as CSV
- IEX regulatory summary tables

#### Page 3: Model Scorecard

- Compare all models side-by-side
- Filter by weather configuration
- Weather impact analysis
- Feature importance visualization
- R² and WAPE bar charts

#### Page 4: Data Management

- View data status and coverage
- Fetch new data from IEX
- Refresh feature engineering
- Train models
- Run benchmarks

### Pipeline Report (pipeline_report.html)

Standalone interactive HTML report generated by `visualize_pipeline.py`:
- 7-day sample price chart
- Model metrics comparison table
- Graphical performance summary
- Confusion matrices for direction prediction

---

## Power BI Integration

### Export Files

The `powerbi_exporter.py` script exports the following CSVs:

| File | Description |
|------|-------------|
| `predictions.csv` | All model predictions for all dates |
| `model_metrics.csv` | Performance metrics for each model |
| `daily_prices.csv` | Historical MCP prices |
| `feature_importance.csv` | Feature importance per model |
| `weather_impact_comparison.csv` | Weather vs no-weather analysis |
| `monthly_drift.csv` | Model drift over time |
| `forecast_YYYY-MM-DD_*.csv` | Daily predictions by model |

### Setup Guide

See [POWERBI_SETUP.md](POWERBI_SETUP.md) for detailed Power BI dashboard setup instructions.

---

## Results Summary

### Best Model Performance (XGBoost with Weather)

```
Holdout Period: January 2025 - April 2026
Training Period: January 2020 - December 2024

RMSE:   966.38 Rs/MWh
MAE:    557.80 Rs/MWh
MAPE:   29.45%
WAPE:   13.80%
R²:     0.8674

Classification (Price Direction ±1%):
AUC-ROC: 0.779
F1:      0.681
Accuracy: 80.2%
```

### Model Comparison (With Weather)

| Model | RMSE | MAE | MAPE | WAPE | R² | AUC-ROC | F1 |
|-------|------|-----|------|------|-----|---------|-----|
| XGBoost | 966.38 | 557.80 | 29.45 | 13.80 | **0.867** | 0.779 | 0.681 |
| LightGBM | 968.30 | 556.93 | 30.45 | 13.78 | 0.867 | 0.776 | 0.677 |
| Random Forest | 987.39 | 566.18 | 27.37 | 14.01 | 0.862 | 0.742 | 0.629 |
| Ridge | 1135.50 | 663.38 | 29.07 | 16.42 | 0.817 | **0.792** | **0.698** |
| Naive | 1700.61 | 928.51 | 35.86 | 23.41 | 0.555 | 0.734 | 0.623 |

---

## Key Learnings

### Technical Insights

1. **Time features dominate**: Hour, day-of-week, and season are stronger predictors than weather
2. **Gradient boosting wins**: XGBoost/LightGBM capture non-linear price patterns better than linear models
3. **Weather impact is minimal**: Only +0.1% R² improvement from temperature features
4. **WAPE > MAPE**: WAPE is more stable for electricity prices (avoids issues with price caps)
5. **Time-based split is critical**: Random splits cause data leakage in time-series data

### Business Insights

1. **56% improvement over baseline**: XGBoost achieves R²=0.867 vs Naive R²=0.555
2. **13.8% WAPE is operational**: Acceptable for trading decisions
3. **Model selection depends on use case**:
   - Price forecasting → XGBoost (best R²)
   - Direction trading → Ridge (best AUC-ROC, F1)
4. **Price caps create challenges**: ₹10,000/MWh cap causes prediction errors

### Engineering Insights

1. **Modular architecture**: Clean separation enables rapid iteration
2. **Automation essential**: One-click pipelines for daily operations
3. **Power BI needs flat CSVs**: Export from parquet to CSV for compatibility

---

## Documentation

| Document | Description |
|----------|-------------|
| [OPERATIONS_GUIDE.md](OPERATIONS_GUIDE.md) | Step-by-step command guide for daily operations |
| [MODEL_EXPLANATION.md](MODEL_EXPLANATION.md) | Detailed model architecture and feature engineering |
| [WEATHER_IMPACT_ANALYSIS.md](WEATHER_IMPACT_ANALYSIS.md) | Analysis of weather feature importance |
| [POWERBI_SETUP.md](POWERBI_SETUP.md) | Power BI dashboard setup guide |
| [POWERBI_DASHBOARD_BUILD_GUIDE.md](POWERBI_DASHBOARD_BUILD_GUIDE.md) | Advanced Power BI tips |
| [PROJECT_LEARNING_SUMMARY.md](PROJECT_LEARNING_SUMMARY.md) | Complete learning summary and key takeaways |

---

## Dependencies

```
Core Data Processing
├── pandas>=2.0.0
├── numpy>=1.24.0
└── pyarrow>=12.0.0

Machine Learning
├── scikit-learn>=1.3.0
├── xgboost>=2.0.0
├── lightgbm>=4.0.0
└── joblib>=1.3.0

Deep Learning
└── torch>=2.0.0

Visualization
├── matplotlib>=3.7.0
├── seaborn>=0.12.0
└── plotly>=5.18.0

Web Scraping & APIs
├── requests>=2.31.0
├── playwright>=1.40.0
└── urllib3>=2.0.0

Dashboard
├── streamlit>=1.30.0
├── tqdm>=4.65.0
└── lxml>=4.9.0

Utilities
└── holidays>=0.35
```

---

## License & Credits

- **Data Source**: Indian Energy Exchange (IEX) - https://www.iexindia.com
- **Weather Data**: Open-Meteo API - https://open-meteo.com
- **Built with**: Python, scikit-learn, XGBoost, LightGBM, Streamlit, Plotly

---

*For questions or issues, refer to the documentation files or open an issue.*