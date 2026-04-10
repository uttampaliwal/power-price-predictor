# Power Price Predictor

A machine learning pipeline for forecasting Indian electricity Day-Ahead Market (DAM) prices from **IEX India** at 15-minute granularity (96 blocks/day).

## Project Structure

```
power_price/
├── data/
│   ├── raw/training/      ← Monthly CSVs: Jan 2020 – Jan 2025
│   └── raw/holdout/       ← Monthly CSVs: 2025 onwards (untouched during training)
├── data/processed/        ← Feature-engineered Parquet files
├── models/<model_name>/   ← Saved model files + per-model metrics
├── predictions/<model_name>/ ← Per-model predictions CSVs + charts
├── src/
│   ├── fetch_data.py      ← IEX scraper (Playwright)
│   ├── preprocess.py      ← Feature engineering
│   ├── evaluate.py        ← Metrics (RMSE, MAE, MAPE, R², AUC-ROC, F1…)
│   ├── predict.py         ← Daily prediction for any date
│   ├── benchmark.py       ← Compare all models on holdout + drift plot
│   └── models/
│       ├── naive_model.py
│       ├── ridge_model.py
│       ├── random_forest_model.py
│       ├── xgboost_model.py
│       ├── lightgbm_model.py
│       ├── lstm_model.py
│       ├── prophet_model.py   ← 🔲 Stub — run when ready
│       └── tft_model.py       ← 🔲 Stub — run when ready
├── requirements.txt
├── README.md
├── MODEL_EXPLANATION.md
└── OPERATIONS_GUIDE.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
playwright install chromium

# 2. Fetch training data (Jan 2020 – Jan 2025)
python src/fetch_data.py --start 2020-01-01 --end 2025-01-31 --split training

# 3. Fetch holdout data (2025 onwards) — kept raw and separate
python src/fetch_data.py --start 2025-01-01 --end 2026-04-08 --split holdout

# 3b. (Optional) Force re-fetch existing data using --overwrite
# python src/fetch_data.py --start 2025-01-01 --end 2026-04-08 --split holdout --overwrite

# 4. Preprocess both splits
python src/preprocess.py --split training
python src/preprocess.py --split holdout

# 5. Train each model (runs independently)
python src/models/naive_model.py
python src/models/ridge_model.py
python src/models/random_forest_model.py
python src/models/xgboost_model.py
python src/models/lightgbm_model.py
python src/models/lstm_model.py

# 6. Benchmark all models on holdout + generate drift plots
python src/benchmark.py

# 7. Predict next-day prices
python src/predict.py --model xgboost
python src/predict.py --model lightgbm --date 2026-04-01
```

## Models

| Model | Family | Status |
|---|---|---|
| Naive (Last-Week) | Baseline | ✅ Active |
| Ridge Regression | Linear ML | ✅ Active |
| Random Forest | Ensemble ML | ✅ Active |
| XGBoost | Boosting | ✅ Active |
| LightGBM | Boosting | ✅ Active |
| LSTM | Deep Learning | ✅ Active |
| Prophet | Statistical | 🔲 Stub |
| TFT | Deep Learning | 🔲 Stub |

## Data Source

**IEX India — Day-Ahead Market**  
URL: `https://www.iexindia.com/market-data/day-ahead-market/market-snapshot`  
Interval: 15-minute blocks (96 per day)  
Data: Purchase Bid (MW), Sell Bid (MW), MCV (MW), MCP (Rs/MWh)

## Metrics

- **Regression**: RMSE, MAE, MAPE, R², WAPE
- **Classification** (price direction): AUC-ROC, F1, Precision, Recall, Accuracy
- All metrics segmented by: season, hour-bucket, month

See [MODEL_EXPLANATION.md](MODEL_EXPLANATION.md) and [OPERATIONS_GUIDE.md](OPERATIONS_GUIDE.md) for details.
