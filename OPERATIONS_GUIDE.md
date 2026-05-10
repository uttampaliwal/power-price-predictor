# Power Price Prediction: End-to-End Command Guide

This guide provides the exact sequential commands required to completely run the pipeline from bare metal to a final forecasted price prediction on a daily basis.

## 1. Data Acquisition
Extract 96-block continuous data from the IEX Day-Ahead Market.

**Initial Training Set Generation** (For 5 years historical memory)
```bash
python src/fetch_data.py --start 2020-01-01 --end 2024-12-31 --split training
```

**Live / Holdout Set Maintenance** (Update this daily to keep the model updated)
```bash
python src/fetch_data.py --start 2025-01-01 --end 2026-03-23 --split holdout
```

## 2. Preprocessing & Feature Engineering
Convert the raw CSVs into highly optimized features (lags, holiday constraints, and volatility indices).

```bash
python src/preprocess.py --split training
python src/preprocess.py --split holdout
```

## 3. Model Training
Fit the models on the 2020-2024 training features. *Note: Training scripts also automatically evaluate using the holdout features internally.*

```bash
# Baseline
python src/models/naive_model.py

# Linear / Tabular Machine Learning
python src/models/ridge_model.py
python src/models/random_forest_model.py
python src/models/xgboost_model.py
python src/models/lightgbm_model.py
```

## 4. Benchmarking & Visualization
Review the relative accuracy and visualize your exact pipeline predictions.

```bash
python src/benchmark.py
python src/visualize_pipeline.py
```
*(This produces comparison charts in the `predictions/` folder and a standalone interactive HTML dashboard `pipeline_report.html` in the root.)*

## 5. Inference / Daily Predictions
Command to predict precisely 96 time-blocks for any given date.

```bash
python src/predict.py --model xgboost --date 2026-03-24
python src/predict.py --model lightgbm --date 2026-03-24
```

> **Pro-Tip on Extracting Specific Times:** 
> The `predict.py` script automatically processes the entire 24-hour cycle for *all* models. If you specifically want to see the price for `12:30 - 12:45`, simply open the generated CSV (`predictions/xgboost/daily/prediction_2026-03-24.csv`) and look at row 51 (the `12:30` time block)!
