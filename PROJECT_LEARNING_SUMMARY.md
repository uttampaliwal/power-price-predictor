# IEX Power Price Prediction - Project Learning Summary

## 1. WHAT WAS DONE

### Project Overview
Built an end-to-end machine learning system to predict Indian Energy Exchange (IEX) Day-Ahead Market (DAM) electricity prices with 15-minute granularity using 5 ML models.

### Key Components Built

#### A. Data Pipeline
- **Data Collection**: Web scraping IEX website for historical DAM prices (2020-2026)
- **Weather Integration**: Fetched temperature/humidity data from Open-Meteo API for Delhi & Mumbai
- **Feature Engineering**: Created time-based features (hour, day, month, season), demand-supply metrics, holiday flags
- **Data Split**: Time-based 70-30 split (Training: 2020-2024, Holdout: 2025-2026)

#### B. Models Implemented & Compared
| Model | Type | R² | Best Use Case |
|-------|------|-----|-------------|
| XGBoost | Gradient Boosting | **0.867** | Price forecasting |
| LightGBM | Gradient Boosting | 0.867 | Price forecasting |
| Random Forest | Bagging | 0.862 | Robust predictions |
| Ridge | Linear | 0.817 | Direction trading |
| Naive | Baseline | 0.555 | Comparison |

#### C. Evaluation Framework
- **Regression Metrics**: R², RMSE, MAE, MAPE, WAPE
- **Classification Metrics**: AUC-ROC, F1 (for price direction prediction ≥1%)
- **Benchmarking**: Automated comparison across all models

#### D. Deployment & Visualization
- **Streamlit Dashboard**: Live tracker, forecast sandbox, model scorecard, data management
- **Power BI Integration**: Auto-exported CSVs for business dashboards
- **Automated Pipelines**: One-click data fetch → preprocess → train → export

---

## 2. DATA FLOW IN DASHBOARD

### Complete Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA FLOW DIAGRAM                        │
└─────────────────────────────────────────────────────────────────────┘

1. RAW DATA COLLECTION
   ┌─────────────────┐         ┌─────────────────┐
   │  IEX Website   │         │  Weather APIs  │
   │  (Web Scrape)  │         │  (Open-Meteo) │
   └────────┬────────┘         └────────┬────────┘
            │                        │
            ▼                        ▼
   ┌────────────────────────────────────────────┐
   │   data/raw/ (training/ & holdout/)      │
   │   dam_2020.csv, dam_2021.csv, ...        │
   │   weather_training.csv, weather_holdout.csv  │
   └────────────────────┬───────────────────────┘
                            │
                            ▼
2. PREPROCESSING (preprocess.py)
   ┌────────────────────────────────────────────┐
   │  • Feature engineering (time, demand-supply)   │
   │  • Weather integration                      │
   │  • Handle missing values                   │
   │  • Normalize/standardize                   │
   └────────────────────┬───────────────────────┘
                            │
                            ▼
   ┌────────────────────────────────────────────┐
   │   data/processed/                          │
   │   training_features.parquet                │
   │   holdout_features.parquet                 │
   └────────────────────┬───────────────────────┘
                            │
                            ▼
3. MODEL TRAINING (predict.py)
   ┌────────────────────────────────────────────┐
   │  • Train 5 models (XGBoost, LightGBM,     │
   │    Random Forest, Ridge, Naive)           │
   │  • Hyperparameter tuning                   │
   │  • Save models to models/ directory       │
   └────────────────────┬───────────────────────┘
                            │
                            ▼
   ┌────────────────────────────────────────────┐
   │   models/ (or models_no_weather/)          │
   │   xgboost/metrics.csv                  │
   │   lightgbm/feature_importance.csv       │
   └────────────────────┬───────────────────────┘
                            │
                            ▼
4. PREDICTION & EVALUATION (evaluate.py)
   ┌────────────────────────────────────────────┐
   │  • Generate predictions on holdout set      │
   │  • Calculate metrics (R², RMSE, AUC, F1)  │
   │  • Save to predictions/ directory          │
   └────────────────────┬───────────────────────┘
                            │
                            ▼
   ┌────────────────────────────────────────────┐
   │   predictions/                               │
   │   xgboost/test_predictions.csv              │
   │   benchmark_metrics.csv                    │
   └────────────────────┬───────────────────────┘
                            │
                            ▼
5. EXPORT TO POWER BI (powerbi_exporter.py)
   ┌────────────────────────────────────────────┐
   │   powerbi_data/                              │
   │   predictions.csv (all models)                 │
   │   model_metrics.csv                        │
   │   daily_prices.csv                        │
   │   feature_importance.csv                  │
   └────────────────────┬───────────────────────┘
                            │
                            ▼
6. STREAMLIT DASHBOARD (app.py)
   ┌────────────────────────────────────────────┐
   │  • Live Tracker: View historical MCP         │
   │  • Forecast Sandbox: Generate new forecasts  │
   │  • Model Scorecard: Compare models         │
   │  • Data Management: Fetch, train, export   │
   └────────────────────────────────────────────┘
                            │
                            ▼
7. POWER BI DASHBOARD
   ┌────────────────────────────────────────────┐
   │  • Connect to powerbi_data/ folder        │
   │  • Build reports & visualizations        │
   │  • Real-time monitoring               │
   └────────────────────────────────────────────┘
```

### Dashboard Pages & Data Sources

| Page | Data Source | Purpose |
|------|-------------|---------|
| **Live Tracker** | `holdout_features.parquet` | View historical prices, volatility |
| **Forecast Sandbox** | `predictions/*/daily/*.csv` | Generate & compare predictions |
| **Model Scorecard** | `models/*/metrics.csv` | Compare R², WAPE, AUC, F1 |
| **Data Management** | `data/raw/*/dam_*.csv` | Fetch new data, retrain |

---

## 3. KEY LEARNINGS

### A. Technical Learnings

#### 1. **Time-Based Features Matter Most**
- **Finding**: Weather adds only 0.1% improvement to R²
- **Learning**: For electricity pricing, temporal patterns (hour, day-of-week, season) are stronger predictors than weather
- **Action**: Focus feature engineering on time-based patterns

#### 2. **Gradient Boosting Works Best for This Problem**
- **Finding**: XGBoost & LightGBM achieve R² = 0.867 (top performance)
- **Learning**: Non-linear models with tree-based ensembles handle electricity price volatility better than linear models
- **Why**: Can capture complex interactions between time, demand, and supply

#### 3. **Evaluation Requires Both Regression & Classification**
- **Regression (R², WAPE)**: Tells us prediction accuracy (R² = 0.867 = 86.7% variance explained)
- **Classification (AUC, F1)**: Tells us directional accuracy (F1 = 0.698 = reliable for trading)
- **Learning**: Different stakeholders need different metrics:
  - Operations team → R², WAPE (exact price)
  - Trading team → F1, AUC (price direction)

#### 4. **Proper Train-Test Split is Critical**
- **Approach**: Time-based split (not random) - Training: 2020-2024, Holdout: 2025-2026
- **Learning**: Never shuffle time-series data - it causes data leakage
- **Result**: True generalization performance (R² = 0.867 on unseen 2025-2026 data)

#### 5. **WAPE is Better Than MAPE for This Domain**
- **Finding**: When prices hit caps (₹10,000), MAPE becomes unstable
- **Learning**: WAPE (Weighted Absolute Percentage Error) is more robust for electricity prices
- **Why**: Normalizes by total actual values, not per-sample

### B. Business Learnings

#### 6. **56% Improvement Over Baseline is Significant**
- Naive baseline: R² = 0.555
- XGBoost: R² = 0.867
- **Business Impact**: This improvement translates to better bidding strategies, cost savings, and profit optimization

#### 7. **13.8% WAPE is Operationally Acceptable**
- Average prediction error of 13.8% is within industry standards
- Electricity prices are inherently volatile (demand spikes, supply shortages)
- **Learning**: Perfect predictions are impossible; focus on actionable accuracy

#### 8. **Model Selection Depends on Use Case**
- **For Price Forecasting** → Use XGBoost (best R², lowest WAPE)
- **For Trading (Direction)** → Use Ridge (best AUC-ROC = 0.792, F1 = 0.698)
- **Learning**: No single "best" model - match model to business objective

### C. Engineering Learnings

#### 9. **Modular Architecture Enables Rapid Iteration**
- Separate scripts for: fetch, preprocess, train, evaluate, export
- **Learning**: Clean separation of concerns makes debugging & feature addition easier
- **Example**: Adding weather features required changes in only 2 files

#### 10. **Automation is Essential for Retraining**
- Built one-click pipelines: "Fetch → Refresh → Export to Power BI"
- **Learning**: Manual retraining doesn't scale; automate data pipelines
- **Future**: Add CI/CD for auto-retraining when new data arrives

#### 11. **Power BI Integration Requires Flat CSVs**
- Exported from parquet (processing) to CSV (Power BI)
- **Learning**: Different tools need different formats; build exporters early
- **Current**: 17 CSV files in `powerbi_data/` for dashboard

### D. Surprising Discoveries

#### 12. **Weather Impact is Minimal (+0.1% R²)**
- **Expected**: Weather would significantly impact prices
- **Actual**: Time features (hour, season) dominate
- **Reason**: IEX is a short-term market; weather affects long-term demand, not 15-minute prices
- **Pivot**: Focused on demand-supply metrics instead of weather models

#### 13. **Price Caps Create Prediction Challenges**
- IEX has ₹10,000/MWh price cap
- When demand surges, prices hit cap → prediction errors increase
- **Learning**: Add binary "cap indicator" feature in future versions

---

## SUMMARY: KEY TAKEAWAYS FOR PRESENTATION

### 1-Minute Elevator Pitch
> "We built an ML system that predicts IEX electricity prices with 87% accuracy (R²=0.867), 
> a 56% improvement over baseline. XGBoost emerged as the best model for price forecasting, 
> while Ridge excels at predicting price direction for trading. Surprisingly, weather adds 
> minimal value - time-based features matter most."

### Top 3 Learnings for Q&A
1. **Time features > Weather features** for short-term electricity price prediction
2. **XGBoost/LightGBM** best for this non-linear, volatile domain
3. **Match model to objective**: R² for forecasting, F1/AUC for trading

### Future Improvements
- Add LSTM/Transformer for sequence modeling
- Real-time API integration with IEX
- Automated retraining pipelines
