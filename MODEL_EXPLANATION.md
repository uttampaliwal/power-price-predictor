# Model Explanation — Power Price Predictor

## What Are We Predicting?

We predict the **Market Clearing Price (MCP)** of electricity in India's **Day-Ahead Market (DAM)** — the price at which power is bought and sold for next-day delivery, at 15-minute resolution (96 time blocks per day).

**Unit**: Rs/MWh (Rupees per Megawatt-hour)  
**Typical range**: 1,000 – 8,000 Rs/MWh, with spikes to 12,000 in extreme demand periods.

---

## Why Not Unsupervised or Reinforcement Learning?

| Approach | Works for price prediction? | Why |
|---|---|---|
| **Supervised Learning** | ✅ Yes — best choice | You have labelled historical data (past prices), so standard regression directly applies |
| **Unsupervised Learning** | Partial — as a tool | Useful for clustering demand regimes or anomaly detection, but cannot predict a specific price value |
| **Reinforcement Learning** | ❌ No — wrong task | RL optimizes a *bidding strategy* (when/how much to buy or sell). It needs a market simulator. Not for forecasting |
| **Deep Learning** | ✅ Yes — LSTM, TFT | Sequence models that capture temporal dependencies — excellent complement to boosting models |

---

## Feature Engineering

All models share the same feature set from `preprocess.py`:

### Time Features
| Feature | Description |
|---|---|
| `block` | Time block index 0–95 (0 = 00:00–00:15) |
| `hour` | Hour of the day 0–23 |
| `day_of_week` | 0=Monday, 6=Sunday |
| `month`, `day_of_year`, `year` | Calendar features |
| `is_weekend` | 1 if Saturday or Sunday |
| `is_holiday` | 1 if Indian public holiday |
| `season` | 0=Winter, 1=Summer, 2=Monsoon, 3=Post-monsoon |
| `hour_bucket` | 0=Night, 1=Morning-peak, 2=Afternoon, 3=Pre-evening, 4=Evening-peak |

### Lag Features (most predictive!)
| Feature | Description |
|---|---|
| `mcp_lag_1d` | Same block's price, yesterday |
| `mcp_lag_7d` | Same block's price, last week |

### Rolling Statistics
| Feature | Description |
|---|---|
| `mcp_rolling_7d_mean` | 7-day average for same block |
| `mcp_rolling_7d_std` | 7-day volatility for same block |
| `mcp_rolling_30d_mean` | 30-day trend for same block |
| `mcp_rolling_30d_std` | 30-day volatility |

---

## Models Explained

### 1. Naive Baseline
- **Prediction**: Same block's price from 7 days ago
- **Why include?**: Sets a minimum bar — any real model must beat this
- **Weakness**: Ignores seasonality shifts, holidays, demand spikes

### 2. Ridge Regression
- **How it works**: Linear regression with L2 regularization (prevents overfitting)
- **Strength**: Extremely fast, interpretable
- **Weakness**: Cannot model non-linear relationships (e.g., price spikes)

### 3. Random Forest
- **How it works**: Trains 300 decision trees in parallel, averages their predictions
- **Strength**: Handles non-linearities, robust to outliers, tells you feature importance
- **Weakness**: Slower than boosting models, can miss temporal patterns

### 4. XGBoost (Primary Recommendation)
- **How it works**: Adds trees one at a time, each correcting errors of the previous
- **Strength**: Best overall balance of accuracy, speed, and interpretability for tabular time-series. Dominant in energy price forecasting competitions
- **Uses**: Early stopping on validation, prevents overfitting automatically

### 5. LightGBM
- **How it works**: Same idea as XGBoost but with leaf-wise tree growth — significantly faster
- **Strength**: Usually matches XGBoost in accuracy but trains 5–10× faster
- **Best for**: Large datasets, iterative experimentation

### 6. LSTM (Long Short-Term Memory)
- **How it works**: A deep learning model that reads a full 7-day sequence (7 × 96 = 672 time steps) and outputs next-day predictions
- **Strength**: Captures complex temporal patterns that tabular models miss
- **Weakness**: Slower to train, sensitive to hyperparameters, needs more data to shine
- **Architecture**: 2-layer LSTM → Dense(256) → Dense(96 outputs)

### 7. Prophet (TODO)
- Built by Facebook, designed for human-scale time series with daily/weekly/yearly seasonality
- Easy to use but limited compared to boosting models for 15-min electricity data

### 8. TFT — Temporal Fusion Transformer (TODO)
- State-of-the-art for multi-horizon time-series forecasting
- Uses attention + LSTM + interpretable feature selection
- Significantly more complex to train

---

## Metrics Explained

### Regression Metrics

| Metric | Formula | Good value | Notes |
|---|---|---|---|
| **RMSE** | √(mean(error²)) | Lower = better | Penalizes large spikes heavily |
| **MAE** | mean(|error|) | Lower = better | Average absolute error in Rs/MWh |
| **MAPE** | mean(|error/true|)×100 | <10% = very good | Scale-independent % error |
| **R²** | 1 – SS_residual/SS_total | >0.9 = excellent | Variance explained by model |
| **WAPE** | sum(|error|)/sum(true)×100 | Lower = better | Weighted %, robust to near-zero prices |

### Classification Metrics (Price Direction: Up vs Down)

For each block, we ask: *did the model predict the correct direction of price movement from the previous day?*

| Metric | Good value | Notes |
|---|---|---|
| **AUC-ROC** | >0.75 is good | Ability to rank "up" vs "down" correctly |
| **F1 Score** | >0.70 | Harmonic mean of Precision & Recall |
| **Precision** | Higher = fewer false alarms | Of predicted "ups", how many were real |
| **Recall** | Higher = catches more moves | Of actual "ups", how many were caught |
| **Accuracy** | >0.65 | Overall directional correctness |

---

## Expected Performance (Indicative)

| Model | RMSE (Rs/MWh) | R² | AUC-ROC |
|---|---|---|---|
| Naive | ~500–700 | ~0.70 | ~0.60 |
| Ridge | ~400–600 | ~0.75 | ~0.65 |
| Random Forest | ~300–450 | ~0.82 | ~0.70 |
| XGBoost | ~250–380 | ~0.87 | ~0.75 |
| LightGBM | ~240–370 | ~0.88 | ~0.76 |
| LSTM | ~280–420 | ~0.85 | ~0.74 |

*Note: Actual values depend on market volatility in the test period.*
