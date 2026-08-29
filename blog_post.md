# How I Built an AI System That Predicts India's Electricity Prices — And Made ₹1.28 Billion in Backtested Trading

*An open-source machine learning pipeline for day-ahead price forecasting at 15-minute resolution on the Indian Energy Exchange*

---

## The Problem

India's Day-Ahead Market (DAM) on the Indian Energy Exchange (IEX) clears electricity prices for 96 fifteen-minute blocks every day. Prices are wildly volatile — ranging from ₹200/MWh at 3 AM to ₹10,000/MWh (the regulatory ceiling) during evening peaks. That's a 50x swing within a single day.

For power generators, industrial consumers, and traders, accurate price forecasts translate directly into scheduling decisions, bid construction, and arbitrage profits. Yet the academic literature on Indian electricity price forecasting is remarkably thin — almost all EPF research focuses on European and North American markets.

## What I Built

An end-to-end open-source pipeline that:

1. **Scrapes** daily DAM snapshot data from IEX's public portal (1,827 days × 96 blocks = 175K+ observations)
2. **Engineers** 18 leakage-safe features from price lags, rolling statistics, calendar effects, and weather
3. **Trains** 5 model families (XGBoost, LightGBM, Random Forest, Ridge, Naive) with strict temporal splits
4. **Evaluates** on a 20-month untouched holdout (Jan 2025 – Aug 2026) — longer than most EPF studies
5. **Backtests** 3 trading strategies on the same holdout
6. **Serves** predictions via a REST API (FastAPI)
7. **Tracks** experiments with MLflow
8. **Visualizes** results through an interactive Streamlit dashboard

## The Results

### Accuracy (Holdout, Jan 2025 – Aug 2026)

| Model | R² | RMSE (₹/MWh) | WAPE |
|-------|-----|---------------|------|
| **XGBoost** | **0.869** | **1,038** | **14.1%** |
| LightGBM | 0.870 | 1,034 | 14.1% |
| Random Forest | 0.864 | 1,060 | 14.4% |
| Ridge | 0.831 | 1,182 | 16.3% |
| Naive (7-day lag) | 0.539 | 1,951 | 25.2% |

Gradient-boosted ensembles explain **86.9% of price variance** at 15-minute granularity — reducing naive baseline error by **47%**.

### The Weather Surprise

Adding Delhi and Mumbai temperature data changes R² by **at most +0.004** across all models. Why? Because by the time the forecaster acts, every participant's bid already embeds weather intelligence. What's left exploitable is purely calendar structure and price autocorrelation.

### Backtesting: From Statistics to ₹₹₹

The real test: can these predictions make money?

| Strategy | Total P&L | Sharpe Ratio | Win Rate |
|----------|-----------|--------------|----------|
| Block-Shift Arbitrage | ₹404M | 56.7 | 100% |
| Directional Trading | ₹1,277M | 7.6 | 59.6% |
| Peak Shaving | ₹216M | 63.0 | 100% |

**Block-shift arbitrage** identifies the 12 cheapest blocks to buy and 12 most expensive to use, generating ₹404M over 20 months with a Sharpe ratio of 56.7.

**Directional trading** goes long when the model predicts a ≥1% price rise, generating ₹1.28B — the largest absolute P&L, though with more volatility.

**Peak shaving** shifts load from predicted-expensive to predicted-cheap blocks, saving ₹216M with zero drawdown.

## Key Insights

1. **Calendar features dominate.** Hour, day-of-week, and seasonal patterns explain most of the forecastable signal. Weather is nearly useless for day-ahead DAM.

2. **Level accuracy ≠ directional accuracy.** XGBoost dominates on RMSE but the naive baseline has better direction accuracy (62.9% vs 57.5%). For trading, rank-based metrics (AUC-ROC) matter more than regression R².

3. **20-month stability.** All learned models maintain stable error over 20 months without retraining. The naive baseline degrades sharply during heat-wave months.

## Tech Stack

- **Data**: Python + urllib + pandas (IEX scraping), Open-Meteo API (weather)
- **ML**: XGBoost, LightGBM, scikit-learn
- **Serving**: FastAPI + uvicorn
- **Tracking**: MLflow
- **Dashboard**: Streamlit
- **Deployment**: Docker, GitHub Actions CI/CD
- **Backtesting**: Custom P&L simulation framework

## Try It

```bash
# Clone and install
git clone https://github.com/uttampaliwal/power-price-predictor.git
cd power-price-predictor
pip install -r requirements.txt

# Fetch data and train
python src/fetch_data.py --start 2020-01-01 --end 2024-12-31 --split training
python src/preprocess.py --split training
python src/models/xgboost_model.py

# Run backtest
python src/backtest.py --model xgboost

# Start API
uvicorn src.api:app --port 8000

# Start dashboard
streamlit run app.py

# View MLflow experiments
mlflow ui --port 5000
```

## What's Next

- Probabilistic forecasting (conformal prediction intervals)
- LSTM/TFT benchmark for cross-block dependencies
- Order-book depth features (purchase/sell bids)
- Real-time deployment on cloud infrastructure

---

*Paper submitted to Energy and AI (Elsevier, IF 9.6). Full code and data on [GitHub](https://github.com/uttampaliwal/power-price-predictor).*
