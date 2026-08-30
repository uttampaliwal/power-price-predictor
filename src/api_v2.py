"""
api_v2.py — FastAPI Serving Endpoint for Conformal Prediction

Provides REST API for:
1. Point forecasts with conformal prediction intervals
2. Model metadata and coverage statistics
3. Health check endpoint

Usage:
    uvicorn src.api_v2:app --host 0.0.0.0 --port 8000
    docker run -p 8000:8000 power-price-predictor
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import json
import time
from typing import List

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import DATA_PROCESSED_DIR, FEATURE_COLS, TARGET_COL
from probabilistic import AdaptiveConformal, SplitConformal

app = FastAPI(
    title="Power Price Predictor",
    description="Conformal prediction for Indian electricity prices (IEX DAM)",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for models and calibration
_models = {}
_cal_data = {}
_metrics = {}


class PredictionRequest(BaseModel):
    features: List[float]
    model_name: str = "xgboost"
    use_conformal: bool = True


class PredictionResponse(BaseModel):
    model: str
    point_forecast: float
    lower_bound: float
    upper_bound: float
    coverage_target: float
    interval_width: float
    timestamp: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    uptime_seconds: float


class MetricsResponse(BaseModel):
    model: str
    method: str
    picp: float
    pinaw: float
    winkler: float
    coverage_gap: float


_start_time = time.time()


@app.on_event("startup")
async def load_models():
    """Load pre-trained models and calibration data on startup."""
    global _models, _cal_data, _metrics

    print("Loading models and calibration data...")

    # Load training data for calibration
    train = pd.read_parquet(
        os.path.join(DATA_PROCESSED_DIR, "training_features.parquet")
    )
    train = train.dropna(subset=FEATURE_COLS + [TARGET_COL])

    # Split into train and calibration
    cal_size = int(len(train) * 0.20)
    X_train = train[FEATURE_COLS].values[:-cal_size]
    y_train = train[TARGET_COL].values[:-cal_size]
    X_cal = train[FEATURE_COLS].values[-cal_size:]
    y_cal = train[TARGET_COL].values[-cal_size:]

    # Train XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        n_jobs=-1,
        random_state=42,
    )
    xgb_model.fit(X_train, y_train, verbose=False)
    _models["xgboost"] = xgb_model

    # Train LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train)
    _models["lightgbm"] = lgb_model

    # Fit conformal calibration
    scp = SplitConformal(alpha=0.10)
    scp.fit(xgb_model, X_cal, y_cal)
    _cal_data["xgboost_scp"] = scp

    scp_lgb = SplitConformal(alpha=0.10)
    scp_lgb.fit(lgb_model, X_cal, y_cal)
    _cal_data["lightgbm_scp"] = scp_lgb

    acp = AdaptiveConformal(alpha=0.10, gamma=0.005, window=500)
    acp.fit(xgb_model, X_cal, y_cal)
    _cal_data["xgboost_acp"] = acp

    print(f"Loaded {len(_models)} models, {len(_cal_data)} calibration sets")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=list(_models.keys()),
        uptime_seconds=time.time() - _start_time,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate point forecast with conformal prediction interval."""
    if request.model_name not in _models:
        raise HTTPException(
            status_code=400, detail=f"Model '{request.model_name}' not found"
        )

    model = _models[request.model_name]
    features = np.array(request.features).reshape(1, -1)

    if features.shape[1] != len(FEATURE_COLS):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(FEATURE_COLS)} features, got {features.shape[1]}",
        )

    start = time.time()

    # Point forecast
    point_forecast = float(model.predict(features)[0])

    # Conformal interval
    if request.use_conformal:
        cal_key = f"{request.model_name}_scp"
        if cal_key in _cal_data:
            intervals = _cal_data[cal_key].predict(model, features)
            lower = float(intervals["lower"].values[0])
            upper = float(intervals["upper"].values[0])
        else:
            lower = point_forecast - 1000
            upper = point_forecast + 1000
    else:
        lower = point_forecast - 1000
        upper = point_forecast + 1000

    latency = time.time() - start

    return PredictionResponse(
        model=request.model_name,
        point_forecast=round(point_forecast, 2),
        lower_bound=round(lower, 2),
        upper_bound=round(upper, 2),
        coverage_target=0.90,
        interval_width=round(upper - lower, 2),
        timestamp=time.time(),
    )


@app.get("/metrics/{model_name}", response_model=MetricsResponse)
async def get_metrics(model_name: str, method: str = "split_conformal"):
    """Get conformal prediction metrics for a model."""
    # Load from results file
    results_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "results",
        "conformal",
        "conformal_results.json",
    )
    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="Metrics not found")

    with open(results_path) as f:
        all_results = json.load(f)

    if model_name not in all_results:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    if method not in all_results[model_name]:
        raise HTTPException(status_code=404, detail=f"Method '{method}' not found")

    m = all_results[model_name][method]
    return MetricsResponse(
        model=model_name,
        method=method,
        picp=m["PICP"],
        pinaw=m["PINAW"],
        winkler=m["Winkler"],
        coverage_gap=m["coverage_gap"],
    )


@app.get("/models")
async def list_models():
    """List available models and their metrics."""
    return {
        "models": list(_models.keys()),
        "calibration_sets": list(_cal_data.keys()),
        "feature_count": len(FEATURE_COLS),
        "features": FEATURE_COLS,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
