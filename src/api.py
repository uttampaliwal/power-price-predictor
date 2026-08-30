"""
api.py — FastAPI Model Serving Endpoint

Serves trained models for real-time day-ahead price predictions.
Supports single-date, date-range, and model-comparison queries.

Usage:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
    # or
    python src/api.py

Endpoints:
    GET  /health              — Health check
    GET  /models              — List available models
    POST /predict             — Predict MCP for a given date
    POST /predict/range       — Predict MCP for a date range
    POST /compare             — Compare predictions across models
    GET  /metrics/{model}     — Get model metrics
"""

import logging
import os
import sys
from datetime import date, timedelta
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    CITIES,
    DATA_RAW_DIR,
    FEATURE_COLS,
    FEATURE_COLS_NO_WEATHER,
    MODELS_DIR,
    MODELS_NO_WEATHER_DIR,
)

logger = logging.getLogger(__name__)

# ─── App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="IEX DAM Price Predictor API",
    description="Day-ahead electricity price forecasting for India's Day-Ahead Market at 15-minute resolution",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Lazy Data Loading ──────────────────────────────────────────────────
_history_cache: dict = {}


def _get_history(split: str = "holdout") -> pd.DataFrame:
    if split not in _history_cache:
        _history_cache[split] = _load_history(split)
    return _history_cache[split]


_model_cache: dict = {}


def _load_model(name: str, use_weather: bool = True):
    key = f"{name}_{'wx' if use_weather else 'nwx'}"
    if key in _model_cache:
        return _model_cache[key]

    if use_weather:
        base = os.path.join(MODELS_DIR, name)
    else:
        base = os.path.join(MODELS_NO_WEATHER_DIR, name)

    if name == "xgboost":
        import xgboost as xgb

        path = os.path.join(base, "xgboost.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"XGBoost model not found: {path}")
        model = xgb.XGBRegressor()
        model.load_model(path)
    elif name == "lightgbm":
        import lightgbm as lgb

        path = os.path.join(
            base, "lightgbm.txt" if use_weather else "lightgbm_no_weather.txt"
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"LightGBM model not found: {path}")
        model = lgb.LGBMRegressor()
        model.booster_ = lgb.Booster(model_file=path)
    elif name == "ridge":
        path = os.path.join(
            base,
            "ridge_pipeline.pkl" if use_weather else "ridge_no_weather_pipeline.pkl",
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Ridge model not found: {path}")
        model = joblib.load(path)
    elif name == "random_forest":
        path = os.path.join(
            base, "random_forest.pkl" if use_weather else "random_forest_no_weather.pkl"
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Random Forest model not found: {path}")
        model = joblib.load(path)
    else:
        raise ValueError(f"Unknown model: {name}")

    _model_cache[key] = model
    return model


# ─── Feature Construction ───────────────────────────────────────────────
def _load_history(split: str = "holdout") -> pd.DataFrame:
    raw_dir = os.path.join(DATA_RAW_DIR, split)
    csvs = sorted(
        [f for f in os.listdir(raw_dir) if f.endswith(".csv") and f.startswith("dam_")]
    )
    if not csvs:
        raise FileNotFoundError(f"No DAM CSVs found in {raw_dir}")
    dfs = [pd.read_csv(os.path.join(raw_dir, f), parse_dates=["date"]) for f in csvs]
    df = pd.concat(dfs, ignore_index=True)
    if "block" not in df.columns:
        df["block"] = df.groupby("date").cumcount()
    df = df.sort_values(["date", "time_block"]).reset_index(drop=True)
    return df


def _load_weather(split: str = "holdout") -> pd.DataFrame:
    path = os.path.join(DATA_RAW_DIR, f"weather_{split}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["time"])


def _build_features_for_date(
    target_date: date, history: pd.DataFrame, weather: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Build feature row for a single prediction date (all 96 blocks)."""
    import holidays

    BLOCKS_PER_DAY = 96
    season_map = {
        1: 0,
        2: 0,
        3: 1,
        4: 1,
        5: 1,
        6: 2,
        7: 2,
        8: 2,
        9: 2,
        10: 3,
        11: 3,
        12: 0,
    }
    hour_bucket = lambda h: (
        0 if h < 6 else (1 if h < 10 else (2 if h < 14 else (3 if h < 18 else 4)))
    )

    target_dt = pd.Timestamp(target_date)
    indian_holidays = holidays.India(years=target_date.year)

    rows = []
    for block in range(BLOCKS_PER_DAY):
        hour = block // 4
        lag1_date = target_date - timedelta(days=1)
        lag7_date = target_date - timedelta(days=7)

        def get_block_mcp(d, b):
            row = history[(history["date"].dt.date == d) & (history["block"] == b)]
            return float(row["mcp_rs_per_mwh"].values[0]) if len(row) else np.nan

        mcp_lag_1d = get_block_mcp(lag1_date, block)
        mcp_lag_7d = get_block_mcp(lag7_date, block)

        mask_7d = (
            (history["block"] == block)
            & (history["date"].dt.date >= target_date - timedelta(days=7))
            & (history["date"].dt.date < target_date)
        )
        recent_7 = history[mask_7d]["mcp_rs_per_mwh"]
        rolling_7d_mean = recent_7.mean() if len(recent_7) > 0 else np.nan
        rolling_7d_std = recent_7.std() if len(recent_7) > 1 else 0.0

        mask_30d = (
            (history["block"] == block)
            & (history["date"].dt.date >= target_date - timedelta(days=30))
            & (history["date"].dt.date < target_date)
        )
        recent_30 = history[mask_30d]["mcp_rs_per_mwh"]
        rolling_30d_mean = recent_30.mean() if len(recent_30) > 0 else np.nan
        rolling_30d_std = recent_30.std() if len(recent_30) > 1 else 0.0

        row = {
            "block": block,
            "hour": hour,
            "day_of_week": target_dt.dayofweek,
            "day_of_year": target_dt.dayofyear,
            "month": target_dt.month,
            "year": target_dt.year,
            "is_weekend": int(target_dt.dayofweek >= 5),
            "is_holiday": int(target_date in indian_holidays),
            "season": season_map.get(target_dt.month, 0),
            "hour_bucket": hour_bucket(hour),
            "mcp_lag_1d": mcp_lag_1d,
            "mcp_lag_7d": mcp_lag_7d,
            "mcp_rolling_7d_mean": rolling_7d_mean,
            "mcp_rolling_7d_std": rolling_7d_std,
            "mcp_rolling_30d_mean": rolling_30d_mean,
            "mcp_rolling_30d_std": rolling_30d_std,
        }

        if weather is not None:
            ts = target_dt + timedelta(hours=hour)
            for city in CITIES:
                city_weather = weather[weather["city"] == city]
                match = city_weather[city_weather["time"] == ts]
                row[f"{city}_apparent_temp"] = (
                    float(match["apparent_temp"].values[0]) if len(match) else np.nan
                )
        else:
            row["delhi_apparent_temp"] = np.nan
            row["mumbai_apparent_temp"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ─── Pydantic Models ────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    model_name: str = Field(
        "xgboost", description="Model name: xgboost, lightgbm, random_forest, ridge"
    )
    target_date: date = Field(..., description="Date to predict (YYYY-MM-DD)")
    use_weather: bool = Field(True, description="Use weather features")


class PredictRangeRequest(BaseModel):
    model_name: str = Field("xgboost")
    start_date: date = Field(..., description="Start date")
    end_date: date = Field(..., description="End date")
    use_weather: bool = Field(True)


class CompareRequest(BaseModel):
    models: list[str] = Field(["xgboost", "lightgbm", "ridge", "random_forest"])
    target_date: date
    use_weather: bool = True


class Prediction(BaseModel):
    block: int
    time_block: str
    predicted_mcp: float


class PredictResponse(BaseModel):
    model_name: str
    target_date: date
    predictions: list[Prediction]
    summary: dict


# ─── Endpoints ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/models")
def list_models():
    available = []
    for name in ["xgboost", "lightgbm", "random_forest", "ridge", "naive"]:
        for use_wx in [True, False]:
            try:
                _load_model(name, use_wx)
                available.append(
                    {
                        "name": name,
                        "use_weather": use_wx,
                        "type": "with-weather" if use_wx else "no-weather",
                    }
                )
            except (FileNotFoundError, ValueError):
                pass
    return {"models": available}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        model = _load_model(req.model_name, req.use_weather)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    history = _get_history("holdout")
    weather = _load_weather("holdout") if req.use_weather else None

    features = _build_features_for_date(req.target_date, history, weather)
    feature_cols = FEATURE_COLS if req.use_weather else FEATURE_COLS_NO_WEATHER
    X = features[feature_cols].values

    preds = model.predict(X)

    time_blocks = [
        f"{h:02d}:{m:02d} - {h:02d}:{m + 15:02d}"
        for h in range(24)
        for m in range(0, 60, 15)
    ]

    predictions = [
        Prediction(
            block=i, time_block=time_blocks[i], predicted_mcp=round(float(preds[i]), 2)
        )
        for i in range(len(preds))
    ]

    summary = {
        "mean_mcp": round(float(np.mean(preds)), 2),
        "min_mcp": round(float(np.min(preds)), 2),
        "max_mcp": round(float(np.max(preds)), 2),
        "std_mcp": round(float(np.std(preds)), 2),
        "peak_block": int(np.argmax(preds)),
        "off_peak_block": int(np.argmin(preds)),
    }

    return PredictResponse(
        model_name=req.model_name,
        target_date=req.target_date,
        predictions=predictions,
        summary=summary,
    )


@app.post("/predict/range")
def predict_range(req: PredictRangeRequest):
    if req.end_date < req.start_date:
        raise HTTPException(status_code=400, detail="end_date must be >= start_date")
    if (req.end_date - req.start_date).days > 365:
        raise HTTPException(status_code=400, detail="Date range limited to 365 days")

    try:
        model = _load_model(req.model_name, req.use_weather)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    history = _get_history("holdout")
    weather = _load_weather("holdout") if req.use_weather else None
    feature_cols = FEATURE_COLS if req.use_weather else FEATURE_COLS_NO_WEATHER

    all_preds = []
    current = req.start_date
    while current <= req.end_date:
        features = _build_features_for_date(current, history, weather)
        X = features[feature_cols].values
        preds = model.predict(X)

        time_blocks = [
            f"{h:02d}:{m:02d} - {h:02d}:{m + 15:02d}"
            for h in range(24)
            for m in range(0, 60, 15)
        ]

        for i in range(len(preds)):
            all_preds.append(
                {
                    "date": str(current),
                    "block": i,
                    "time_block": time_blocks[i],
                    "predicted_mcp": round(float(preds[i]), 2),
                }
            )
        current += timedelta(days=1)

    return {
        "model_name": req.model_name,
        "start_date": str(req.start_date),
        "end_date": str(req.end_date),
        "total_predictions": len(all_preds),
        "mean_mcp": round(float(np.mean([p["predicted_mcp"] for p in all_preds])), 2),
        "predictions": all_preds[:100],  # Return first 100 for brevity
    }


@app.post("/compare")
def compare(req: CompareRequest):
    results = {}
    for model_name in req.models:
        try:
            model = _load_model(model_name, req.use_weather)
            history = _get_history("holdout")
            weather = _load_weather("holdout") if req.use_weather else None
            feature_cols = FEATURE_COLS if req.use_weather else FEATURE_COLS_NO_WEATHER

            features = _build_features_for_date(req.target_date, history, weather)
            X = features[feature_cols].values
            preds = model.predict(X)

            results[model_name] = {
                "mean_mcp": round(float(np.mean(preds)), 2),
                "min_mcp": round(float(np.min(preds)), 2),
                "max_mcp": round(float(np.max(preds)), 2),
                "peak_block": int(np.argmax(preds)),
            }
        except (FileNotFoundError, ValueError) as e:
            results[model_name] = {"error": str(e)}

    return {
        "target_date": str(req.target_date),
        "use_weather": req.use_weather,
        "models": results,
    }


@app.get("/metrics/{model_name}")
def get_metrics(model_name: str):
    results = {}
    for split_dir, label in [
        (MODELS_DIR, "with_weather"),
        (MODELS_NO_WEATHER_DIR, "no_weather"),
    ]:
        metrics_path = os.path.join(split_dir, model_name, "metrics.csv")
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            results[label] = df.to_dict(orient="records")[0] if len(df) > 0 else {}

    if not results:
        raise HTTPException(
            status_code=404, detail=f"No metrics found for model: {model_name}"
        )

    return {"model_name": model_name, "metrics": results}


# ─── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
