import os
from datetime import date

# Directory Structure

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODELS_NO_WEATHER_DIR = os.path.join(BASE_DIR, "models_no_weather")
PREDS_DIR = os.path.join(BASE_DIR, "predictions")

for d in [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    MODELS_NO_WEATHER_DIR,
    PREDS_DIR,
]:
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(DATA_RAW_DIR, "training"), exist_ok=True)
    os.makedirs(os.path.join(DATA_RAW_DIR, "holdout"), exist_ok=True)

# Constants

TARGET_COL = "mcp_rs_per_mwh"

# Weather constants
CITIES = {
    "delhi": {"lat": 28.6139, "lon": 77.2090},
    "mumbai": {"lat": 19.0760, "lon": 72.8777},
}

# Weather feature columns to be excluded in no-weather mode
WEATHER_FEATURE_COLS = ["delhi_apparent_temp", "mumbai_apparent_temp"]

# Feature columns (single source of truth)
FEATURE_COLS = [
    "block",
    "hour",
    "day_of_week",
    "day_of_year",
    "month",
    "year",
    "is_weekend",
    "is_holiday",
    "season",
    "hour_bucket",
    "mcp_lag_1d",
    "mcp_lag_7d",
    "mcp_rolling_7d_mean",
    "mcp_rolling_7d_std",
    "mcp_rolling_30d_mean",
    "mcp_rolling_30d_std",
    "delhi_apparent_temp",
    "mumbai_apparent_temp",
]

FEATURE_COLS_NO_WEATHER = [c for c in FEATURE_COLS if c not in WEATHER_FEATURE_COLS]

# Pipeline Dates

DEFAULT_TRAIN_START = "2020-01-01"


HOLDOUT_START_DATE = date(2025, 1, 1)


# Model List

MODEL_LIST = ["naive", "ridge", "random_forest", "xgboost", "lightgbm", "lstm"]
