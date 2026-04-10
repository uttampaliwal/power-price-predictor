import os
from datetime import date
import sys
sys.stdout.reconfigure(encoding="utf-8")

# -----------------------------------------------------------------------------
# Directory Structure
# -----------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR = os.path.join(BASE_DIR, "models")
PREDS_DIR = os.path.join(BASE_DIR, "predictions")

# Ensure critical directories exist
for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, PREDS_DIR]:
    os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(DATA_RAW_DIR, "training"), exist_ok=True)
    os.makedirs(os.path.join(DATA_RAW_DIR, "holdout"), exist_ok=True)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
TARGET_COL = "mcp_rs_per_mwh"

# Weather constants
CITIES = {
    "delhi": {"lat": 28.6139, "lon": 77.2090},
    "mumbai": {"lat": 19.0760, "lon": 72.8777}
}

# -----------------------------------------------------------------------------
# Pipeline Dates
# -----------------------------------------------------------------------------
DEFAULT_TRAIN_START = "2020-01-01"
# Training data is typically considered Jan 2020 - Jan 2025 (or whenever moved)
# This boundary is dynamic in the dashboard but these serve as defaults.
HOLDOUT_START_DATE = date(2025, 1, 1)

# -----------------------------------------------------------------------------
# Model List
# -----------------------------------------------------------------------------
MODEL_LIST = ["naive", "ridge", "random_forest", "xgboost", "lightgbm", "lstm", "prophet", "tft"]
