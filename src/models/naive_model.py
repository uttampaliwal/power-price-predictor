"""
models/naive_model.py — Naive Baseline: Last-Week Same Block

Prediction: For each time block (0–95) in the test day,
use the MCP value from exactly 7 days prior (same block).

This is the simplest possible baseline. Any useful model must beat this.

Usage:
    python src/models/naive_model.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
import pandas as pd
from evaluate import compute_all_metrics, evaluate_by_segment, print_metrics_table

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "naive")
PREDS_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "predictions", "naive")


def load_data():
    train_path = os.path.join(DATA_DIR, "training_features.parquet")
    test_path  = os.path.join(DATA_DIR, "holdout_features.parquet")
    train = pd.read_parquet(train_path)
    test  = pd.read_parquet(test_path)
    return train, test


def predict(test_df: pd.DataFrame) -> np.ndarray:
    """Use mcp_lag_7d as the naive prediction."""
    return test_df["mcp_lag_7d"].values


def run():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PREDS_DIR,  exist_ok=True)

    print("=== Naive Baseline Model ===\n")
    train_df, test_df = load_data()

    # Drop rows where lag is missing
    test_df = test_df.dropna(subset=["mcp_lag_7d"]).copy()

    y_true = test_df["mcp_rs_per_mwh"].values
    y_pred = predict(test_df)

    metrics = compute_all_metrics(y_true, y_pred)
    print_metrics_table("Naive (Last-Week Same Block)", metrics)

    # Segment breakdown
    by_season = evaluate_by_segment(test_df, y_pred, segment_col="season")
    by_hour   = evaluate_by_segment(test_df, y_pred, segment_col="hour_bucket")
    print("\nBy Season:\n", by_season.to_string(index=False))
    print("\nBy Hour Bucket:\n", by_hour.to_string(index=False))

    # Save results
    results = pd.DataFrame([{"model": "naive", **metrics}])
    results.to_csv(os.path.join(MODELS_DIR, "metrics.csv"), index=False)

    preds_df = test_df[["date", "time_block", "block", "mcp_rs_per_mwh"]].copy()
    preds_df["predicted_mcp"] = y_pred
    preds_df.to_csv(os.path.join(PREDS_DIR, "test_predictions.csv"), index=False)

    by_season.to_csv(os.path.join(MODELS_DIR, "metrics_by_season.csv"), index=False)
    by_hour.to_csv(os.path.join(MODELS_DIR, "metrics_by_hour.csv"), index=False)
    print(f"\n[OK] Results saved to {MODELS_DIR} and {PREDS_DIR}")


if __name__ == "__main__":
    run()
