"""
models/ridge_model.py — Ridge Regression Model (Regularized Linear Baseline)

Features: all time + lag + rolling columns.
Trains a separate ridge regressor per block OR one global model.
Default: one global model (fast, reasonable baseline).

Usage:
    python src/models/ridge_model.py                # weather features
    python src/models/ridge_model.py --no-weather   # ablation twin

Merged from the former *_no_weather_model.py twin (2026-11): the
--no-weather run stays byte-compatible with it — same
models_no_weather/ridge output dir, same ridge_no_weather_pipeline.pkl
artifact + metric label, so api.py / benchmark.py / backtest.py keep
consuming both variants unchanged.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import FEATURE_COLS, FEATURE_COLS_NO_WEATHER
from config import TARGET_COL as TARGET
from evaluate import compute_all_metrics, evaluate_by_segment, print_metrics_table

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "ridge")
MODELS_NO_WEATHER_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models_no_weather", "ridge")
PREDS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "predictions", "ridge")
PREDS_NO_WEATHER_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "predictions", "ridge_no_weather")


def load_data(use_weather=True):
    train = pd.read_parquet(
        os.path.join(DATA_DIR, f"training_features{'_no_weather' if not use_weather else ''}.parquet")
    )
    test = pd.read_parquet(
        os.path.join(DATA_DIR, f"holdout_features{'_no_weather' if not use_weather else ''}.parquet")
    )
    return train, test


def get_xy(df, features):
    df = df.dropna(subset=features + [TARGET])
    X = df[features].values.astype(float)
    y = df[TARGET].values.astype(float)
    return X, y, df


def run(use_weather=True):
    """Train Ridge; use_weather=False runs the no-weather ablation twin."""
    suffix = "" if use_weather else "_no_weather"
    features = FEATURE_COLS if use_weather else FEATURE_COLS_NO_WEATHER
    models_dir = MODELS_DIR if use_weather else MODELS_NO_WEATHER_DIR
    preds_dir = PREDS_DIR if use_weather else PREDS_NO_WEATHER_DIR

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    print(f"=== Ridge Regression Model{' (No Weather)' if suffix else ''} ===\n")
    train_df, test_df = load_data(use_weather)

    # Time-series val split: last 60 days of training
    cutoff = train_df["date"].max() - pd.Timedelta(days=60)
    val_df = train_df[train_df["date"] > cutoff]
    train_df = train_df[train_df["date"] <= cutoff]

    X_train, y_train, _ = get_xy(train_df, features)
    X_val, y_val, _ = get_xy(val_df, features)
    X_test, y_test, test_clean = get_xy(test_df, features)

    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # Build pipeline: scale -> ridge
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    pipe.fit(X_train, y_train)

    # Validate
    y_val_pred = pipe.predict(X_val)
    val_metrics = compute_all_metrics(y_val, y_val_pred)
    print("\nValidation metrics:")
    print_metrics_table(f"Ridge{' No Weather' if suffix else ''} (Val)", val_metrics)

    # Test (holdout)
    y_pred = pipe.predict(X_test)
    metrics = compute_all_metrics(y_test, y_pred)
    print_metrics_table(f"Ridge{' No Weather' if suffix else ''} (Holdout Test)", metrics)

    by_season = evaluate_by_segment(test_clean, y_pred, "season")
    by_hour = evaluate_by_segment(test_clean, y_pred, "hour_bucket")
    print("\nBy Season:\n", by_season.to_string(index=False))
    print("\nBy Hour Bucket:\n", by_hour.to_string(index=False))

    # Save
    joblib.dump(pipe, os.path.join(models_dir, f"ridge{suffix}_pipeline.pkl"))
    pd.DataFrame([{"model": f"ridge{suffix}", **metrics}]).to_csv(os.path.join(models_dir, "metrics.csv"), index=False)
    by_season.to_csv(os.path.join(models_dir, "metrics_by_season.csv"), index=False)
    by_hour.to_csv(os.path.join(models_dir, "metrics_by_hour.csv"), index=False)

    preds_df = test_clean[["date", "time_block", "block", TARGET]].copy()
    preds_df["predicted_mcp"] = y_pred
    preds_df.to_csv(os.path.join(preds_dir, "test_predictions.csv"), index=False)

    print(f"\n[OK] Model saved to {models_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the Ridge model (or its no-weather ablation twin).")
    parser.add_argument(
        "--no-weather", action="store_true", help="train the no-weather ablation variant into models_no_weather/"
    )
    run(use_weather=not parser.parse_args().no_weather)
