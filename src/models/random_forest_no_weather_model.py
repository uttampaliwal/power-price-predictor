"""
models/random_forest_no_weather_model.py — Random Forest Model WITHOUT Weather Features

Same as random_forest_model.py but trained WITHOUT weather data.
Used for comparison to evaluate weather feature impact.

Usage:
    python src/models/random_forest_no_weather_model.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from evaluate import compute_all_metrics, evaluate_by_segment, print_metrics_table

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models_no_weather", "random_forest")
PREDS_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "predictions", "random_forest_no_weather")

# FEATURE_COLS WITHOUT weather features
FEATURE_COLS = [
    "block", "hour", "day_of_week", "day_of_year", "month", "year",
    "is_weekend", "is_holiday", "season", "hour_bucket",
    "mcp_lag_1d", "mcp_lag_7d",
    "mcp_rolling_7d_mean", "mcp_rolling_7d_std",
    "mcp_rolling_30d_mean", "mcp_rolling_30d_std",
]
TARGET = "mcp_rs_per_mwh"


def load_data():
    train = pd.read_parquet(os.path.join(DATA_DIR, "training_features_no_weather.parquet"))
    test  = pd.read_parquet(os.path.join(DATA_DIR, "holdout_features_no_weather.parquet"))
    return train, test


def get_xy(df):
    df = df.dropna(subset=FEATURE_COLS + [TARGET])
    X  = df[FEATURE_COLS].values.astype(float)
    y  = df[TARGET].values.astype(float)
    return X, y, df


def run():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PREDS_DIR,  exist_ok=True)

    print("=== Random Forest Model (No Weather) ===\n")
    train_df, test_df = load_data()

    cutoff   = train_df["date"].max() - pd.Timedelta(days=60)
    val_df   = train_df[train_df["date"] >  cutoff]
    train_df = train_df[train_df["date"] <= cutoff]

    X_train, y_train, _ = get_xy(train_df)
    X_val,   y_val,   _ = get_xy(val_df)
    X_test,  y_test, test_clean = get_xy(test_df)

    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    print("  Training Random Forest (this may take a few minutes)...")
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    print_metrics_table("Random Forest No Weather (Val)", compute_all_metrics(y_val, y_val_pred))

    y_pred = model.predict(X_test)
    metrics = compute_all_metrics(y_test, y_pred)
    print_metrics_table("Random Forest No Weather (Holdout Test)", metrics)

    by_season = evaluate_by_segment(test_clean, y_pred, "season")
    by_hour   = evaluate_by_segment(test_clean, y_pred, "hour_bucket")
    print("\nBy Season:\n", by_season.to_string(index=False))
    print("\nBy Hour Bucket:\n", by_hour.to_string(index=False))

    # Feature importance
    fi = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nTop 10 Feature Importances:\n", fi.head(10).to_string(index=False))
    fi.to_csv(os.path.join(MODELS_DIR, "feature_importance.csv"), index=False)

    # Save
    joblib.dump(model, os.path.join(MODELS_DIR, "random_forest_no_weather.pkl"))
    pd.DataFrame([{"model": "random_forest_no_weather", **metrics}]).to_csv(
        os.path.join(MODELS_DIR, "metrics.csv"), index=False)
    by_season.to_csv(os.path.join(MODELS_DIR, "metrics_by_season.csv"), index=False)
    by_hour.to_csv(os.path.join(MODELS_DIR, "metrics_by_hour.csv"), index=False)

    preds_df = test_clean[["date", "time_block", "block", TARGET]].copy()
    preds_df["predicted_mcp"] = y_pred
    preds_df.to_csv(os.path.join(PREDS_DIR, "test_predictions.csv"), index=False)

    print(f"\n[OK] Model saved to {MODELS_DIR}")


if __name__ == "__main__":
    run()