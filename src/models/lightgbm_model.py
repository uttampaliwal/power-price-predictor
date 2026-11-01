"""
models/lightgbm_model.py — LightGBM Gradient Boosting Model

Faster alternative to XGBoost, often comparable in accuracy.
Uses early stopping on time-series validation split.

Usage:
    python src/models/lightgbm_model.py                # weather features
    python src/models/lightgbm_model.py --no-weather   # ablation twin

Merged from the former *_no_weather_model.py twin (2026-11): the
--no-weather run stays byte-compatible with it — same
models_no_weather/lightgbm output dir, same lightgbm_no_weather.txt
artifact + metric label, so api.py / benchmark.py / backtest.py keep
consuming both variants unchanged.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lightgbm as lgb
import pandas as pd

from config import FEATURE_COLS, FEATURE_COLS_NO_WEATHER
from config import TARGET_COL as TARGET
from evaluate import compute_all_metrics, evaluate_by_segment, print_metrics_table

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "lightgbm")
MODELS_NO_WEATHER_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models_no_weather", "lightgbm")
PREDS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "predictions", "lightgbm")
PREDS_NO_WEATHER_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "predictions", "lightgbm_no_weather")


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
    """Train LightGBM; use_weather=False runs the no-weather ablation twin."""
    suffix = "" if use_weather else "_no_weather"
    features = FEATURE_COLS if use_weather else FEATURE_COLS_NO_WEATHER
    models_dir = MODELS_DIR if use_weather else MODELS_NO_WEATHER_DIR
    preds_dir = PREDS_DIR if use_weather else PREDS_NO_WEATHER_DIR

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    print(f"=== LightGBM Model{' (No Weather)' if suffix else ''} ===\n")
    train_df, test_df = load_data(use_weather)

    cutoff = train_df["date"].max() - pd.Timedelta(days=60)
    val_df = train_df[train_df["date"] > cutoff]
    train_df = train_df[train_df["date"] <= cutoff]

    X_train, y_train, _ = get_xy(train_df, features)
    X_val, y_val, _ = get_xy(val_df, features)
    X_test, y_test, test_clean = get_xy(test_df, features)

    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_lambda=1.0,
        reg_alpha=0.1,
        n_jobs=-1,
        random_state=42,
    )

    print("  Training LightGBM with early stopping...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=True),
            lgb.log_evaluation(100),
        ],
    )

    y_val_pred = model.predict(X_val)
    print_metrics_table(f"LightGBM{' No Weather' if suffix else ''} (Val)", compute_all_metrics(y_val, y_val_pred))

    y_pred = model.predict(X_test)
    metrics = compute_all_metrics(y_test, y_pred)
    print_metrics_table(f"LightGBM{' No Weather' if suffix else ''} (Holdout Test)", metrics)

    by_season = evaluate_by_segment(test_clean, y_pred, "season")
    by_hour = evaluate_by_segment(test_clean, y_pred, "hour_bucket")
    print("\nBy Season:\n", by_season.to_string(index=False))
    print("\nBy Hour Bucket:\n", by_hour.to_string(index=False))

    # Feature importance
    fi = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    print("\nTop 10 Feature Importances:\n", fi.head(10).to_string(index=False))
    fi.to_csv(os.path.join(models_dir, "feature_importance.csv"), index=False)

    # Save
    model.booster_.save_model(os.path.join(models_dir, f"lightgbm{suffix}.txt"))
    pd.DataFrame([{"model": f"lightgbm{suffix}", **metrics}]).to_csv(
        os.path.join(models_dir, "metrics.csv"), index=False
    )
    by_season.to_csv(os.path.join(models_dir, "metrics_by_season.csv"), index=False)
    by_hour.to_csv(os.path.join(models_dir, "metrics_by_hour.csv"), index=False)

    preds_df = test_clean[["date", "time_block", "block", TARGET]].copy()
    preds_df["predicted_mcp"] = y_pred
    preds_df.to_csv(os.path.join(preds_dir, "test_predictions.csv"), index=False)

    print(f"\n[OK] Model saved to {models_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the LightGBM model (or its no-weather ablation twin).")
    parser.add_argument(
        "--no-weather", action="store_true", help="train the no-weather ablation variant into models_no_weather/"
    )
    run(use_weather=not parser.parse_args().no_weather)
